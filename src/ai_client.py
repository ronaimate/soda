"""OpenRouter HTTP client for structured AI responses."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import httpx

logger = logging.getLogger("soda.ai_client")


@dataclass
class CodingResult:
    success: bool
    blocked: Optional[str] = None
    files_written: int = 0
    output: str = ""
    error: Optional[str] = None


def parse_json_from_llm_output(text: str) -> dict:
    """Extract and parse JSON from LLM output (handles markdown fences)."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in response")
    return json.loads(cleaned[start : end + 1])


async def call_openrouter_json(
    api_key: str,
    model: str,
    system: str,
    user_prompt: str,
    retries: int = 3,
) -> dict:
    """Call OpenRouter chat completions and return parsed JSON."""
    model_str = model if "/" in model else f"openrouter/{model}"
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(timeout=180) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://soda.local",
                    },
                    json={
                        "model": model_str,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.2,
                    },
                )
        except Exception as e:
            last_error = str(e)
            logger.warning("OpenRouter attempt %s/%s failed: %s", attempt, retries, e)
            await asyncio.sleep(2 * attempt)
            continue

        if resp.status_code != 200:
            last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
            if 400 <= resp.status_code < 500:
                raise RuntimeError(last_error)
            await asyncio.sleep(2 * attempt)
            continue

        content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            last_error = "Empty content"
            await asyncio.sleep(2 * attempt)
            continue

        try:
            return parse_json_from_llm_output(content)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = str(e)
            logger.warning("JSON parse failed attempt %s/%s: %s", attempt, retries, e)
            await asyncio.sleep(2 * attempt)

    raise RuntimeError(f"OpenRouter failed after {retries} attempts: {last_error}")


def _safe_write_file(workdir: Path, rel_path: str, content: str) -> None:
    path = Path(rel_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe path: {rel_path}")
    target = (workdir / path).resolve()
    if not str(target).startswith(str(workdir.resolve())):
        raise ValueError(f"Path escapes workdir: {rel_path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


CODING_SYSTEM = (
    "You are a software developer. Implement ONLY the assigned task.\n"
    "Return ONLY valid JSON with this shape:\n"
    '{"files": [{"path": "relative/path", "content": "file contents"}], "blocked": null}\n'
    "If you cannot complete the task, set blocked to a short explanation string and files to [].\n"
    "Do not include markdown fences or any text outside the JSON object."
)


async def run_openrouter_coding_task(
    assignee: Any,
    prompt: str,
    workdir: Path,
    api_key: str,
) -> CodingResult:
    """Run coding via OpenRouter and write files to workdir."""
    model = getattr(assignee, "model", None) or "anthropic/claude-sonnet-4"
    try:
        result = await call_openrouter_json(
            api_key=api_key,
            model=model,
            system=CODING_SYSTEM,
            user_prompt=prompt,
        )
    except Exception as e:
        return CodingResult(success=False, error=str(e))

    blocked = result.get("blocked")
    if blocked:
        return CodingResult(success=False, blocked=str(blocked), output=json.dumps(result, indent=2))

    files = result.get("files") or []
    written = 0
    for item in files:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        content = item.get("content")
        if not path or content is None:
            continue
        try:
            _safe_write_file(workdir, str(path), str(content))
            written += 1
        except ValueError as e:
            logger.warning("Skipped unsafe path %s: %s", path, e)

    output = json.dumps(result, indent=2)
    if written == 0:
        return CodingResult(
            success=False,
            error="AI returned no writable files",
            output=output,
        )

    return CodingResult(success=True, files_written=written, output=output)


def architect_from_snapshot(snapshot: dict) -> SimpleNamespace:
    """Reconstruct a minimal architect object for background tasks."""
    return SimpleNamespace(**snapshot)
