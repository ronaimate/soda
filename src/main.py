import asyncio
import json
import logging
import os
import re
import subprocess
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Tuple, Any

import git
import httpx
from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .database import (
    GlobalSetting, Idea, Project, Task, TaskComment, TaskDependency, User,
    TaskGitState, UserDefaultSize, async_session, init_db, sa_select,
)
from .utils import (
    get_setting, get_opencode_api_key, write_opencode_auth, get_or_404,
    get_user_default_sizes, set_user_default_sizes, find_user_by_size, VALID_SIZES,
)
from .github_service import GitHubService
from .models import CallbackPayload, TaskMovePayload, CommentPayload
from .operations import get_operation_command, set_operation_command, get_all_operation_commands, generate_task_prompt
from .scaffold import get_scaffold_files
from .ai_client import run_openrouter_coding_task, architect_from_snapshot, parse_json_from_llm_output
from . import autopilot
from .git_utils import build_github_clone_url, resolve_github_username

MODEL_PRESETS = {
    "openrouter": {
        "planning": [
            {"id": "anthropic/claude-sonnet-4", "name": "Claude Sonnet 4 (recommended)"},
            {"id": "openai/gpt-4o", "name": "GPT-4o"},
            {"id": "google/gemini-2.5-pro-preview", "name": "Gemini 2.5 Pro"},
        ],
        "coding": [
            {"id": "deepseek/deepseek-chat", "name": "DeepSeek Chat (recommended)"},
            {"id": "anthropic/claude-3-haiku", "name": "Claude Haiku"},
            {"id": "google/gemini-2.0-flash-001", "name": "Gemini Flash"},
        ],
    },
    "opencode": {
        "planning": [
            {"id": "anthropic/claude-sonnet-4", "name": "Claude Sonnet 4 (recommended)"},
            {"id": "openai/gpt-4o", "name": "GPT-4o"},
            {"id": "google/gemini-2.5-pro-preview", "name": "Gemini 2.5 Pro"},
        ],
        "coding": [
            {"id": "deepseek/deepseek-chat", "name": "DeepSeek Chat (recommended)"},
            {"id": "anthropic/claude-3-haiku", "name": "Claude Haiku"},
            {"id": "google/gemini-2.0-flash-001", "name": "Gemini Flash"},
        ],
    },
    "minimax": {
        "planning": [
            {"id": "MiniMax-Text-01", "name": "MiniMax Text 01 (recommended)"},
        ],
        "coding": [
            {"id": "MiniMax-Text-01", "name": "MiniMax Text 01 (recommended)"},
        ],
    },
}

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
OPENCODE_AUTH = Path("/root/.local/share/opencode/auth.json")

# Architect project planning can be slow (especially via OpenCode CLI)
ARCHITECT_OPENCODE_TIMEOUT_SEC = 600
ARCHITECT_HTTP_TIMEOUT_SEC = 300

# In-memory process tracker: task_id -> (asyncio.subprocess.Process, stdout_fd, stderr_fd)
running_processes: dict[int, Tuple[asyncio.subprocess.Process, Any, Any]] = {}

# Idempotency guard: track tasks currently being post-processed
_processing_tasks: set[int] = set()

# Track active idea generation tasks
generation_tasks: set[int] = set()


def uses_auto_integrate(project) -> bool:
    """Autopilot: Architect review + hidden merge (auto and step run modes)."""
    return bool(project and getattr(project, "run_mode", None) in ("auto", "step"))


def parse_pending_questions(raw: Optional[str]) -> list[str]:
    """Parse pending_questions JSON; only return a list of question strings."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [q for q in parsed if isinstance(q, str) and q.strip()]
    return []


def format_generation_error(exc: Exception) -> str:
    """Turn an exception into a user-readable generation failure message."""
    if isinstance(exc, HTTPException):
        detail = exc.detail
        if isinstance(detail, str):
            msg = detail
        elif isinstance(detail, list):
            msg = "; ".join(str(d) for d in detail)
        else:
            msg = str(detail)
    else:
        msg = str(exc).strip() or type(exc).__name__

    lower = msg.lower()
    if "401" in msg or "unauthorized" in lower or "invalid api key" in lower:
        return (
            f"{msg}\n\n"
            "The API key may be wrong or expired. Check Settings and the architect user's provider."
        )
    if "404" in msg and "model" in lower:
        return (
            f"{msg}\n\n"
            "The selected model may not exist for this provider. Edit the architect user and pick another model."
        )
    if "429" in msg or "rate limit" in lower:
        return f"{msg}\n\nThe provider rate-limited the request. Wait a moment and try again."
    if "no ai provider enabled" in lower:
        return f"{msg}\n\nEnable at least one provider in Settings and add an API key."
    return msg


def parse_idea_state(idea: Idea) -> tuple[list[str], Optional[str]]:
    """Return (questions, generation_error) for an idea."""
    questions = parse_pending_questions(idea.pending_questions)
    error = idea.generation_error
    if not error and idea.pending_questions:
        try:
            parsed = json.loads(idea.pending_questions)
            if isinstance(parsed, dict) and parsed.get("error"):
                error = str(parsed["error"])
                questions = []
        except Exception:
            pass
    return questions, error


# Context for post-processing after AI completes
_post_process_ctx: dict[int, dict] = {}

# Track in-flight project expansion (idea/bug → new tasks)
expand_in_progress: set[int] = set()

# Centralized logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/soda.log')
    ]
)
logger = logging.getLogger("soda")
git_logger = logging.getLogger("soda.git")
watchdog_logger = logging.getLogger("soda.watchdog")


# ─── App factory ────────────────────────────────────────────────────

def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init_db()
        try:
            async with async_session() as session:
                if await get_setting("setup_complete", "false") == "true":
                    await sync_ai_users_from_settings(session)
                    await session.commit()
        except Exception as e:
            logger.warning(f"AI user sync on startup skipped: {e}")
        watchdog_task = asyncio.create_task(_watchdog_check())
        yield
        watchdog_task.cancel()

    app = FastAPI(title="Soda", lifespan=lifespan)

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.middleware("http")
    async def require_setup_middleware(request: Request, call_next):
        path = request.url.path
        if path.startswith(("/static", "/setup", "/settings", "/api/settings", "/api/models", "/api/callback")):
            return await call_next(request)
        if await get_setting("setup_complete", "false") != "true":
            if path.startswith("/api/"):
                return JSONResponse({"detail": "Setup required"}, status_code=403)
            return RedirectResponse("/setup", status_code=302)
        return await call_next(request)

    @app.get("/setup", response_class=HTMLResponse)
    async def setup_page(request: Request):
        setup_done = await get_setting("setup_complete", "false") == "true"
        planning = await get_setting("planning_model", "anthropic/claude-sonnet-4")
        coding = await get_setting("coding_model", "deepseek/deepseek-chat")
        primary = await get_setting("primary_ai_provider", "opencode")
        settings = {}
        async with async_session() as session:
            result = await session.execute(sa_select(GlobalSetting))
            settings = {row.key: row.value for row in result.scalars().all()}
        return templates.TemplateResponse("setup_wizard.html", {
            "request": request,
            "settings": settings,
            "default_planning": planning,
            "default_coding": coding,
            "primary_ai_provider": primary,
            "setup_complete": setup_done,
            "model_presets_json": json.dumps(MODEL_PRESETS),
        })

    @app.post("/setup")
    async def setup_submit(request: Request):
        data = await request.form()
        form = {k: str(v) for k, v in data.items()}
        for cb in ("provider_opencode_enabled", "provider_openrouter_enabled", "provider_minimax_enabled"):
            if cb not in form:
                form[cb] = "false"

        enabled_with_key = []
        for p in ("opencode", "openrouter", "minimax"):
            if form.get(f"provider_{p}_enabled") == "true":
                key = form.get(f"{p}_api_key", "").strip()
                if key:
                    enabled_with_key.append(p)
        if not enabled_with_key:
            raise HTTPException(400, "Enable at least one provider and enter its API key.")

        git_token = form.get("git_token", "").strip()
        if not git_token:
            raise HTTPException(400, "GitHub token is required.")
        gh_user = await resolve_github_username(git_token)
        if not gh_user:
            raise HTTPException(400, "Invalid GitHub token. Check the token has repo access.")

        primary = form.get("primary_ai_provider", "").strip()
        if primary not in enabled_with_key:
            primary = enabled_with_key[0]
        form["primary_ai_provider"] = primary
        form["git_username"] = gh_user
        form["setup_complete"] = "true"

        async with async_session() as session:
            await _apply_settings_form(session, form, preserve_empty_keys=False)
        return {"ok": True}

    # ── Template helpers ────────────────────────────────────────────

    def status_label(col: str) -> str:
        labels = {
            "backlog": "Backlog",
            "running": "Running",
            "blocked": "Blocked",
            "review": "Review",
            "done": "Done",
        }
        return labels.get(col, col)

    # ── Helper: write OpenCode auth ─────────────────────────────────

    def _write_opencode_auth(user: User) -> None:
        """Write AI user's API key and model to OpenCode auth.json."""
        import json as _json
        auth_dir = OPENCODE_AUTH.parent
        auth_dir.mkdir(parents=True, exist_ok=True)
        auth_data = {}
        if user.api_key:
            auth_data["apiKey"] = user.api_key
        if user.provider:
            auth_data["provider"] = user.provider
        else:
            # Infer provider from model name when not set on user
            if user.model and user.model.startswith("openrouter/"):
                auth_data["provider"] = "openrouter"
        if user.model:
            auth_data["model"] = user.model
        # If user has no API key, don't overwrite auth.json — let the
        # global API key handling in callers manage it.
        if not user.api_key:
            return
        with open(OPENCODE_AUTH, "w") as f:
            _json.dump(auth_data, f)

    # ── Helpers: get API key from settings ─────────────────

    async def _get_opencode_api_key() -> str:
        """Get OpenCode API key from global settings."""
        async with async_session() as session:
            result = await session.execute(
                sa_select(GlobalSetting).where(GlobalSetting.key == "opencode_api_key")
            )
            setting = result.scalar_one_or_none()
            return (setting.value or "").strip() if setting else ""

    async def _get_openrouter_api_key() -> str:
        """Get OpenRouter API key from global settings."""
        async with async_session() as session:
            result = await session.execute(
                sa_select(GlobalSetting).where(GlobalSetting.key == "openrouter_api_key")
            )
            setting = result.scalar_one_or_none()
            return (setting.value or "").strip() if setting else ""

    async def _get_minimax_api_key() -> str:
        """Get Minimax API key from global settings."""
        async with async_session() as session:
            result = await session.execute(
                sa_select(GlobalSetting).where(GlobalSetting.key == "minimax_api_key")
            )
            setting = result.scalar_one_or_none()
            return (setting.value or "").strip() if setting else ""

    PROVIDER_NAMES = {
        "opencode": "OpenCode",
        "openrouter": "OpenRouter",
        "minimax": "Minimax",
    }

    async def _get_api_key_for_provider(provider: str) -> str:
        if provider == "openrouter":
            return await _get_openrouter_api_key()
        if provider == "minimax":
            return await _get_minimax_api_key()
        return await _get_opencode_api_key()

    async def get_enabled_providers() -> list[dict]:
        """Return list of providers that are enabled in settings with their config."""
        providers = []
        for p in ["opencode", "openrouter", "minimax"]:
            enabled = (await get_setting(f"provider_{p}_enabled", "false")) == "true"
            if not enabled:
                continue
            key = await _get_api_key_for_provider(p)
            providers.append({
                "id": p,
                "name": PROVIDER_NAMES.get(p, p),
                "api_key": key,
                "configured": bool(key),
            })
        return providers

    async def get_first_enabled_provider(*, require_key: bool = False) -> str | None:
        """Return the first enabled provider, optionally requiring a configured API key."""
        primary = await get_setting("primary_ai_provider", "")
        if primary:
            enabled = (await get_setting(f"provider_{primary}_enabled", "false")) == "true"
            if enabled and (not require_key or await _get_api_key_for_provider(primary)):
                return primary
        for p in ["opencode", "openrouter", "minimax"]:
            enabled = (await get_setting(f"provider_{p}_enabled", "false")) == "true"
            if not enabled:
                continue
            if require_key and not await _get_api_key_for_provider(p):
                continue
            return p
        return None

    async def resolve_user_provider(user) -> str:
        """Use the user's provider if enabled and configured, else the primary/first working provider."""
        provider = getattr(user, "provider", None) if not isinstance(user, dict) else user.get("provider")
        if provider:
            enabled = (await get_setting(f"provider_{provider}_enabled", "false")) == "true"
            if enabled and await _get_api_key_for_provider(provider):
                return provider
        resolved = await get_first_enabled_provider(require_key=True)
        if not resolved:
            raise HTTPException(
                400,
                "No AI provider configured. Open Settings, enable a provider, and add an API key.",
            )
        return resolved

    async def sync_ai_users_from_settings(session, data: dict | None = None) -> None:
        """Align Architect/Coder with global provider + model settings."""
        primary = None
        planning = "anthropic/claude-sonnet-4"
        coding = "deepseek/deepseek-chat"
        if data:
            planning = data.get("planning_model", planning)
            coding = data.get("coding_model", coding)
            candidate = (data.get("primary_ai_provider") or "").strip()
            enabled = []
            for p in ("opencode", "openrouter", "minimax"):
                if data.get(f"provider_{p}_enabled") == "true" and (data.get(f"{p}_api_key") or "").strip():
                    enabled.append(p)
            if candidate in enabled:
                primary = candidate
            elif enabled:
                primary = enabled[0]
        else:
            primary = await get_first_enabled_provider(require_key=True)
            planning = await get_setting("planning_model", planning)
            coding = await get_setting("coding_model", coding)
        if primary:
            await _upsert_setting(session, "primary_ai_provider", primary)
        for name, model in (("Architect", planning), ("Coder", coding)):
            result = await session.execute(sa_select(User).where(User.name == name))
            user = result.scalar_one_or_none()
            if user and primary:
                user.provider = primary
                user.model = model
                user.api_key = None

    async def _upsert_setting(session, key: str, value: str) -> None:
        result = await session.execute(sa_select(GlobalSetting).where(GlobalSetting.key == key))
        row = result.scalar_one_or_none()
        if row:
            row.value = value
        else:
            session.add(GlobalSetting(key=key, value=value))

    async def _apply_settings_form(session, data: dict, *, preserve_empty_keys: bool = False) -> None:
        """Persist settings form fields and sync AI users."""
        merged = dict(data)
        if preserve_empty_keys:
            for p in ("opencode", "openrouter", "minimax"):
                key_name = f"{p}_api_key"
                if not (merged.get(key_name) or "").strip():
                    result = await session.execute(
                        sa_select(GlobalSetting).where(GlobalSetting.key == key_name)
                    )
                    row = result.scalar_one_or_none()
                    if row and row.value:
                        merged[key_name] = row.value
            if not (merged.get("git_token") or "").strip():
                result = await session.execute(
                    sa_select(GlobalSetting).where(GlobalSetting.key == "git_token")
                )
                row = result.scalar_one_or_none()
                if row and row.value:
                    merged["git_token"] = row.value
        for key, value in merged.items():
            await _upsert_setting(session, key, str(value))
        await sync_ai_users_from_settings(session, merged)
        await session.commit()
        opencode_key = (merged.get("opencode_api_key") or "").strip()
        if merged.get("provider_opencode_enabled") == "true" and opencode_key:
            write_opencode_auth(
                opencode_key,
                provider="opencode",
                model=merged.get("planning_model", "anthropic/claude-sonnet-4"),
            )

    async def _get_effective_api_key(user_provider: str = None) -> str:
        """Get the API key for the given provider, or the first enabled provider."""
        provider = user_provider or await get_first_enabled_provider()
        if not provider:
            return ""
        return await _get_api_key_for_provider(provider)

    def _architect_snapshot(user: User) -> dict:
        return {
            "id": user.id,
            "name": user.name,
            "provider": user.provider,
            "model": user.model,
            "api_key": user.api_key,
            "system_prompt": user.system_prompt,
        }

    _VALID_COMPLEXITIES = frozenset({"XS", "S", "M", "L", "XL"})

    def _normalize_complexity(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        upper = str(value).upper().strip()
        return upper if upper in _VALID_COMPLEXITIES else None

    # ── Helper: run execute command ─────────────────────────────────

    async def _run_execute_command(
        task: Task,
        assignee: User,
        comments: Optional[list] = None,
        depends_on_ids: Optional[list] = None,
    ) -> None:
        """Run the task_run operation command as a subprocess.
        The command is read from the op_cmd_task_run setting (not from the user).
        The prompt is generated dynamically from the task context.
        Clones the project repo, checks out main, provides context,
        then after AI completes: git commit/push, create PR, send callback."""
        # Get the operation command (NOT from user)
        operation_cmd = await get_operation_command("task_run")
        if not operation_cmd:
            logger.error(f"Task {task.id}: op_cmd_task_run is empty")
            try:
                async with async_session() as err_session:
                    err_session.add(TaskComment(
                        task_id=task.id,
                        author="Soda",
                        content="❌ Task run command is not configured.",
                    ))
                    await err_session.commit()
            except Exception:
                pass
            return

        # ── Register task as "starting" in running_processes so watchdog doesn't kill it ──
        from datetime import datetime as _dt_start
        running_processes[task.id] = (None, "starting", _dt_start.utcnow().isoformat())
        logger.info(f"Task {task.id}: registered as starting, launching AI run...")

        # Use provided comments/depends_on_ids, or fetch fresh
        if comments is None:
            async with async_session() as session:
                comments_result = await session.execute(
                    sa_select(TaskComment).where(TaskComment.task_id == task.id).order_by(TaskComment.created_at)
                )
                comments = [
                    {"author": c.author, "content": c.content, "created_at": str(c.created_at)}
                    for c in comments_result.scalars().all()
                ]
        if depends_on_ids is None:
            async with async_session() as session:
                deps_result = await session.execute(
                    sa_select(TaskDependency.depends_on_id).where(TaskDependency.task_id == task.id)
                )
                depends_on_ids = [row[0] for row in deps_result.all()]

        # Write auth.json: always use assignee's provider, with correct API key
        # Priority: assignee.api_key > global key matching assignee.provider
        user_provider = await resolve_user_provider(assignee)
        user_api_key = await _get_api_key_for_provider(user_provider)
        if not user_api_key:
            logger.error(f"Task {task.id}: no API key for provider {user_provider}")
            return
        logger.info(f"Task {task.id}: using provider={user_provider}, model={assignee.model}")

        # Write auth.json in OpenCode CLI expected format
        # Format: {"credentials": [{"provider": "<provider>", "key": "<api_key>"}]}
        # Also write model/provider at top-level for opencode run to pick up
        auth_dir = OPENCODE_AUTH.parent
        auth_dir.mkdir(parents=True, exist_ok=True)
        auth_data = {
            "credentials": [
                {"provider": user_provider or "opencode", "key": user_api_key}
            ]
        }
        if assignee.model:
            auth_data["model"] = assignee.model
        if user_provider and user_provider != "opencode":
            auth_data["provider"] = user_provider
        with open(OPENCODE_AUTH, "w") as f:
            json.dump(auth_data, f)
        logger.info(f"Task {task.id}: auth.json written — provider={user_provider}, model={assignee.model}, key_prefix={user_api_key[:8]}...")

        # Also update opencode.jsonc to set the correct model
        # OpenCode CLI reads model from opencode.jsonc which can override auth.json
        OPENCODE_CONFIG = Path("/root/.config/opencode/opencode.jsonc")
        try:
            import json as _json
            config_data = {}
            if OPENCODE_CONFIG.exists():
                with open(OPENCODE_CONFIG, "r") as _f:
                    config_data = _json.load(_f)
            # Update model to match the assignee's provider/model
            if assignee.model:
                config_data["model"] = assignee.model
            else:
                config_data.pop("model", None)
            if "$schema" not in config_data:
                config_data["$schema"] = "https://opencode.ai/config.json"
            OPENCODE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
            with open(OPENCODE_CONFIG, "w") as _f:
                _json.dump(config_data, _f, indent=2)
            logger.info(f"Task {task.id}: opencode.jsonc updated — model={assignee.model}")
        except Exception as config_err:
            logger.warning(f"Task {task.id}: failed to update opencode.jsonc: {config_err}")

        # Get project + settings for prompt generation
        async with async_session() as session:
            setting_res = await session.execute(
                sa_select(GlobalSetting).where(GlobalSetting.key == "callback_url")
            )
            setting = setting_res.scalar_one_or_none()
            callback_url = setting.value if setting else "http://localhost:8000/api/callback"

            project = await session.get(Project, task.project_id)
            project_name = project.name if project else ""
            repo_name = project.repo_name if project else ""
            repo_url = project.repo_url if project else ""

            git_username = await get_setting("git_username")
            git_token = await get_setting("git_token")
            default_branch = await get_setting("git_default_branch", "main")

        # Generate the full prompt dynamically (from operations module)
        full_prompt = await generate_task_prompt(task, project, comments, depends_on_ids)

        # Build authenticated repo URL (x-access-token avoids @ in email usernames breaking git)
        auth_repo_url = repo_url
        if repo_url and git_token:
            auth_repo_url = build_github_clone_url(repo_url, git_token)

        # Create task workdir and clone repo
        workdir_base = Path("/tmp/soda-task-workdirs")
        workdir_base.mkdir(parents=True, exist_ok=True)
        workdir = workdir_base / f"task-{task.id}"
        workdir.mkdir(parents=True, exist_ok=True)

        # Write prompt to file (prevents shell quoting issues with special chars)
        prompt_file = workdir / ".soda-prompt.txt"
        prompt_file.write_text(full_prompt)

        # ── Save the full prompt as a comment FIRST so user sees it immediately ──
        try:
            async with async_session() as prompt_session:
                model_info = ""
                if assignee.model:
                    model_info = f"\n\n**Model:** `{assignee.model}`"
                if assignee.provider:
                    model_info += f" (provider: `{assignee.provider}`)"
                from datetime import datetime as _dt
                start_msg = (
                    f"📋 **Prompt sent to AI:**{model_info}\n\n"
                    f"```\n{full_prompt}\n```\n\n"
                    f"---\n⏱️ AI run started at {_dt.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC. "
                    f"Watching process..."
                )
                prompt_session.add(TaskComment(
                    task_id=task.id,
                    author="Soda",
                    content=start_msg,
                ))
                await prompt_session.commit()
                logger.info(f"Task {task.id}: prompt comment saved to DB")
        except Exception as e:
            logger.error(f"Task {task.id}: failed to save prompt comment: {e}")
            try:
                async with async_session() as err_session:
                    err_session.add(TaskComment(
                        task_id=task.id,
                        author="Soda",
                        content=f"⚠️ Failed to save full prompt to DB: {e}",
                    ))
                    await err_session.commit()
            except Exception:
                pass

        # ── Now do the git clone (long timeout, may take 60-180s) ──
        if auth_repo_url:
            import shutil
            import subprocess as sp
            import tempfile
            clone_error = None
            try:
                # Clone into a temp dir first (workdir is not empty — has .soda-prompt.txt)
                clone_tmp = Path(tempfile.mkdtemp(prefix=f"clone-task-{task.id}-"))
                try:
                    sp.run(["git", "clone", "--branch", default_branch, "--single-branch", auth_repo_url, str(clone_tmp)],
                           check=True, capture_output=True, timeout=180)
                    # Move cloned contents into workdir (overwrite everything except .soda-*)
                    for item in clone_tmp.iterdir():
                        dest = workdir / item.name
                        if dest.name.startswith(".soda-"):
                            continue  # don't overwrite soda metadata
                        if dest.exists():
                            if dest.is_dir():
                                shutil.rmtree(dest)
                            else:
                                dest.unlink()
                        shutil.move(str(item), str(dest))
                finally:
                    if clone_tmp.exists():
                        shutil.rmtree(clone_tmp, ignore_errors=True)
            except Exception as e:
                clone_error = str(e)[:500]
                logger.warning(f"Task {task.id}: git clone failed: {clone_error}")
                running_processes.pop(task.id, None)
                _post_process_ctx.pop(task.id, None)
                try:
                    async with async_session() as err_session:
                        err_session.add(TaskComment(
                            task_id=task.id,
                            author="Soda",
                            content=(
                                f"❌ **Git clone failed** — cannot run task without the project repo.\n\n"
                                f"```\n{clone_error}\n```\n\n"
                                f"Check Settings → GitHub username (use your GitHub handle, not email) and token."
                            ),
                        ))
                        t = await err_session.get(Task, task.id)
                        if t:
                            t.board_column = "blocked"
                        await err_session.commit()
                except Exception:
                    pass
                return

        # Store context for post-processing (OpenRouter + OpenCode paths)
        _post_process_ctx[task.id] = {
            "callback_url": callback_url,
            "workdir": str(workdir),
            "auth_repo_url": auth_repo_url,
            "repo_url": repo_url,
            "repo_name": repo_name,
            "git_username": git_username,
            "git_token": git_token,
            "default_branch": default_branch,
            "project_id": task.project_id,
            "openrouter_sync": False,
        }

        if user_provider == "openrouter":
            _post_process_ctx[task.id]["openrouter_sync"] = True
            stdout_file = workdir / ".soda-stdout.log"
            stderr_file = workdir / ".soda-stderr.log"
            try:
                coding = await run_openrouter_coding_task(assignee, full_prompt, workdir, user_api_key)
                stdout_file.write_text(coding.output or "")
                if coding.error:
                    stderr_file.write_text(coding.error)
            except Exception as e:
                stderr_file.write_text(str(e))
                coding = None

            running_processes.pop(task.id, None)

            if coding and coding.blocked:
                async with async_session() as session:
                    t = await session.get(Task, task.id)
                    if t:
                        t.board_column = "blocked"
                        session.add(TaskComment(
                            task_id=task.id, author="Soda",
                            content=f"⚠️ AI reported it's blocked:\n\n{coding.blocked}",
                        ))
                        await session.commit()
                _post_process_ctx.pop(task.id, None)
                return

            if not coding or not coding.success:
                err = (coding.error if coding else "Unknown error") or "Coding failed"
                async with async_session() as session:
                    t = await session.get(Task, task.id)
                    if t:
                        t.board_column = "blocked"
                        session.add(TaskComment(
                            task_id=task.id, author="Soda",
                            content=f"⚠️ **OpenRouter coding failed:** {err[:1000]}",
                        ))
                        await session.commit()
                _post_process_ctx.pop(task.id, None)
                return

            await _post_process_task(task.id)
            return

        # Build env — use the same effective API key for subprocess
        env = os.environ.copy()
        if user_api_key:
            env["OPENCODE_API_KEY"] = user_api_key
            env["OPENROUTER_API_KEY"] = user_api_key

        # Resolve template variables in the OPERATION command (not user's command)
        cmd = operation_cmd
        cmd = cmd.replace("{{task.id}}", str(task.id))
        cmd = cmd.replace("{{task.title}}", task.title or "")
        cmd = cmd.replace("{{task.description}}", task.description or "")
        cmd = cmd.replace("{{task.complexity}}", task.complexity or "")
        cmd = cmd.replace("{{task.comments}}", json.dumps(comments))
        cmd = cmd.replace("{{project.name}}", project_name)
        cmd = cmd.replace("{{callback.url}}", callback_url)
        cmd = cmd.replace("{{task.workdir}}", str(workdir))
        cmd = cmd.replace("'{{task.prompt}}'", f'"$(cat {prompt_file})"')
        cmd = cmd.replace("{{task.prompt}}", str(prompt_file))

        # Run the operation command as subprocess
        stdout_file = workdir / ".soda-stdout.log"
        stderr_file = workdir / ".soda-stderr.log"
        stdout_fd = open(stdout_file, "w")
        stderr_fd = open(stderr_file, "w")
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=stdout_fd,
            stderr=stderr_fd,
            cwd=str(workdir),
            env=env,
        )
        running_processes[task.id] = (proc, stdout_fd, stderr_fd)

        async def _monitor_task_subprocess(tid: int, process, out_fd, err_fd):
            try:
                await process.wait()
            finally:
                try:
                    out_fd.close()
                    err_fd.close()
                except Exception:
                    pass
                running_processes.pop(tid, None)
                await _post_process_task(tid)

        asyncio.create_task(_monitor_task_subprocess(task.id, proc, stdout_fd, stderr_fd))

        _post_process_ctx[task.id].update({
            "callback_url": callback_url,
            "workdir": str(workdir),
            "auth_repo_url": auth_repo_url,
            "repo_url": repo_url,
            "repo_name": repo_name,
            "git_username": git_username,
            "git_token": git_token,
            "default_branch": default_branch,
            "project_id": task.project_id,
        })


    async def _ensure_task_review_workdir(
        task_id: int,
        workdir: Path,
        auth_repo_url: str,
        feature_branch: str,
        default_branch: str,
    ) -> Path:
        """Ensure task workdir exists on the feature branch (clone from GitHub if needed)."""
        import shutil

        workdir = Path(workdir)
        if auth_repo_url and (workdir / ".git").exists():
            try:
                repo = git.Repo(workdir)
                try:
                    repo.git.remote("set-url", "origin", auth_repo_url)
                except Exception:
                    pass
                try:
                    repo.git.fetch("origin", feature_branch, default_branch)
                except Exception:
                    pass
                try:
                    repo.git.checkout("-B", feature_branch, f"origin/{feature_branch}")
                except Exception:
                    repo.git.checkout("-B", feature_branch)
                return workdir
            except Exception as e:
                logger.warning(f"Task {task_id}: resetting workdir failed: {e}")

        if not auth_repo_url:
            return workdir

        workdir.parent.mkdir(parents=True, exist_ok=True)
        if workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)
        try:
            git.Repo.clone_from(auth_repo_url, workdir, branch=feature_branch)
        except Exception:
            repo = git.Repo.clone_from(auth_repo_url, workdir, branch=default_branch)
            try:
                repo.git.fetch("origin", feature_branch)
                repo.git.checkout("-B", feature_branch, f"origin/{feature_branch}")
            except Exception as e:
                logger.warning(f"Task {task_id}: checkout {feature_branch} failed: {e}")
        return workdir

    async def _latest_coder_output(task_id: int) -> str:
        async with async_session() as session:
            result = await session.execute(
                sa_select(TaskComment)
                .where(TaskComment.task_id == task_id, TaskComment.author == "AI")
                .order_by(TaskComment.created_at.desc())
            )
            comment = result.scalars().first()
        if not comment:
            return ""
        return (comment.content or "")[:10000]

    async def _run_git_in_workdir(workdir: Path, *args: str) -> tuple[str, int]:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            cwd=str(workdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        text = (stdout.decode() or stderr.decode() or "").strip()
        return text, proc.returncode or 0

    async def _collect_git_review_context(
        workdir: Path,
        default_branch: str,
        feature_branch: str,
    ) -> str:
        """Build diff/file context for architect review (works after commit)."""
        if not workdir.exists():
            return "(workdir missing — cannot inspect changes)"

        sections: list[str] = []
        base_refs = [
            default_branch,
            f"origin/{default_branch}",
            "HEAD~1",
        ]
        seen: set[str] = set()
        for base in base_refs:
            for title_suffix, args in (
                ("stat", ["diff", "--stat", f"{base}...HEAD"]),
                ("patch", ["diff", f"{base}...HEAD"]),
            ):
                text, _ = await _run_git_in_workdir(workdir, *args)
                if not text or text in seen:
                    continue
                seen.add(text)
                limit = 7000 if title_suffix == "patch" else 2000
                sections.append(f"## Diff vs {base} ({title_suffix})\n{text[:limit]}")

        for title, args in (
            ("Latest commit", ["show", "--stat", "--oneline", "-1", "HEAD"]),
            ("Latest commit patch", ["show", "HEAD", "--format="]),
            ("Uncommitted vs HEAD", ["diff", "--stat", "HEAD"]),
            ("Uncommitted changes", ["diff", "--stat"]),
        ):
            text, _ = await _run_git_in_workdir(workdir, *args)
            if text and text not in seen:
                seen.add(text)
                limit = 7000 if "patch" in title.lower() else 2000
                sections.append(f"## {title}\n{text[:limit]}")

        names, _ = await _run_git_in_workdir(
            workdir, "show", "--name-only", "--pretty=format:", "HEAD",
        )
        if names:
            for name in names.splitlines():
                name = name.strip()
                if not name or name.startswith(".") or "/." in name:
                    continue
                if not any(name.endswith(ext) for ext in (".js", ".html", ".css", ".ts", ".tsx", ".json")):
                    continue
                fp = workdir / name
                if fp.is_file():
                    try:
                        body = fp.read_text()[:5000]
                        sections.append(f"## File: {name}\n```\n{body}\n```")
                    except Exception:
                        pass

        if not sections:
            return "(no changes detected in git history or working tree)"
        return "\n\n".join(sections)[:18000]

    async def _architect_review_changes(
        task_id: int,
        workdir: Path,
        default_branch: str = "main",
        feature_branch: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Architect quick review before auto-merge. Returns (approved, notes)."""
        feature_branch = feature_branch or f"task-{task_id}"
        try:
            diff_context = await _collect_git_review_context(workdir, default_branch, feature_branch)
        except Exception as e:
            logger.warning(f"Task {task_id}: could not read diff for review: {e}")
            return True, ""

        async with async_session() as session:
            task = await session.get(Task, task_id)
            arch_r = await session.execute(sa_select(User).where(User.name == "Architect"))
            architect = arch_r.scalar_one_or_none()
            project = await session.get(Project, task.project_id) if task else None

        if not task or not architect:
            return True, ""

        coder_output = await _latest_coder_output(task_id)
        project_hint = ""
        if project and project.description:
            project_hint = f"\nProject context: {project.description}"

        coder_section = ""
        if coder_output:
            coder_section = f"\n\nCoder AI output (what was implemented):\n{coder_output}"

        prompt = f"""Review this completed coding task before it is integrated.

This is an isolated app repository for the project (HTML/CSS/JS or similar).
Do NOT review Soda's own Python backend files (database.py, main.py, etc.) — they are not part of this repo.
Judge only the files/changes shown below from the project repo.

Task: {task.title}
Description: {task.description or ''}{project_hint}

Changes to review:
{diff_context}{coder_section}

If the diff or file contents show the task requirements were implemented, approve.
Only reject if the changes are clearly wrong, incomplete, or unrelated.

Return ONLY valid JSON:
{{"approved": true, "notes": "brief OK message"}}
or
{{"approved": false, "notes": "what needs fixing"}}

No other text."""

        try:
            review = await _call_architect(architect, prompt)
            approved = bool(review.get("approved", True))
            notes = str(review.get("notes", "") or "")
            return approved, notes
        except Exception as e:
            logger.warning(f"Task {task_id}: architect review failed: {e}")
            return True, ""

    async def _auto_integrate_task(
        task_id: int,
        workdir: Path,
        repo_name: str,
        git_username: str,
        git_token: str,
        default_branch: str,
        feature_branch: Optional[str] = None,
        repo_url: str = "",
    ) -> None:
        """Architect review + merge to main, then mark done (autopilot only)."""
        feature_branch = feature_branch or f"task-{task_id}"
        auth_repo_url = build_github_clone_url(repo_url, git_token) if repo_url and git_token else ""
        if auth_repo_url:
            workdir = await _ensure_task_review_workdir(
                task_id, workdir, auth_repo_url, feature_branch, default_branch,
            )
            async with async_session() as session:
                git_state = await session.get(TaskGitState, task_id)
                if git_state:
                    git_state.workdir = str(workdir)
                    git_state.branch = feature_branch
                    git_state.repo = repo_name
                else:
                    session.add(TaskGitState(
                        task_id=task_id,
                        branch=feature_branch,
                        repo=repo_name,
                        workdir=str(workdir),
                    ))
                await session.commit()

        approved, review_notes = await _architect_review_changes(
            task_id, workdir, default_branch, feature_branch,
        )
        async with async_session() as session:
            task = await session.get(Task, task_id)
            if not task:
                return
            if not approved:
                task.board_column = "blocked"
                session.add(TaskComment(
                    task_id=task_id,
                    author="Soda",
                    content=f"⚠️ Needs changes before continuing:\n\n{review_notes or 'Review did not approve.'}",
                ))
                await session.commit()
                await autopilot.on_task_blocked(task_id)
                return

        gh = GitHubService(git_username, git_token)
        merge_result = await gh.merge_branch(repo_name, feature_branch, default_branch)
        async with async_session() as session:
            task = await session.get(Task, task_id)
            if not task:
                return
            if merge_result.get("status") == "merged":
                task.board_column = "done"
                done_msg = f"✓ **Done:** {task.title}"
                if review_notes:
                    done_msg += "\n\n✅ Verified."
                session.add(TaskComment(task_id=task_id, author="Soda", content=done_msg))
                await session.commit()
                await autopilot.on_task_completed(task_id)
                return
            task.board_column = "blocked"
            session.add(TaskComment(
                task_id=task_id,
                author="Soda",
                content=f"Could not integrate changes: {merge_result.get('message', 'merge failed')}",
            ))
            await session.commit()
            await autopilot.on_task_blocked(task_id)

    async def _recover_post_process_ctx(task_id: int) -> Optional[dict]:
        """Rebuild post-process context when the in-memory ctx was lost."""
        async with async_session() as session:
            task = await session.get(Task, task_id)
            if not task:
                return None
            project = await session.get(Project, task.project_id)
            if not project:
                return None
            git_username = await get_setting("git_username")
            git_token = await get_setting("git_token")
            default_branch = await get_setting("git_default_branch", "main")
            cb_res = await session.execute(
                sa_select(GlobalSetting).where(GlobalSetting.key == "callback_url")
            )
            cb = cb_res.scalar_one_or_none()

        repo_url = project.repo_url or ""
        auth_repo_url = build_github_clone_url(repo_url, git_token) if repo_url and git_token else ""
        workdir = Path(f"/tmp/soda-task-workdirs/task-{task_id}")
        return {
            "callback_url": cb.value if cb else "http://localhost:8000/api/callback",
            "workdir": str(workdir),
            "auth_repo_url": auth_repo_url,
            "repo_url": repo_url,
            "repo_name": project.repo_name or "",
            "git_username": git_username,
            "git_token": git_token,
            "default_branch": default_branch,
            "project_id": project.id,
            "openrouter_sync": False,
        }

    async def _task_has_ai_output(task_id: int) -> bool:
        workdir = Path(f"/tmp/soda-task-workdirs/task-{task_id}")
        stdout = workdir / ".soda-stdout.log"
        if stdout.exists():
            try:
                if stdout.read_text().strip():
                    return True
            except Exception:
                pass
        async with async_session() as session:
            result = await session.execute(
                sa_select(TaskComment.id)
                .where(TaskComment.task_id == task_id, TaskComment.author == "AI")
                .limit(1)
            )
            return result.scalar_one_or_none() is not None

    async def _post_process_task(task_id: int) -> None:
        """After AI process completes: git commit/push, create PR, update task status."""
        # Idempotency guard: prevent duplicate processing
        if task_id in _processing_tasks:
            logger.info(f"Task {task_id} already being processed, skipping")
            return
        
        _processing_tasks.add(task_id)
        
        try:
            ctx = _post_process_ctx.pop(task_id, None)
            if not ctx:
                ctx = await _recover_post_process_ctx(task_id)
            if not ctx:
                logger.warning(f"Task {task_id}: No context found for post-processing")
                return

            workdir = Path(ctx["workdir"])
            auth_repo_url = ctx["auth_repo_url"]
            repo_name = ctx["repo_name"]
            git_username = ctx["git_username"]
            git_token = ctx["git_token"]
            default_branch = ctx["default_branch"]
            openrouter_sync = ctx.get("openrouter_sync", False)

            # Check for AI blocking message in stdout
            # The prompt tells AI: "If you cannot complete the task, describe what is blocking you as the last line of your output"
            blocked_reason = ""
            stdout_file = workdir / ".soda-stdout.log"
            if stdout_file.exists() and not openrouter_sync:
                try:
                    stdout_text = stdout_file.read_text().strip()
                    if stdout_text:
                        lines = stdout_text.split("\n")
                        last_line = lines[-1].strip().lower() if lines else ""
                        block_phrases = [
                            "i cannot complete", "i am stuck", "i need help",
                            "i am unable to", "i'm stuck", "i'm blocked",
                            "cannot complete this", "blocked:",
                        ]
                        if any(kw in last_line for kw in block_phrases):
                            blocked_reason = "\n".join(lines[-5:])
                except Exception:
                    pass

            if blocked_reason:
                # AI reported it's blocked
                async with async_session() as session:
                    task = await session.get(Task, task_id)
                    if task:
                        task.board_column = "blocked"
                        session.add(TaskComment(task_id=task_id, author="Soda",
                            content=f"⚠️ AI reported it's blocked:\n\n{blocked_reason}"))
                        await session.commit()
                        await autopilot.on_task_blocked(task_id)
                return

            # Check for execution errors (shell quoting, command not found, etc.)
            stdout_text = ""
            stderr_text = ""
            if stdout_file.exists():
                try:
                    stdout_text = stdout_file.read_text().strip()
                except Exception:
                    pass
            stderr_file = workdir / ".soda-stderr.log"
            if stderr_file.exists():
                try:
                    stderr_text = stderr_file.read_text().strip()
                except Exception:
                    pass

            if not stdout_text and not openrouter_sync:
                # No AI output at all — indicates execution error
                error_detail = stderr_text[:2000] if stderr_text else "Unknown error (no output)"
                async with async_session() as session:
                    task = await session.get(Task, task_id)
                    if task:
                        task.board_column = "blocked"
                        session.add(TaskComment(task_id=task_id, author="Soda",
                            content=f"⚠️ **Execution error:** OpenCode did not produce any output.\n\n```\n{error_detail}\n```"))
                        await session.commit()
                        await autopilot.on_task_blocked(task_id)
                return

            async with async_session() as session:
                task_row = await session.get(Task, task_id)
                project_row = await session.get(Project, task_row.project_id) if task_row else None
                auto_integrate = uses_auto_integrate(project_row)

            pr_url, git_error, no_changes = await _git_commit_push_and_pr(
                task_id=task_id,
                workdir=workdir,
                auth_repo_url=auth_repo_url,
                repo_name=repo_name,
                username=git_username,
                token=git_token,
                default_branch=default_branch,
                create_pr=not auto_integrate,
            )

            async with async_session() as session:
                task = await session.get(Task, task_id)
                if not task:
                    return
                feature_branch = f"task-{task_id}"

                if auto_integrate:
                    if no_changes and not git_error:
                        task.board_column = "done"
                        session.add(TaskComment(
                            task_id=task_id,
                            author="Soda",
                            content=f"✓ **Done:** {task.title}\n\n(No file changes — already up to date.)",
                        ))
                        await session.commit()
                        await autopilot.on_task_completed(task_id)
                        return
                    if git_error and pr_url is None:
                        task.board_column = "blocked"
                        session.add(TaskComment(
                            task_id=task_id,
                            author="Soda",
                            content=f"⚠️ **Could not save changes to GitHub**\n\n{git_error}",
                        ))
                        await session.commit()
                        await autopilot.on_task_blocked(task_id)
                        return
                    if pr_url is not None and not git_error:
                        await session.commit()
                        await _auto_integrate_task(
                            task_id, workdir, repo_name, git_username, git_token,
                            default_branch, feature_branch, ctx.get("repo_url", ""),
                        )
                        return
                    task.board_column = "blocked"
                    session.add(TaskComment(
                        task_id=task_id,
                        author="Soda",
                        content="⚠️ **Could not save changes to GitHub**",
                    ))
                    await session.commit()
                    await autopilot.on_task_blocked(task_id)
                    return

                if pr_url:
                    task.board_column = "review"
                    session.add(TaskComment(task_id=task_id, author="Soda",
                        content=f"📦 **Pull Request created:** {pr_url}"))
                    git_state = await session.get(TaskGitState, task_id)
                    if git_state:
                        git_state.branch = feature_branch
                        git_state.repo = repo_name
                        git_state.workdir = str(workdir)
                    else:
                        session.add(TaskGitState(
                            task_id=task_id,
                            branch=feature_branch,
                            repo=repo_name,
                            workdir=str(workdir),
                        ))
                elif git_username and git_token:
                    task.board_column = "blocked"
                    detail = git_error or "No file changes could be committed."
                    session.add(TaskComment(task_id=task_id, author="Soda",
                        content=f"⚠️ **Could not save changes to GitHub**\n\n{detail}"))
                    await autopilot.on_task_blocked(task_id)
                else:
                    task.board_column = "blocked"
                    session.add(TaskComment(task_id=task_id, author="Soda",
                        content="⚠️ GitHub auth not configured."))
                    await autopilot.on_task_blocked(task_id)
                await session.commit()
        finally:
            _processing_tasks.discard(task_id)
            logger.info(f"Task {task_id}: Post-processing completed")


    async def _git_commit_push_and_pr(
        task_id: int,
        workdir: Path,
        auth_repo_url: str,
        repo_name: str,
        username: str,
        token: str,
        default_branch: str,
        create_pr: bool = True,
    ) -> tuple[Optional[str], str, bool]:
        """Commit and push task work. Returns (pr_url_or_empty, error_message, no_changes)."""
        if not username or not token or not auth_repo_url:
            return None, "GitHub credentials not configured.", False

        import shutil

        feature_branch = f"task-{task_id}"
        workdir_path = Path(workdir)

        try:
            if (workdir_path / ".git").exists():
                repo = git.Repo(workdir_path)
            else:
                repo_workdir = Path(f"/tmp/soda-pr-workdirs/task-{task_id}")
                repo_workdir.parent.mkdir(parents=True, exist_ok=True)
                if repo_workdir.exists():
                    shutil.rmtree(repo_workdir)
                repo = git.Repo.clone_from(auth_repo_url, repo_workdir)
                try:
                    repo.git.checkout(default_branch)
                except Exception:
                    repo.git.checkout("-b", default_branch)
                for item in workdir_path.iterdir():
                    if item.name.startswith(".soda-") or item.name == ".git":
                        continue
                    dest = repo_workdir / item.name
                    if item.is_dir():
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)

            try:
                repo.git.remote("set-url", "origin", auth_repo_url)
            except Exception:
                try:
                    repo.create_remote("origin", auth_repo_url)
                except Exception:
                    pass

            repo.git.checkout("-B", feature_branch)
            repo.git.add(A=True)
            status = (repo.git.status("--porcelain") or "").strip()
            if not status:
                logger.info(f"Task {task_id}: no file changes to commit")
                return "", "", True

            repo.index.commit(f"feat: task {task_id}")
            repo.git.push("-u", "origin", feature_branch)

            if not create_pr:
                return "", "", False

            gh_service = GitHubService(username, token)
            pr_result = await gh_service.create_pull_request(
                repo_name=repo_name,
                title=f"Task {task_id}",
                head=feature_branch,
                base=default_branch,
                body=f"Task {task_id} — created by Soda",
            )
            if pr_result["success"]:
                logger.info(f"PR created: {pr_result['pr_url']}")
                return pr_result["pr_url"], "", False
            return None, pr_result.get("error") or "Pull request creation failed.", False
        except Exception as e:
            err = str(e)
            logger.error(f"git/pr error for task {task_id}: {err}")
            return None, err, False

    # ── Frontend pages ──────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index_page(request: Request):
        async with async_session() as session:
            result = await session.execute(sa_select(Project).order_by(Project.name))
            projects = result.scalars().all()
        return templates.TemplateResponse(
            "index.html", {"request": request, "projects": projects, "current_project": None}
        )

    @app.get("/ideas", response_class=HTMLResponse)
    async def ideas_page(request: Request):
        async with async_session() as session:
            result = await session.execute(
                sa_select(Idea).order_by(Idea.created_at.desc())
            )
            ideas_raw = result.scalars().all()
            result = await session.execute(
                sa_select(User).where(User.type == "ai").order_by(User.name)
            )
            ai_users = result.scalars().all()
            result = await session.execute(sa_select(Project).order_by(Project.name))
            projects = result.scalars().all()
            proj_by_idea = {p.source_idea_id: p for p in projects if p.source_idea_id}

        ideas = []
        for i in ideas_raw:
            questions, generation_error = parse_idea_state(i)
            from types import SimpleNamespace
            iv = SimpleNamespace()
            for attr in ["id", "title", "description", "system_prompt", "architect_user_id", "status"]:
                setattr(iv, attr, getattr(i, attr))
            iv.questions = questions
            iv.generation_error = generation_error
            proj = proj_by_idea.get(i.id)
            iv.project_id = proj.id if proj else None
            ideas.append(iv)

        return templates.TemplateResponse(
            "ideas.html",
            {
                "request": request,
                "ideas": ideas,
                "ai_users": ai_users,
                "projects": projects,
                "show_advanced_nav": False,
            },
        )

    @app.get("/project/{project_id}", response_class=HTMLResponse)
    async def board_page(request: Request, project_id: int):
        async with async_session() as session:
            project = await session.get(Project, project_id)
            if not project:
                raise HTTPException(404, "Project not found")
            result = await session.execute(
                sa_select(Task)
                .where(Task.project_id == project_id)
                .order_by(Task.position, Task.created_at)
            )
            tasks = result.scalars().all()
            result = await session.execute(sa_select(User).order_by(User.name))
            users = result.scalars().all()
            result = await session.execute(sa_select(Project).order_by(Project.name))
            projects = result.scalars().all()

            # Fetch comments per task
            comments_map = {}
            # Fetch dependencies per task
            deps_map = {}
            task_ids = [t.id for t in tasks]
            task_by_id = {t.id: t for t in tasks}
            done_ids = {t.id for t in tasks if t.board_column == "done"}
            unmet_ids = set()
            if task_ids:
                dep_result = await session.execute(
                    sa_select(TaskDependency.task_id, TaskDependency.depends_on_id)
                    .where(TaskDependency.task_id.in_(task_ids))
                )
                for task_id, dep_id in dep_result.all():
                    deps_map.setdefault(task_id, []).append(dep_id)

                for tid, dep_ids in deps_map.items():
                    if any(d not in done_ids for d in dep_ids):
                        unmet_ids.add(tid)

            for t in tasks:
                cr = await session.execute(
                    sa_select(TaskComment).where(TaskComment.task_id == t.id).order_by(TaskComment.created_at)
                )
                comments_map[t.id] = cr.scalars().all()
                t.has_unmet_deps = t.id in unmet_ids
                t.is_running = t.board_column == "running"
                if t.has_unmet_deps:
                    dep_ids = deps_map.get(t.id, [])
                    t.unmet_dep_titles = [
                        task_by_id[d].title for d in dep_ids
                        if d in task_by_id and d not in done_ids
                    ]
                else:
                    t.unmet_dep_titles = []
                t.block_reason = None
                for c in reversed(comments_map[t.id]):
                    if c.author == "Soda" and ("⚠️" in c.content or "❌" in c.content):
                        plain = c.content.replace("**", "").split("\n")[0][:120]
                        t.block_reason = plain
                        break

        pipeline_status = await autopilot.get_pipeline_status(project_id)
        done_count = sum(1 for t in tasks if t.board_column == "done")
        total_count = len(tasks)

        return templates.TemplateResponse(
            "board.html",
            {
                "request": request,
                "project": project,
                "tasks": tasks,
                "users": users,
                "projects": projects,
                "comments_map": {str(k): v for k, v in comments_map.items()},
                "status_label": status_label,
                "pipeline": pipeline_status,
                "done_count": done_count,
                "total_count": total_count,
                "show_advanced_nav": bool(project.advanced_mode),
                "auto_integrate": uses_auto_integrate(project),
            },
        )

    @app.get("/users", response_class=HTMLResponse)
    async def users_page(request: Request):
        async with async_session() as session:
            result = await session.execute(sa_select(User).order_by(User.name))
            users = result.scalars().all()
            result = await session.execute(sa_select(Project).order_by(Project.name))
            projects = result.scalars().all()
        return templates.TemplateResponse(
            "users.html",
            {"request": request, "users": users, "projects": projects, "show_advanced_nav": True},
        )

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        async with async_session() as session:
            result = await session.execute(sa_select(GlobalSetting))
            settings = {row.key: row.value for row in result.scalars().all()}
            result = await session.execute(sa_select(Project).order_by(Project.name))
            projects = result.scalars().all()
        return templates.TemplateResponse(
            "settings.html",
            {
                "request": request,
                "settings": settings,
                "projects": projects,
                "show_advanced_nav": False,
                "model_presets_json": json.dumps(MODEL_PRESETS),
            },
        )

    # ── API: Projects ──────────────────────────────────────────────

    @app.get("/api/projects")
    async def list_projects():
        async with async_session() as session:
            result = await session.execute(sa_select(Project).order_by(Project.name))
            projects = result.scalars().all()
            return [
                {"id": p.id, "name": p.name, "description": p.description, "created_at": str(p.created_at)}
                for p in projects
            ]

    @app.post("/api/projects")
    async def create_project(name: str = Form(...), description: str = Form("")):
        async with async_session() as session:
            project = Project(name=name, description=description)
            session.add(project)
            await session.commit()
            await session.refresh(project)
            return {"id": project.id, "name": project.name}

    @app.get("/api/projects/{project_id}")
    async def get_project(project_id: int):
        async with async_session() as session:
            project = await session.get(Project, project_id)
            if not project:
                raise HTTPException(404)
            return {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "merger_user_id": project.merger_user_id,
            }

    @app.patch("/api/projects/{project_id}")
    async def update_project(project_id: int, name: Optional[str] = Form(None), description: Optional[str] = Form(None), merger_user_id: Optional[int] = Form(None)):
        async with async_session() as session:
            project = await session.get(Project, project_id)
            if not project:
                raise HTTPException(404)
            if name:
                project.name = name
            if description is not None:
                project.description = description
            if merger_user_id is not None:
                project.merger_user_id = merger_user_id if merger_user_id > 0 else None
            await session.commit()
            return {"ok": True}

    # ── API: Tasks ─────────────────────────────────────────────────

    @app.get("/api/projects/{project_id}/tasks")
    async def list_tasks(project_id: int):
        async with async_session() as session:
            result = await session.execute(
                sa_select(Task).where(Task.project_id == project_id).order_by(Task.position, Task.created_at)
            )
            tasks = result.scalars().all()
            return [
                {
                    "id": t.id,
                    "title": t.title,
                    "description": t.description,
                    "column": t.board_column,
                    "assignee_id": t.assignee_id,
                    "complexity": t.complexity,
                    "position": t.position,
                }
                for t in tasks
            ]

    @app.post("/api/projects/{project_id}/tasks")
    async def create_task(
        project_id: int,
        title: str = Form(...),
        description: str = Form(""),
        assignee_id: Optional[int] = Form(None),
        complexity: Optional[str] = Form(None),
        is_bug: bool = Form(False),
        questions: str = Form(""),  # JSON array of strings, only if is_bug
    ):
        async with async_session() as session:
            project = await get_or_404(session, Project, project_id, "Project")
            # Get max position
            result = await session.execute(
                sa_select(Task).where(Task.project_id == project_id).order_by(Task.position.desc()).limit(1)
            )
            last = result.scalar_one_or_none()
            pos = (last.position + 1) if last else 0

            # Auto-assign based on complexity if no assignee specified
            final_assignee_id = assignee_id
            if final_assignee_id is None and complexity:
                # Use the new find_user_by_size that uses task_types ARRAY
                from .utils import find_user_by_size
                assigned_user_id = await find_user_by_size(session, complexity.lower())
                if assigned_user_id:
                    final_assignee_id = assigned_user_id
                    logger.info(f"Task '{title}' auto-assigned to user {assigned_user_id} based on size {complexity}")

            task = Task(
                project_id=project_id,
                title=title,
                description=description,
                assignee_id=final_assignee_id,
                complexity=complexity,
                position=pos,
                is_bug=is_bug,
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)

            # If bug with questions, save them as a comment
            if is_bug and questions:
                import json as _json
                try:
                    qlist = _json.loads(questions)
                    if isinstance(qlist, list) and qlist:
                        content = "🐛 Bug report questions:\n" + "\n".join(f"- {q}" for q in qlist)
                        session.add(TaskComment(task_id=task.id, author="user", content=content))
                        await session.commit()
                except Exception:
                    pass

            return {"id": task.id, "title": task.title, "column": task.board_column, "assignee_id": task.assignee_id, "is_bug": task.is_bug}

    @app.get("/api/tasks/{task_id}")
    async def get_task(task_id: int):
        async with async_session() as session:
            task = await get_or_404(session, Task, task_id)
            result = await session.execute(
                sa_select(TaskComment).where(TaskComment.task_id == task_id).order_by(TaskComment.created_at)
            )
            comments = [
                {"id": c.id, "author": c.author, "content": c.content, "created_at": str(c.created_at)}
                for c in result.scalars().all()
            ]
            # Get dependencies: tasks that this task depends on
            dep_result = await session.execute(
                sa_select(TaskDependency.depends_on_id).where(TaskDependency.task_id == task_id)
            )
            depends_on = [row[0] for row in dep_result.all()]
            # Get dependents: tasks that depend on this task
            dep_result2 = await session.execute(
                sa_select(TaskDependency.task_id).where(TaskDependency.depends_on_id == task_id)
            )
            depended_by = [row[0] for row in dep_result2.all()]
            return {
                "id": task.id,
                "project_id": task.project_id,
                "title": task.title,
                "description": task.description,
                "column": task.board_column,
                "assignee_id": task.assignee_id,
                "complexity": task.complexity,
                "position": task.position,
                "is_bug": task.is_bug,
                "comments": comments,
                "depends_on": depends_on,
                "depended_by": depended_by,
            }

    @app.patch("/api/tasks/{task_id}")
    async def update_task(
        task_id: int,
        title: Optional[str] = Form(None),
        description: Optional[str] = Form(None),
        assignee_id: Optional[int] = Form(None),
        complexity: Optional[str] = Form(None),
    ):
        async with async_session() as session:
            task = await session.get(Task, task_id)
            if not task:
                raise HTTPException(404)
            if title:
                task.title = title
            if description is not None:
                task.description = description
            if assignee_id is not None:
                task.assignee_id = assignee_id if assignee_id > 0 else None
            if complexity is not None:
                task.complexity = complexity if complexity else None
            await session.commit()
            return {"ok": True}

    @app.delete("/api/projects/{project_id}")
    async def delete_project(project_id: int):
        """Delete a project and all its tasks, comments, and git states."""
        async with async_session() as session:
            project = await session.get(Project, project_id)
            if not project:
                raise HTTPException(404, "Project not found")
            
            # Get all tasks for this project
            result = await session.execute(
                sa_select(Task).where(Task.project_id == project_id)
            )
            tasks = result.scalars().all()
            task_ids = [t.id for t in tasks]
            
            if task_ids:
                # Delete task comments first
                await session.execute(
                    TaskComment.__table__.delete().where(TaskComment.task_id.in_(task_ids))
                )
                # Delete task git states
                await session.execute(
                    TaskGitState.__table__.delete().where(TaskGitState.task_id.in_(task_ids))
                )
                # Delete task dependencies (both directions)
                try:
                    await session.execute(
                        TaskDependency.__table__.delete().where(
                            TaskDependency.task_id.in_(task_ids) | 
                            TaskDependency.depends_on_id.in_(task_ids)
                        )
                    )
                except Exception:
                    pass
                # Delete the tasks themselves
                await session.execute(
                    Task.__table__.delete().where(Task.id.in_(task_ids))
                )
            
            # Now delete the project (no more FK violations)
            await session.execute(
                Project.__table__.delete().where(Project.id == project_id)
            )
            await session.commit()
            
            return {"ok": True, "message": f"Project '{project.name}' and all its tasks deleted."}

    @app.post("/api/tasks/{task_id}/dependencies")
    async def add_task_dependency(task_id: int, payload: dict):
        """Add a dependency: task_id depends on depends_on_id."""
        depends_on_id = payload.get("depends_on_id")
        if not depends_on_id:
            raise HTTPException(400, "depends_on_id is required")
        
        async with async_session() as session:
            # Verify both tasks exist
            task = await session.get(Task, task_id)
            if not task:
                raise HTTPException(404, "Task not found")
            dep_task = await session.get(Task, depends_on_id)
            if not dep_task:
                raise HTTPException(404, "Dependency task not found")
            
            # Check for circular dependency
            if task_id == depends_on_id:
                raise HTTPException(400, "Cannot depend on itself")
            
            # Check if dependency already exists
            existing = await session.execute(
                sa_select(TaskDependency).where(
                    TaskDependency.task_id == task_id,
                    TaskDependency.depends_on_id == depends_on_id
                )
            )
            if existing.scalar_one_or_none():
                raise HTTPException(400, "Dependency already exists")
            
            session.add(TaskDependency(task_id=task_id, depends_on_id=depends_on_id))
            await session.commit()
            return {"ok": True}

    @app.delete("/api/tasks/{task_id}/dependencies/{depends_on_id}")
    async def remove_task_dependency(task_id: int, depends_on_id: int):
        """Remove a dependency."""
        async with async_session() as session:
            await session.execute(
                TaskDependency.__table__.delete().where(
                    TaskDependency.task_id == task_id,
                    TaskDependency.depends_on_id == depends_on_id
                )
            )
            await session.commit()
            return {"ok": True}

    @app.delete("/api/tasks/{task_id}")
    async def delete_task(task_id: int):
        async with async_session() as session:
            task = await session.get(Task, task_id)
            if not task:
                raise HTTPException(404)
            await session.delete(task)
            await session.commit()
            return {"ok": True}

    # ── API: Move task ─────────────────────────────────────────────

    @app.post("/api/tasks/{task_id}/move")
    async def move_task(task_id: int, payload: TaskMovePayload):
        async with async_session() as session:
            task = await session.get(Task, task_id)
            if not task:
                raise HTTPException(404, "Task not found")

            old_column = task.board_column
            new_column = payload.column

            if new_column not in ("backlog", "running", "blocked", "review", "done"):
                raise HTTPException(400, f"Invalid column: {new_column}")

            # Check if moving to running — only one task allowed in progress at a time
            if new_column == "running":
                running_result = await session.execute(
                    sa_select(Task).where(
                        Task.project_id == task.project_id,
                        Task.board_column == "running",
                        Task.id != task_id
                    )
                )
                if running_result.scalar_one_or_none():
                    raise HTTPException(400,
                        "Another task is already in progress. This feature does not support parallel task execution yet."
                    )

            # Check dependencies: if moving to running/review/done, all dependencies must be done
            if new_column in ("running", "review", "done"):
                dep_result = await session.execute(
                    sa_select(TaskDependency.depends_on_id).where(TaskDependency.task_id == task_id)
                )
                dep_ids = [row[0] for row in dep_result.all()]
                if dep_ids:
                    # Check if all dependencies are done
                    dep_tasks_result = await session.execute(
                        sa_select(Task).where(Task.id.in_(dep_ids))
                    )
                    dep_tasks = dep_tasks_result.scalars().all()
                    not_done = [t for t in dep_tasks if t.board_column != "done"]
                    if not_done:
                        dep_titles = ", ".join(t.title for t in not_done)
                        raise HTTPException(400,
                            f"Cannot move to '{new_column}': waiting on tasks: {dep_titles}"
                        )

            task.board_column = new_column

            # If moving from blocked to running, kill old process before starting a new one
            if old_column == "blocked" and new_column == "running":
                if task.id in running_processes:
                    proc_info = running_processes[task.id]
                    if isinstance(proc_info, tuple) and len(proc_info) >= 1 and proc_info[0] is not None:
                        proc = proc_info[0]
                        if proc.returncode is None:
                            proc.kill()
                    del running_processes[task.id]

            # ── Save prompt comment FIRST (before AI run), so user always sees it ──
            if new_column == "running" and task.assignee_id:
                assignee_check = await session.get(User, task.assignee_id)
                if assignee_check and assignee_check.type == "ai":
                    # Build prompt preview (we'll save the full one inside _run_execute_command too)
                    from datetime import datetime as _dt
                    preview = (
                        f"📋 **Task moved to Running**\n\n"
                        f"**Title:** {task.title}\n"
                        f"**Description:** {task.description or '(no description)'}\n"
                        f"**Complexity:** {task.complexity or 'not specified'}\n"
                        f"**Assignee:** {assignee_check.name} (AI)\n"
                    )
                    if assignee_check.model:
                        preview += f"**Model:** `{assignee_check.model}`\n"
                    if assignee_check.provider:
                        preview += f"**Provider:** `{assignee_check.provider}`\n"
                    preview += f"\n⏱️ Started at {_dt.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
                    preview += "Full prompt and AI output will appear below once the process starts..."
                    session.add(TaskComment(
                        task_id=task.id,
                        author="Soda",
                        content=preview,
                    ))

            # ── If moving to Running and assignee is an AI user, execute command ──
            if new_column == "running":
                if task.assignee_id:
                    assignee = await session.get(User, task.assignee_id)
                    if assignee and assignee.type == "ai":
                        # Fetch comments BEFORE starting AI
                        comments_result = await session.execute(
                            sa_select(TaskComment).where(TaskComment.task_id == task.id).order_by(TaskComment.created_at)
                        )
                        comments = [
                            {"author": c.author, "content": c.content, "created_at": str(c.created_at)}
                            for c in comments_result.scalars().all()
                        ]
                        # Fetch dependencies for prompt context
                        deps_result = await session.execute(
                            sa_select(TaskDependency.depends_on_id).where(TaskDependency.task_id == task.id)
                        )
                        depends_on_ids = [row[0] for row in deps_result.all()]
                        # Commit the preview comment before launching AI
                        await session.commit()
                        try:
                            await _run_execute_command(
                                task, assignee, comments=comments, depends_on_ids=depends_on_ids
                            )
                        except Exception as e:
                            logger.error(f"Task {task.id}: _run_execute_command failed: {e}")
                            task.board_column = "blocked"
                            try:
                                async with async_session() as err_session:
                                    err_session.add(TaskComment(
                                        task_id=task.id,
                                        author="Soda",
                                        content=f"❌ **AI run failed to start:** {str(e)[:1000]}",
                                    ))
                                    await err_session.commit()
                            except Exception:
                                pass

            # If moving to backlog, kill process
            if new_column == "backlog" and task.id in running_processes:
                proc, stdout_fd, stderr_fd = running_processes[task.id]
                if proc.returncode is None:
                    proc.kill()
                del running_processes[task.id]

            project = await session.get(Project, task.project_id)
            if project and uses_auto_integrate(project):
                if new_column == "running":
                    project.pipeline_state = "running"
                    project.current_task_id = task.id
                elif new_column == "blocked":
                    project.pipeline_state = "waiting_user"
                    project.current_task_id = task.id

            await session.commit()
            return {"ok": True, "column": new_column}

    async def _pipeline_start_task(task_id: int) -> None:
        await move_task(task_id, TaskMovePayload(column="running"))

    @app.get("/api/projects/{project_id}/pipeline")
    async def get_pipeline(project_id: int):
        return await autopilot.get_pipeline_status(project_id)

    @app.post("/api/projects/{project_id}/pipeline/pause")
    async def pipeline_pause_api(project_id: int):
        await autopilot.pipeline_pause(project_id)
        return await autopilot.get_pipeline_status(project_id)

    @app.post("/api/projects/{project_id}/pipeline/resume")
    async def pipeline_resume_api(project_id: int):
        return await autopilot.pipeline_resume(project_id)

    @app.post("/api/projects/{project_id}/pipeline/next")
    async def pipeline_next_api(project_id: int):
        return await autopilot.pipeline_next(project_id)

    @app.patch("/api/projects/{project_id}/pipeline/mode")
    async def pipeline_mode_api(project_id: int, payload: dict):
        mode = payload.get("run_mode", "step")
        await autopilot.pipeline_set_mode(project_id, mode)
        return await autopilot.get_pipeline_status(project_id)

    @app.patch("/api/projects/{project_id}/advanced")
    async def project_advanced_mode(project_id: int, payload: dict):
        async with async_session() as session:
            project = await get_or_404(session, Project, project_id)
            project.advanced_mode = bool(payload.get("advanced_mode", False))
            advanced = project.advanced_mode
            await session.commit()
        return {"ok": True, "advanced_mode": advanced}

    @app.post("/api/projects/{project_id}/expand")
    async def expand_project(project_id: int, message: str = Form(...), kind: str = Form("idea")):
        """Free-text idea or bug → Architect creates relevant tasks."""
        if kind not in ("idea", "bug"):
            kind = "idea"
        message = (message or "").strip()
        if not message:
            raise HTTPException(400, "Message is required.")
        if project_id in expand_in_progress:
            raise HTTPException(409, "Already generating tasks for this project.")
        async with async_session() as session:
            await get_or_404(session, Project, project_id, "Project")
        asyncio.create_task(_expand_project_background(project_id, message, kind))
        return {"status": "generating", "kind": kind}

    @app.get("/api/projects/{project_id}/expand/status")
    async def expand_project_status(project_id: int):
        return {"generating": project_id in expand_in_progress}

    @app.post("/api/tasks/{task_id}/finish")
    async def finish_task_api(task_id: int):
        """Finish a task whose AI run completed but post-processing did not run."""
        async with async_session() as session:
            task = await session.get(Task, task_id)
            if not task:
                raise HTTPException(404, "Task not found")
            if task.board_column not in ("blocked", "running"):
                raise HTTPException(400, f"Task is in '{task.board_column}', cannot finish.")
            if task.board_column == "blocked":
                task.board_column = "running"
                await session.commit()
        await _post_process_task(task_id)
        return {"ok": True}

    @app.post("/api/tasks/{task_id}/integrate")
    async def integrate_task_api(task_id: int):
        """Run Architect review + auto-merge for autopilot tasks stuck in review."""
        async with async_session() as session:
            task = await session.get(Task, task_id)
            if not task:
                raise HTTPException(404, "Task not found")
            project = await session.get(Project, task.project_id)
            if not uses_auto_integrate(project):
                raise HTTPException(400, "Auto integration only applies to autopilot projects.")
            if task.board_column not in ("review", "running", "blocked"):
                raise HTTPException(400, f"Task is in '{task.board_column}', cannot integrate.")
            git_state = await session.get(TaskGitState, task_id)
            repo_name = project.repo_name or (git_state.repo if git_state else "")
            repo_url = project.repo_url or ""
            workdir = Path(git_state.workdir) if git_state and git_state.workdir else Path(f"/tmp/soda-task-workdirs/task-{task_id}")
            feature_branch = (git_state.branch if git_state and git_state.branch else f"task-{task_id}")
            git_username = await get_setting("git_username")
            git_token = await get_setting("git_token")
            default_branch = await get_setting("git_default_branch", "main")

        if not repo_name or not git_username or not git_token:
            raise HTTPException(400, "GitHub not configured or repo missing.")

        await _auto_integrate_task(
            task_id, workdir, repo_name, git_username, git_token,
            default_branch, feature_branch, repo_url,
        )
        return {"ok": True}

    # ── API: Comments ──────────────────────────────────────────────

    @app.get("/api/tasks/{task_id}/comments")
    async def list_comments(task_id: int):
        async with async_session() as session:
            result = await session.execute(
                sa_select(TaskComment).where(TaskComment.task_id == task_id).order_by(TaskComment.created_at)
            )
            return [
                {"id": c.id, "author": c.author, "content": c.content, "created_at": str(c.created_at)}
                for c in result.scalars().all()
            ]

    @app.post("/api/tasks/{task_id}/comments")
    async def add_comment(task_id: int, payload: CommentPayload):
        async with async_session() as session:
            task = await session.get(Task, task_id)
            if not task:
                raise HTTPException(404)
            comment = TaskComment(
                task_id=task_id, author=payload.author, content=payload.content
            )
            session.add(comment)
            await session.commit()
            await session.refresh(comment)
            return {"id": comment.id, "author": comment.author, "content": comment.content}

    # ── API: Ideas ─────────────────────────────────────────────────

    def _idea_to_dict(i: "Idea") -> dict:
        questions, generation_error = parse_idea_state(i)
        return {
            "id": i.id,
            "title": i.title,
            "description": i.description,
            "system_prompt": i.system_prompt,
            "architect_user_id": i.architect_user_id,
            "status": i.status,
            "questions": questions,
            "generation_error": generation_error,
            "created_at": str(i.created_at),
            "project_id": None,  # filled by caller via Project.source_idea_id
            "project_name": None,
        }

    @app.get("/api/ideas")
    async def list_ideas():
        async with async_session() as session:
            result = await session.execute(
                sa_select(Idea).order_by(Idea.created_at.desc())
            )
            ideas = result.scalars().all()
            # Build a map: idea_id -> Project (one project per idea via source_idea_id)
            idea_ids = [i.id for i in ideas]
            proj_map: dict[int, "Project"] = {}
            if idea_ids:
                proj_result = await session.execute(
                    sa_select(Project).where(Project.source_idea_id.in_(idea_ids))
                )
                for p in proj_result.scalars().all():
                    if p.source_idea_id is not None:
                        proj_map[p.source_idea_id] = p

            out = []
            for i in ideas:
                d = _idea_to_dict(i)
                proj = proj_map.get(i.id)
                if proj:
                    d["project_id"] = proj.id
                    d["project_name"] = proj.name
                out.append(d)
            return out

    @app.post("/api/ideas")
    async def create_idea(
        title: str = Form(...),
        description: str = Form(""),
        system_prompt: str = Form(""),
        architect_user_id: Optional[int] = Form(None),
    ):
        async with async_session() as session:
            idea = Idea(
                title=title,
                description=description,
                system_prompt=system_prompt,
                architect_user_id=architect_user_id,
                status="active",
            )
            session.add(idea)
            await session.commit()
            await session.refresh(idea)
            return {"id": idea.id, "title": idea.title, "status": idea.status}

    @app.patch("/api/ideas/{idea_id}")
    async def update_idea(
        idea_id: int,
        title: Optional[str] = Form(None),
        description: Optional[str] = Form(None),
        system_prompt: Optional[str] = Form(None),
        architect_user_id: Optional[int] = Form(None),
    ):
        async with async_session() as session:
            idea = await session.get(Idea, idea_id)
            if not idea:
                raise HTTPException(404, "Idea not found")
            if title:
                idea.title = title
            if description is not None:
                idea.description = description
            if system_prompt is not None:
                idea.system_prompt = system_prompt
            if architect_user_id is not None:
                idea.architect_user_id = architect_user_id if architect_user_id > 0 else None
            await session.commit()
            return {"ok": True}

    @app.delete("/api/ideas/{idea_id}")
    async def delete_idea(idea_id: int):
        async with async_session() as session:
            idea = await session.get(Idea, idea_id)
            if not idea:
                raise HTTPException(404, "Idea not found")
            await session.delete(idea)
            await session.commit()
            return {"ok": True}

    async def _call_architect(architect: "User", prompt: str) -> dict:
        """Call the architect AI and return parsed JSON result.
        Uses OpenRouter API directly when OpenRouter provider is configured,
        otherwise uses OpenCode CLI."""
        import logging
        import httpx as _httpx
        logger = logging.getLogger(__name__)

        logger.info(f"Calling architect {architect.name} with prompt length: {len(prompt)}")

        p = await resolve_user_provider(architect)
        if p == "openrouter":
            return await _call_openrouter_architect(architect, prompt)
        if p == "minimax":
            return await _call_minimax_architect(architect, prompt)
        return await _call_opencode_architect(architect, prompt)

    async def _call_openrouter_architect(architect: "User", prompt: str) -> dict:
        """Call architect via direct OpenRouter API call (no OpenCode agent/tools)."""
        api_key = await _get_openrouter_api_key()
        if not api_key:
            raise HTTPException(400, "OpenRouter API key not configured in Settings")

        model = architect.model or "anthropic/claude-sonnet-4"
        model_str = model if "/" in model else f"openrouter/{model}"

        last_error = None
        # Try up to 3 times — OpenRouter often returns transient empty responses
        for attempt in range(1, 4):
            async with httpx.AsyncClient(timeout=ARCHITECT_HTTP_TIMEOUT_SEC) as client:
                try:
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
                                {"role": "system", "content": "You are an expert software architect. You output ONLY valid JSON, no other text."},
                                {"role": "user", "content": prompt},
                            ],
                            "temperature": 0.2,
                        },
                    )
                except Exception as e:
                    last_error = f"HTTP error: {e}"
                    logger.warning(f"OpenRouter attempt {attempt}/3 failed: {e}")
                    await asyncio.sleep(2 * attempt)
                    continue

                if resp.status_code != 200:
                    err_detail = resp.text[:500]
                    last_error = f"HTTP {resp.status_code}: {err_detail}"
                    logger.warning(f"OpenRouter attempt {attempt}/3 failed: {last_error}")
                    # 4xx is permanent (bad model, invalid key) — don't retry
                    if 400 <= resp.status_code < 500:
                        raise HTTPException(500, f"Architect API error: {last_error}")
                    await asyncio.sleep(2 * attempt)
                    continue

                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                if not content:
                    last_error = "Empty content in response"
                    logger.warning(f"OpenRouter attempt {attempt}/3 returned empty content")
                    await asyncio.sleep(2 * attempt)
                    continue

                # Got content — break out of retry loop
                break
        else:
            # All retries exhausted
            raise HTTPException(500, f"Architect returned empty response from OpenRouter after 3 attempts: {last_error}")

        output = content.strip()
        try:
            result = parse_json_from_llm_output(output)
            logger.info(f"Successfully parsed architect response: {result.get('type')}")
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse architect JSON: {str(e)}, Output: {output[:500]}")
            raise HTTPException(500, f"Failed to parse architect JSON response: {str(e)}")

    async def _call_minimax_architect(architect: "User", prompt: str) -> dict:
        """Call architect via Minimax API."""
        api_key = await _get_minimax_api_key()
        if not api_key:
            raise HTTPException(400, "Minimax API key not configured in Settings")

        model = architect.model or "MiniMax-Text-01"
        last_error = None
        for attempt in range(1, 4):
            async with httpx.AsyncClient(timeout=ARCHITECT_HTTP_TIMEOUT_SEC) as client:
                try:
                    resp = await client.post(
                        "https://api.minimax.chat/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": "You are an expert software architect. You output ONLY valid JSON, no other text."},
                                {"role": "user", "content": prompt},
                            ],
                            "temperature": 0.2,
                        },
                    )
                except Exception as e:
                    last_error = f"HTTP error: {e}"
                    logger.warning(f"Minimax attempt {attempt}/3 failed: {e}")
                    await asyncio.sleep(2 * attempt)
                    continue

                if resp.status_code != 200:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
                    logger.warning(f"Minimax attempt {attempt}/3 failed: {last_error}")
                    if 400 <= resp.status_code < 500:
                        raise HTTPException(500, f"Architect API error: {last_error}")
                    await asyncio.sleep(2 * attempt)
                    continue

                content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                if not content:
                    last_error = "Empty content in response"
                    await asyncio.sleep(2 * attempt)
                    continue
                break
        else:
            raise HTTPException(500, f"Architect returned empty response from Minimax after 3 attempts: {last_error}")

        try:
            return parse_json_from_llm_output(content.strip())
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(500, f"Failed to parse architect JSON response: {str(e)}")

    async def _call_opencode_architect(architect: "User", prompt: str) -> dict:
        """Call architect via OpenCode CLI (for OpenCode provider)."""
        import logging
        logger = logging.getLogger(__name__)

        _write_opencode_auth(architect)
        api_key = await _get_opencode_api_key()
        env = os.environ.copy()
        if api_key:
            env["OPENCODE_API_KEY"] = api_key

        logger.info(f"Calling architect {architect.name} via OpenCode with prompt length: {len(prompt)}")

        proc = await asyncio.create_subprocess_shell(
            f'opencode run --pure {json.dumps(prompt)}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=ARCHITECT_OPENCODE_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.error(f"Architect timed out after {ARCHITECT_OPENCODE_TIMEOUT_SEC}s")
            raise HTTPException(
                504,
                f"Architect AI timed out after {ARCHITECT_OPENCODE_TIMEOUT_SEC // 60} minutes. "
                "Try again — large projects can take a while.",
            )
        output = stdout.decode().strip()
        error_output = stderr.decode().strip()

        logger.info(f"Architect stdout length: {len(output)}")
        logger.info(f"Architect stderr length: {len(error_output)}")
        if error_output:
            logger.error(f"Architect stderr: {error_output}")

        if proc.returncode != 0:
            logger.error(f"Architect process failed with code {proc.returncode}")
            raise HTTPException(500, f"Architect process failed (code {proc.returncode}): {error_output[:500]}")

        try:
            result = parse_json_from_llm_output(output)
            logger.info(f"Successfully parsed architect response: {result.get('type')}")
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse architect JSON: {str(e)}, Output: {output[:500]}")
            raise HTTPException(500, f"Failed to parse architect JSON response: {str(e)}")

    async def _append_tasks_to_project(
        project_id: int,
        task_specs: list[dict],
        is_bug: bool = False,
    ) -> list[int]:
        """Add new backlog tasks to an existing project."""
        if not task_specs:
            return []
        async with async_session() as session:
            result = await session.execute(
                sa_select(Task)
                .where(Task.project_id == project_id)
                .order_by(Task.position.desc())
                .limit(1)
            )
            last = result.scalar_one_or_none()
            pos = (last.position + 1) if last else 0
            chain_dep_id = last.id if last else None

            coder_res = await session.execute(sa_select(User).where(User.name == "Coder"))
            coder = coder_res.scalar_one_or_none()
            default_coder_id = coder.id if coder else None

            created_ids: list[int] = []
            prev_new_id = chain_dep_id
            for i, t in enumerate(task_specs):
                task = Task(
                    project_id=project_id,
                    title=t.get("title", "Untitled"),
                    description=t.get("description", ""),
                    complexity=_normalize_complexity(t.get("complexity")),
                    board_column="backlog",
                    position=pos + i,
                    assignee_id=default_coder_id,
                    is_bug=is_bug,
                )
                session.add(task)
                await session.flush()
                created_ids.append(task.id)

                dep_indices = list(t.get("depends_on") or [])
                if dep_indices:
                    for dep_idx in dep_indices:
                        if isinstance(dep_idx, int) and 0 <= dep_idx < len(created_ids) - 1:
                            session.add(TaskDependency(
                                task_id=task.id,
                                depends_on_id=created_ids[dep_idx],
                            ))
                elif prev_new_id:
                    session.add(TaskDependency(task_id=task.id, depends_on_id=prev_new_id))
                prev_new_id = task.id

            project = await session.get(Project, project_id)
            if project and project.pipeline_state == "complete":
                project.pipeline_state = "paused"
            await session.commit()
            return created_ids

    async def _verify_project_completion(project_id: int) -> bool:
        """Architect final check — add missing tasks if the plan is incomplete."""
        async with async_session() as session:
            project = await session.get(Project, project_id)
            if not project:
                return False
            tasks_res = await session.execute(
                sa_select(Task).where(Task.project_id == project_id).order_by(Task.position, Task.created_at)
            )
            tasks = tasks_res.scalars().all()
            if not tasks or any(t.board_column != "done" for t in tasks):
                return False
            arch_res = await session.execute(sa_select(User).where(User.name == "Architect"))
            architect = arch_res.scalar_one_or_none()
            if not architect:
                return False

        task_lines = "\n".join(
            f"- ✅ {t.title}: {(t.description or '')[:120]}" for t in tasks
        )
        prompt = f"""Final review before marking this project complete.

Project: {project.name}
Goal: {project.description or ''}

Completed tasks:
{task_lines}

Compare the original goal with what was built. If anything important is still missing, return new tasks to add.
Do NOT repeat work already done. Only add genuinely missing pieces.

Return ONLY valid JSON:
{{"complete": true}}
or
{{"complete": false, "tasks": [{{"title": "...", "description": "...", "complexity": "XS|S|M|L|XL"}}]}}"""

        try:
            result = await _call_architect(architect, prompt)
        except Exception as e:
            logger.warning(f"Project {project_id} completion review failed: {e}")
            return False

        if result.get("complete", True):
            return False
        new_specs = result.get("tasks") or []
        if not isinstance(new_specs, list) or not new_specs:
            return False
        created = await _append_tasks_to_project(project_id, new_specs)
        if created:
            async with async_session() as session:
                p = await session.get(Project, project_id)
                if p:
                    p.pipeline_state = "idle"
                    await session.commit()
            logger.info(f"Project {project_id}: completion review added {len(created)} task(s)")
            return True
        return False

    async def _expand_project_background(project_id: int, message: str, kind: str = "idea") -> None:
        """Architect interprets free-text idea/bug and creates relevant tasks."""
        expand_in_progress.add(project_id)
        try:
            async with async_session() as session:
                project = await session.get(Project, project_id)
                if not project:
                    return
                tasks_res = await session.execute(
                    sa_select(Task)
                    .where(Task.project_id == project_id)
                    .order_by(Task.position, Task.created_at)
                )
                tasks = tasks_res.scalars().all()
                arch_res = await session.execute(sa_select(User).where(User.name == "Architect"))
                architect = arch_res.scalar_one_or_none()
                if not architect:
                    return

            status_icon = {"done": "✅", "running": "▶", "blocked": "❓", "review": "👁", "backlog": "📋"}
            task_lines = "\n".join(
                f"- {status_icon.get(t.board_column, '•')} [{t.board_column}] {t.title}: {(t.description or '')[:100]}"
                for t in tasks
            )
            kind_label = "bug fix" if kind == "bug" else "feature/enhancement"
            prompt = f"""You are an Architect helping evolve an existing software project.

Project: {project.name}
Description: {project.description or ''}

Current tasks:
{task_lines or '(none)'}

The user wants to add a {kind_label}:
\"\"\"{message}\"\"\"

Review the codebase context implied by completed tasks and the user request.
Create only the tasks needed for this request — focused, relevant, not a full replan.
New tasks will be appended to the backlog and run after existing work.

Return ONLY valid JSON:
{{
  "type": "tasks",
  "tasks": [
    {{"title": "...", "description": "...", "complexity": "XS|S|M|L|XL", "depends_on": []}}
  ]
}}

Use "depends_on" with 0-based indices within the new tasks array only.
Keep tasks small and actionable. Return ONLY JSON."""

            result = await _call_architect(architect, prompt)
            if result.get("type") == "tasks" or result.get("tasks"):
                specs = result.get("tasks") or []
                if isinstance(specs, list) and specs:
                    created = await _append_tasks_to_project(project_id, specs, is_bug=(kind == "bug"))
                    run_mode = "step"
                    async with async_session() as session:
                        project = await session.get(Project, project_id)
                        if project:
                            run_mode = project.run_mode or "step"
                        if created:
                            session.add(TaskComment(
                                task_id=created[0],
                                author="Soda",
                                content=f"💡 **From your {'bug report' if kind == 'bug' else 'idea'}:**\n\n{message}",
                            ))
                            await session.commit()
                    if run_mode == "auto":
                        await autopilot.pipeline_resume(project_id)
        except Exception as e:
            logger.error(f"Expand project {project_id} failed: {e}")
        finally:
            expand_in_progress.discard(project_id)

    async def _create_project_from_result(
        idea: "Idea",
        result: dict,
        repo_name: str = None,
        repo_private: bool = True,
        run_mode: str = "step",
    ) -> dict:
        """Create project and tasks from architect generate response.
        Also creates a GitHub repo for the project.
        If repo creation fails, the entire project generation fails (no tasks created)."""
        async with async_session() as session:
            existing = await session.execute(
                sa_select(Project).where(Project.source_idea_id == idea.id)
            )
            if existing.scalar_one_or_none():
                raise HTTPException(400, "A project already exists for this idea")

        username = await get_setting("git_username")
        token = await get_setting("git_token")

        if not username or not token:
            raise HTTPException(400,
                "⚠️ GitHub auth is required to generate projects. "
                "Please configure git_username and git_token in Settings. "
                "The token needs 'repo' scope to create repositories."
            )

        if repo_name:
            repo_name = re.sub(r'[^a-z0-9-]', '', repo_name.lower().replace(' ', '-'))[:100]
        else:
            repo_name = result.get("project_name", idea.title).lower().replace(" ", "-")
            repo_name = re.sub(r'[^a-z0-9-]', '', repo_name)[:100]

        if not repo_name:
            repo_name = f"soda-{idea.id}"

        project_name = result.get("project_name", idea.title)
        project_description = result.get("project_description", idea.description)
        tech_stack = result.get("tech_stack", "generic")
        default_branch = await get_setting("git_default_branch", "main")

        ensure_result = await _ensure_github_repo(username, repo_name, token, private=repo_private)
        if ensure_result["status"] == "error":
            raise HTTPException(400,
                f"⚠️ Failed to create GitHub repo '{repo_name}': {ensure_result['message']}. "
                "Please check your git_token has 'repo' scope and the repo name is valid. "
                "Project generation was cancelled."
            )

        if ensure_result["status"] == "created":
            scaffold_files = get_scaffold_files(tech_stack, project_name, project_description or "")
            gh_service = GitHubService(username, token)
            ok = await gh_service.commit_files(
                repo_name, scaffold_files, "chore: initial project scaffold", branch=default_branch
            )
            if not ok:
                logger.warning("Scaffold commit failed for %s — repo has default files only", repo_name)

        repo_url = ensure_result["data"].get("html_url", f"https://github.com/{username}/{repo_name}")

        # Create project and tasks in DB
        async with async_session() as session:
            project = Project(
                name=result.get("project_name", idea.title),
                description=result.get("project_description", idea.description),
                repo_name=repo_name,
                repo_url=repo_url,
                source_idea_id=idea.id,
            )
            session.add(project)
            await session.commit()
            await session.refresh(project)

            # Resolve assignee role to user ID
            def _resolve_assignee_id(assignee_role: str) -> Optional[int]:
                """Map assignee_role to an existing AI user ID."""
                role_map = {
                    "coder": "Coder",
                    "architect": "Architect",
                    "junior": "Coder",
                    "medior": "Coder",
                    "senior": "Coder",
                    "task_manager": "Architect",
                }
                target_name = role_map.get((assignee_role or "coder").lower(), "Coder")
                return target_name

            coder_res = await session.execute(sa_select(User).where(User.name == "Coder"))
            coder_user = coder_res.scalar_one_or_none()
            default_coder_id = coder_user.id if coder_user else None

            assignee_name_to_id: dict[str, int] = {}
            for t in result.get("tasks", []):
                role = t.get("assignee_role", "coder")
                name = _resolve_assignee_id(role)
                if name and name not in assignee_name_to_id:
                    assignee_user = await session.execute(
                        sa_select(User).where(User.name == name)
                    )
                    user_obj = assignee_user.scalar_one_or_none()
                    if user_obj:
                        assignee_name_to_id[name] = user_obj.id

            # Create tasks and track their DB IDs in order
            task_db_ids = []
            for i, t in enumerate(result.get("tasks", [])):
                assignee_id = default_coder_id
                role = t.get("assignee_role", "coder")
                resolved_name = _resolve_assignee_id(role)
                if resolved_name:
                    assignee_id = assignee_name_to_id.get(resolved_name, default_coder_id)

                task = Task(
                    project_id=project.id,
                    title=t.get("title", "Untitled"),
                    description=t.get("description", ""),
                    complexity=_normalize_complexity(t.get("complexity")),
                    board_column="backlog",
                    position=i,
                    assignee_id=assignee_id,
                )
                session.add(task)
                await session.flush()  # Get the task ID
                task_db_ids.append(task.id)

            # Create dependencies: architect indices + ensure sequential order
            for i, t in enumerate(result.get("tasks", [])):
                depends_on_indices = list(t.get("depends_on") or [])
                if i > 0 and (i - 1) not in depends_on_indices:
                    depends_on_indices.append(i - 1)
                if depends_on_indices:
                    task_id = task_db_ids[i]
                    for dep_idx in depends_on_indices:
                        if isinstance(dep_idx, int) and 0 <= dep_idx < len(task_db_ids) and dep_idx != i:
                            dep_task_id = task_db_ids[dep_idx]
                            # Check if dependency already exists (avoid duplicates)
                            existing = await session.execute(
                                sa_select(TaskDependency).where(
                                    TaskDependency.task_id == task_id,
                                    TaskDependency.depends_on_id == dep_task_id
                                )
                            )
                            if not existing.scalar_one_or_none():
                                session.add(TaskDependency(task_id=task_id, depends_on_id=dep_task_id))

            idea_obj = await session.get(Idea, idea.id)
            idea_obj.status = "generated"
            idea_obj.pending_questions = None
            idea_obj.generation_error = None
            project.run_mode = run_mode if run_mode in ("auto", "step") else "step"
            project.pipeline_state = "paused"
            await session.commit()

        await autopilot.init_pipeline(project.id, run_mode=run_mode)
        asyncio.create_task(autopilot.maybe_auto_start_after_generation(project.id))

        return {
            "status": "generated",
            "project_id": project.id,
            "project_name": project.name,
            "tasks_count": len(result.get("tasks", [])),
            "repo_url": repo_url,
            "repo_name": repo_name,
        }

    @app.post("/api/ideas/{idea_id}/generate")
    async def generate_from_idea(
        idea_id: int,
        architect_user_id: int = Form(None),
        repo_name: str = Form(None),
        repo_private: str = Form("true"),
        run_mode: str = Form("step"),
    ):
        """Start generating a project from an idea using the Architect AI."""
        if run_mode not in ("auto", "step"):
            run_mode = "step"
        async with async_session() as session:
            idea = await session.get(Idea, idea_id)
            if not idea:
                raise HTTPException(404, "Idea not found")

            arch_id = architect_user_id or idea.architect_user_id
            if not arch_id:
                architect_row = await session.execute(
                    sa_select(User).where(User.name == "Architect")
                )
                arch = architect_row.scalar_one_or_none()
                if arch:
                    arch_id = arch.id
                    idea.architect_user_id = arch.id
            if not arch_id:
                raise HTTPException(400, "No Architect AI user found. Complete setup first.")

            architect = await session.get(User, arch_id)
            if not architect or architect.type != "ai":
                raise HTTPException(400, "Architect must be an AI user")

            idea.architect_user_id = arch_id
            idea.status = "generating"
            idea.generation_error = None
            idea.generation_run_mode = run_mode
            await session.commit()

            sys_prompt = architect.system_prompt or ""
            if idea.system_prompt:
                sys_prompt += "\n\n" + idea.system_prompt

            prompt = f"""You are an Architect AI. Generate a project plan from this idea.

Title: {idea.title}
Description: {idea.description}

{sys_prompt}

If you need clarification before generating, return ONLY this JSON:
{{
  "type": "questions",
  "questions": ["Question 1?", "Question 2?"]
}}

If you are ready to generate, return ONLY this JSON:
{{
  "type": "generate",
  "project_name": "...",
  "project_description": "...",
  "tech_stack": "python-fastapi|node-express|static-html|generic",
  "tasks": [
    {{"title": "...", "description": "...", "complexity": "XS|S|M|L|XL", "assignee_role": "coder", "depends_on": []}}
  ]
}}

IMPORTANT: Each task can have a "depends_on" field with indices of previous tasks it depends on.
- tasks[0] should always have "depends_on": [] (no dependencies, can start immediately)
- Subsequent tasks should depend on earlier tasks that must be completed first
- Use task indices (0-based) for dependencies, e.g., "depends_on": [0, 1]
- Create a logical dependency chain: setup → core → features → tests → deploy
- A task can depend on multiple previous tasks if needed
- Choose tech_stack based on the idea (python-fastapi for Python APIs, node-express for Node, static-html for simple sites)

Return ONLY valid JSON, no other text."""

            proj_result = await session.execute(
                sa_select(Project).where(Project.source_idea_id == idea_id)
            )
            project_for_idea = proj_result.scalar_one_or_none()
            architect_snapshot = _architect_snapshot(architect)

        asyncio.create_task(_generate_project_background(
            idea_id=idea_id,
            architect_snapshot=architect_snapshot,
            prompt=prompt,
            repo_name=repo_name,
            repo_private=repo_private == "true",
            run_mode=run_mode,
        ))

        return {
            "status": "generating",
            "idea_id": idea_id,
            "project_id": project_for_idea.id if project_for_idea else None,
        }

    async def _fail_idea_generation(idea_id: int, exc: Exception) -> None:
        """Record a generation failure without blocking retry."""
        error_msg = format_generation_error(exc)
        logger.error(f"Generation failed for idea {idea_id}: {error_msg}")
        async with async_session() as session:
            idea_obj = await session.get(Idea, idea_id)
            if idea_obj:
                idea_obj.status = "active"
                idea_obj.generation_error = error_msg
                idea_obj.pending_questions = None
                await session.commit()

    async def _generate_project_background(
        idea_id: int,
        architect_snapshot: dict,
        prompt: str,
        repo_name: Optional[str] = None,
        repo_private: bool = True,
        run_mode: str = "step",
    ):
        """Background task: call architect, create project, handle errors."""
        generation_tasks.add(idea_id)
        architect = architect_from_snapshot(architect_snapshot)
        try:
            try:
                result = await _call_architect(architect, prompt)
            except Exception as e:
                await _fail_idea_generation(idea_id, e)
                return

            if result.get("type") == "questions":
                questions = result.get("questions", [])
                if not isinstance(questions, list):
                    await _fail_idea_generation(
                        idea_id,
                        HTTPException(500, "Architect returned invalid questions format"),
                    )
                    return
                async with async_session() as session:
                    idea_obj = await session.get(Idea, idea_id)
                    if idea_obj:
                        idea_obj.status = "active"
                        idea_obj.generation_error = None
                        idea_obj.pending_questions = json.dumps(questions)
                        await session.commit()
                return

            if result.get("type") == "generate":
                try:
                    async with async_session() as session:
                        idea = await session.get(Idea, idea_id)
                    await _create_project_from_result(
                        idea, result,
                        repo_name=repo_name,
                        repo_private=repo_private,
                        run_mode=run_mode or getattr(idea, "generation_run_mode", None) or "step",
                    )
                except Exception as e:
                    await _fail_idea_generation(idea_id, e)
                return

            await _fail_idea_generation(
                idea_id,
                HTTPException(500, "Unexpected architect response. Try again or pick another model."),
            )
        finally:
            generation_tasks.discard(idea_id)

    @app.post("/api/ideas/{idea_id}/answer")
    async def answer_idea_questions(idea_id: int, answers: str = Form(...)):
        """Submit answers to architect questions and continue generation (async)."""
        async with async_session() as session:
            idea = await session.get(Idea, idea_id)
            if not idea:
                raise HTTPException(404, "Idea not found")
            if not idea.architect_user_id:
                raise HTTPException(400, "No architect user selected")
            architect = await session.get(User, idea.architect_user_id)
            if not architect or architect.type != "ai":
                raise HTTPException(400, "Architect must be an AI user")

            pending_questions = []
            if idea.pending_questions:
                try:
                    pending_questions = json.loads(idea.pending_questions)
                except Exception:
                    pending_questions = []

            idea.status = "generating"
            idea.generation_error = None
            idea_title = idea.title
            idea_description = idea.description
            idea_system_prompt = idea.system_prompt
            idea_run_mode = idea.generation_run_mode or "step"
            architect_snapshot = _architect_snapshot(architect)

            proj_result2 = await session.execute(
                sa_select(Project).where(Project.source_idea_id == idea_id)
            )
            project_for_idea2 = proj_result2.scalar_one_or_none()
            await session.commit()

        try:
            answers_list = json.loads(answers)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid answers format")

        sys_prompt = architect_snapshot.get("system_prompt") or ""
        if idea_system_prompt:
            sys_prompt += "\n\n" + idea_system_prompt

        qa_pairs = "\n".join([
            f"Q: {q}\nA: {a}"
            for q, a in zip(pending_questions, answers_list)
        ])

        prompt = f"""You are an Architect AI. Generate a project plan from this idea.

Title: {idea_title}
Description: {idea_description}

{sys_prompt}

You previously asked questions and received these answers:
{qa_pairs}

Now generate the project. Return ONLY this JSON:
{{
  "type": "generate",
  "project_name": "...",
  "project_description": "...",
  "tech_stack": "python-fastapi|node-express|static-html|generic",
  "tasks": [
    {{"title": "...", "description": "...", "complexity": "XS|S|M|L|XL", "assignee_role": "coder", "depends_on": []}}
  ]
}}

If you still have questions, return:
{{
  "type": "questions",
  "questions": ["Question?"]
}}

IMPORTANT: Each task can have a "depends_on" field with indices of previous tasks it depends on.
- tasks[0] should always have "depends_on": [] (no dependencies)
- Use task indices (0-based) for dependencies
- Create a logical dependency chain
- Choose tech_stack based on the idea

Return ONLY valid JSON, no other text."""

        asyncio.create_task(_generate_project_background(
            idea_id=idea_id,
            architect_snapshot=architect_snapshot,
            prompt=prompt,
            repo_name=None,
            repo_private=True,
            run_mode=idea_run_mode,
        ))

        return {
            "status": "generating",
            "idea_id": idea_id,
            "project_id": project_for_idea2.id if project_for_idea2 else None,
        }

    # ── API: Users ─────────────────────────────────────────────────

    @app.get("/api/users")
    async def list_users():
        async with async_session() as session:
            result = await session.execute(sa_select(User).order_by(User.name))
            users = result.scalars().all()
            return [
                {
                    "id": u.id,
                    "name": u.name,
                    "type": u.type,
                    "provider": u.provider,
                    "model": u.model,
                    "system_prompt": u.system_prompt,
                    "task_types": u.task_types or [],
                }
                for u in users
            ]

    @app.get("/api/users/{user_id}/default-sizes")
    async def get_user_sizes(user_id: int):
        """Get default polo sizes for a user."""
        async with async_session() as session:
            user = await get_or_404(session, User, user_id)
            sizes = await get_user_default_sizes(session, user_id)
            return {"user_id": user_id, "default_sizes": sizes}

    @app.put("/api/users/{user_id}/task-types")
    async def update_user_task_types(user_id: int, payload: dict):
        """
        Set task types for a user.
        Body: {"task_types": ["xs", "s", "task_manager"]}
        Sizes (xs/s/m/l/xl) can only be assigned to one user each.
        Special roles (task_manager, merger) can have multiple users.
        """
        task_types = payload.get("task_types", [])
        if not isinstance(task_types, list):
            raise HTTPException(400, "task_types must be a list")

        valid_sizes = ["xs", "s", "m", "l", "xl"]
        valid_roles = ["task_manager", "merger"]
        for t in task_types:
            if t not in valid_sizes and t not in valid_roles:
                raise HTTPException(400, f"Invalid task_type: {t}")

        async with async_session() as session:
            user = await get_or_404(session, User, user_id)
            # Check uniqueness for sizes
            for t in task_types:
                if t in valid_sizes:
                    conflict = await session.execute(
                        sa_select(User).where(
                            User.task_types.contains([t]),
                            User.id != user_id,
                        )
                    )
                    other = conflict.scalars().first()
                    if other:
                        raise HTTPException(400, f"Size '{t}' is already assigned to user '{other.name}'")
            user.task_types = task_types
            await session.commit()
            return {"user_id": user_id, "task_types": task_types}

    @app.get("/api/sizes")
    async def list_sizes():
        """List all task types and their assigned users."""
        async with async_session() as session:
            result = await session.execute(sa_select(User))
            users = result.scalars().all()
            assignments = {}
            for u in users:
                for t in (u.task_types or []):
                    assignments[t] = {"user_id": u.id, "user_name": u.name}
            return {
                "valid_sizes": ["xs", "s", "m", "l", "xl"],
                "valid_roles": ["task_manager", "merger"],
                "sizes": assignments,
            }

    @app.post("/api/users")
    async def create_user(
        name: str = Form(...),
        type: str = Form(...),
        provider: str = Form(""),
        api_key: str = Form(""),
        model: str = Form(""),
        system_prompt: str = Form(""),
    ):
        async with async_session() as session:
            user = User(
                name=name,
                type=type,
                provider=provider or None,
                api_key=api_key or None,
                model=model or None,
                system_prompt=system_prompt or None,
                task_types=[],
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return {"id": user.id, "name": user.name, "type": user.type}

    @app.patch("/api/users/{user_id}")
    async def update_user(
        user_id: int,
        name: Optional[str] = Form(None),
        type: Optional[str] = Form(None),
        provider: Optional[str] = Form(None),
        api_key: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
        system_prompt: Optional[str] = Form(None),
    ):
        async with async_session() as session:
            user = await session.get(User, user_id)
            if not user:
                raise HTTPException(404)
            if name:
                user.name = name
            if type is not None:
                user.type = type
            if api_key is not None:
                user.api_key = api_key or None
            if provider is not None:
                user.provider = provider or None
            if model is not None:
                user.model = model or None
            if system_prompt is not None:
                user.system_prompt = system_prompt or None
            await session.commit()
            return {"ok": True}

    @app.delete("/api/users/{user_id}")
    async def delete_user(user_id: int):
        async with async_session() as session:
            user = await session.get(User, user_id)
            if not user:
                raise HTTPException(404)
            await session.delete(user)
            await session.commit()
            return {"ok": True}

    # ── API: Settings ──────────────────────────────────────────────

    @app.get("/api/settings")
    async def get_settings():
        async with async_session() as session:
            result = await session.execute(sa_select(GlobalSetting))
            return {row.key: row.value for row in result.scalars().all()}

    @app.patch("/api/settings")
    async def update_settings(request: Request):
        """Update global settings. Accepts all settings as form data."""
        data = await request.form()
        form = {k: str(v) for k, v in data.items()}
        for cb in ("provider_opencode_enabled", "provider_openrouter_enabled", "provider_minimax_enabled"):
            if cb not in form:
                form[cb] = "false"

        git_token = form.get("git_token", "").strip()
        if git_token:
            gh_user = await resolve_github_username(git_token)
            if gh_user:
                form["git_username"] = gh_user

        async with async_session() as session:
            await _apply_settings_form(session, form, preserve_empty_keys=True)

        return {"ok": True, "updated": list(form.keys())}

    # ── API: Operation Commands ───────────────────────────────────────

    @app.get("/api/operations")
    async def list_operations():
        """List all operation commands (task_run, merge, etc.)."""
        try:
            cmds = await get_all_operation_commands()
            return cmds
        except Exception as e:
            logger.error(f"Error listing operations: {e}")
            return {}

    @app.patch("/api/operations/{op}")
    async def update_operation(op: str, request: Request):
        """Update an operation command template."""
        try:
            body = await request.json()
            value = body.get("value", "")
        except Exception:
            data = await request.form()
            value = str(data.get("value", ""))
        try:
            await set_operation_command(op, value)
            return {"ok": True}
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            logger.error(f"Error updating operation {op}: {e}")
            raise HTTPException(500, str(e))

    # ── API: Models ────────────────────────────────────────────────

    @app.get("/api/providers")
    async def list_providers():
        """List all enabled providers with their config."""
        try:
            providers = await get_enabled_providers()
            return {"providers": providers}
        except Exception as e:
            logger.error(f"Error listing providers: {e}")
            return {"providers": []}

    @app.get("/api/models")
    async def list_models(provider: str = Query(None)):
        """List available AI models for a specific provider.
        If no provider is given, uses the first enabled one."""
        try:
            if not provider:
                providers = await get_enabled_providers()
                if not providers:
                    return []
                provider = providers[0]["id"]

            if provider == "openrouter":
                return await _fetch_openrouter_models(await _get_openrouter_api_key())
            elif provider == "minimax":
                return await _fetch_openrouter_models(await _get_minimax_api_key(), base_url="https://api.minimax.chat/v1")
            else:
                return await _fetch_opencode_models(await _get_opencode_api_key())
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return []

    @app.post("/api/models/preview")
    async def preview_models(provider: str = Form(...), api_key: str = Form(...)):
        """Fetch models using a key from the setup form (before settings are saved)."""
        key = api_key.strip()
        if not key:
            return []
        if provider == "openrouter":
            return await _fetch_openrouter_models(key)
        if provider == "minimax":
            return await _fetch_openrouter_models(key, base_url="https://api.minimax.chat/v1")
        return await _fetch_opencode_models(key)

    async def _fetch_openrouter_models(api_key: str, base_url: str = "https://openrouter.ai/api/v1") -> list[dict]:
        """Fetch models from OpenAI-compatible API (OpenRouter, Minimax, etc)."""
        if not api_key:
            return []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if resp.status_code != 200:
                    logger.error(f"OpenAI-compatible API error ({base_url}): {resp.status_code}")
                    return []
                data = resp.json()
                models = []
                for m in data.get("data", []):
                    models.append({"id": m["id"], "name": m.get("name", m["id"])})
                return models
        except Exception as e:
            logger.error(f"OpenAI-compatible fetch error ({base_url}): {e}")
            return []

    async def _fetch_opencode_models(api_key: str) -> list[dict]:
        """Fetch models from OpenCode CLI."""
        auth_dir = OPENCODE_AUTH.parent
        auth_dir.mkdir(parents=True, exist_ok=True)
        auth_data = {}
        if api_key:
            auth_data["apiKey"] = api_key
        with open(OPENCODE_AUTH, "w") as f:
            json.dump(auth_data, f)

        env = os.environ.copy()
        if api_key:
            env["OPENCODE_API_KEY"] = api_key

        proc = await asyncio.create_subprocess_shell(
            "opencode models",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            return []

        output = stdout.decode().strip()
        if not output:
            return []

        models = []
        for line in output.split("\n"):
            line = line.strip()
            if line:
                models.append({"id": line, "name": line})
        return models

    # ── API: Callback ──────────────────────────────────────────────

    @app.post("/api/callback")
    async def callback(
        request: Request,
        taskId: int = Query(None),
        status: str = Query(None),
        question: Optional[str] = Query(None),
        summary: Optional[str] = Query(None),
    ):
        """Callback endpoint. Saves AI output as comment, then triggers post-processing."""
        if taskId is not None and status is not None:
            payload = CallbackPayload(taskId=taskId, status=status, question=question, summary=summary)
        else:
            try:
                body = await request.json()
                payload = CallbackPayload(**body)
            except Exception:
                raise HTTPException(422, "Invalid callback payload")
        async with async_session() as session:
            task = await session.get(Task, payload.taskId)
            if not task:
                raise HTTPException(404, "Task not found")
            await session.refresh(task)
            if task.board_column != "running":
                return {"ok": True, "message": f"Ignored: task already in '{task.board_column}' state"}

            if payload.status == "blocked" and payload.question:
                task.board_column = "blocked"
                session.add(TaskComment(task_id=task.id, author="AI", content=payload.question))
            elif payload.status == "review":
                # Collect AI output for comment
                ai_output = ""
                if payload.summary:
                    ai_output = f"**Summary:** {payload.summary}"
                workdir = Path(f"/tmp/soda-task-workdirs/task-{payload.taskId}")
                stdout_log = workdir / ".soda-stdout.log"
                stderr_log = workdir / ".soda-stderr.log"
                if stdout_log.exists():
                    try:
                        t = stdout_log.read_text().strip()
                        if t:
                            ai_output += f"\n\n**AI Output:**\n{t[:3000]}"
                    except Exception:
                        pass
                if stderr_log.exists():
                    try:
                        t = stderr_log.read_text().strip()
                        if t:
                            ai_output += f"\n\n**Stderr:**\n{t[:500]}"
                    except Exception:
                        pass
                if workdir.exists():
                    files = [f for f in workdir.iterdir() if not f.name.startswith(".soda-")]
                    if files:
                        file_list = "\n".join(f"  • {f.name}{'/' if f.is_dir() else ''}" for f in sorted(files)[:30])
                        ai_output += f"\n\n**Files created:**\n{file_list}"
                if ai_output:
                    session.add(TaskComment(task_id=task.id, author="AI", content=ai_output))

            # Close process file descriptors only after the process has exited
            if payload.taskId in running_processes:
                proc_info = running_processes[payload.taskId]
                if isinstance(proc_info, tuple):
                    proc, stdout_fd, stderr_fd = proc_info
                    if proc.returncode is not None:
                        try:
                            stdout_fd.close()
                            stderr_fd.close()
                        except Exception:
                            pass
                        del running_processes[payload.taskId]
                        logger.info(f"Task {payload.taskId}: Process exited with code {proc.returncode}, triggering post-processing")
                        asyncio.create_task(_post_process_task(payload.taskId))
                    else:
                        logger.info(f"Task {payload.taskId}: Process still running, post-processing deferred to monitor")
                else:
                    # Legacy format - remove and trigger
                    del running_processes[payload.taskId]
                    asyncio.create_task(_post_process_task(payload.taskId))
            
            await session.commit()
        return {"ok": True}

    # ── API: Git Commit & Push ─────────────────────────────────────

    async def _get_setting(session, key: str, default: str = "") -> str:
        """Helper to get a global setting value."""
        result = await session.execute(
            sa_select(GlobalSetting).where(GlobalSetting.key == key)
        )
        setting = result.scalar_one_or_none()
        return setting.value if setting else default

    async def _create_github_pr(
        session,
        task: Task,
        project: "Project",
        workdir: Path,
        username: str,
        token: str,
        repo_name: str,
        default_branch: str,
    ) -> Optional[str]:
        """Create a GitHub PR for a reviewed task. Returns PR URL or None."""
        import logging
        logger = logging.getLogger(__name__)

        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }

        # Determine the project's repo name from the task's git state or use default
        result = await session.execute(
            sa_select(TaskGitState).where(TaskGitState.task_id == task.id)
        )
        git_state = result.scalar_one_or_none()

        # Use the project's repo (from git_state) or fall back to default_repo
        target_repo = git_state.repo if git_state and git_state.repo else repo_name
        target_branch = git_state.branch if git_state and git_state.branch else default_branch

        # Create a feature branch for this task
        feature_branch = f"task-{task.id}-{re.sub(r'[^a-z0-9-]', '', task.title.lower().replace(' ', '-'))[:50]}"

        # Clone or update the repo
        repo_workdir = Path(f"/tmp/soda-pr-workdirs/task-{task.id}")
        repo_workdir.parent.mkdir(parents=True, exist_ok=True)

        repo_url = f"https://{username}:{token}@github.com/{username}/{target_repo}.git"

        try:
            if repo_workdir.exists():
                repo = git.Repo(repo_workdir)
                origin = repo.remotes.origin
                origin.fetch()
                # Checkout the target branch
                try:
                    repo.git.checkout(target_branch)
                except Exception:
                    repo.git.checkout('-b', target_branch)
                origin.pull()
            else:
                repo = git.Repo.clone_from(repo_url, repo_workdir)
                try:
                    repo.git.checkout(target_branch)
                except Exception:
                    repo.git.checkout('-b', target_branch)

            # Create feature branch
            repo.git.checkout('-b', feature_branch)

            # Copy task workdir contents to repo (excluding .soda-* logs)
            if workdir.exists():
                for item in workdir.iterdir():
                    if item.name.startswith(".soda-"):
                        continue
                    dest = repo_workdir / item.name
                    if item.is_dir():
                        if dest.exists():
                            import shutil
                            shutil.rmtree(dest)
                        import shutil
                        shutil.copytree(item, dest)
                    else:
                        import shutil
                        shutil.copy2(item, dest)

            # Also create/update task info file
            task_info_file = repo_workdir / f"task-{task.id}-info.md"
            task_info = f"""# Task {task.id}: {task.title}

**Status:** {task.board_column}
**Created:** {task.created_at}
**Description:**
{task.description or 'No description'}

---
*Auto-generated by Soda*
"""
            task_info_file.write_text(task_info)

            # Commit and push
            repo.git.add(A=True)
            if repo.is_dirty() or repo.untracked_files:
                commit_msg = f"feat: task {task.id} - {task.title}"
                repo.index.commit(commit_msg)
                repo.git.push('origin', feature_branch)

                # Create PR via GitHubService
                github_service = GitHubService(username, token)
                pr_result = await github_service.create_pull_request(
                    repo_name=target_repo,
                    title=f"Task {task.id}: {task.title}",
                    head=feature_branch,
                    base=target_branch,
                    body=f"## Task {task.id}: {task.title}\n\n{task.description or 'No description'}\n\n**Complexity:** {task.complexity or 'N/A'}\n\n*Created by Soda*"
                )
                
                if pr_result['success']:
                    logger.info(f"Created PR: {pr_result['pr_url']}")
                    return pr_result['pr_url']
                else:
                    logger.error(f"PR creation failed: {pr_result['error']}")
                    return None
            else:
                logger.info(f"No changes to commit for task {task.id}")
                return None

        except Exception as e:
            logger.error(f"Error creating PR for task {task.id}: {e}")
            return None

    async def _ensure_github_repo(owner: str, repo_name: str, token: str, private: bool = True) -> dict:
        """Ensure the GitHub repository exists, create if it doesn't.
        Also commits a .gitignore file to the repo."""
        gh_service = GitHubService(owner, token)
        
        # Check if repository exists
        repo_exists = await gh_service.check_repo_exists(repo_name)
        
        if repo_exists:
            # Commit .gitignore if not present
            await gh_service.commit_gitignore(repo_name)
            return {"status": "exists", "data": {"name": repo_name, "html_url": f"https://github.com/{owner}/{repo_name}"}}
        
        # Create the repository
        result = await gh_service.create_repository(repo_name, private=private, auto_init=True)
        
        if result["success"]:
            # Commit .gitignore to the newly created repo
            await gh_service.commit_gitignore(repo_name)
            return {"status": "created", "data": result["data"]}
        else:
            return {"status": "error", "message": result["error"]}

    @app.post("/api/tasks/{task_id}/merge")
    async def merge_task_branch(task_id: int, payload: Optional[dict] = None):
        """Merge a task's branch into main and mark task as done.
        Payload (optional): {comment: str, move_back_to: 'running'|'blocked'}
        """
        payload = payload or {}
        move_back_to = payload.get("move_back_to")
        comment = payload.get("comment", "")

        async with async_session() as session:
            task = await session.get(Task, task_id)
            if not task:
                raise HTTPException(404, "Task not found")
            if task.board_column != "review":
                raise HTTPException(400, "Task must be in 'review' to merge")

            # Get project
            project = await session.get(Project, task.project_id)
            if not project or not project.repo_name:
                raise HTTPException(400, "Project has no repository")

            # Get git settings
            username = await get_setting("git_username", "")
            token = await get_setting("git_token", "")
            default_branch = await get_setting("git_default_branch", "main")

            if not username or not token:
                raise HTTPException(400, "Git credentials not configured in Settings")

            # Get task branch from git state
            branch_result = await session.execute(
                sa_select(TaskGitState).where(TaskGitState.task_id == task_id)
            )
            git_state = branch_result.scalar_one_or_none()
            task_branch = git_state.branch if git_state else f"task-{task_id}"

            gh = GitHubService(username, token)
            result = await gh.merge_branch(project.repo_name, task_branch, default_branch)

            if result.get("status") == "merged":
                task.board_column = "done"
                session.add(TaskComment(
                    task_id=task_id,
                    author="user",
                    content=f"✅ Merged into {default_branch}\n\n{comment}".strip(),
                ))
                await session.commit()
                return {"ok": True, "column": "done", "merged": True}
            elif result.get("status") == "conflict":
                # Move back to specified column (running or blocked) with conflict info
                target = move_back_to if move_back_to in ("running", "blocked") else "blocked"
                task.board_column = target
                session.add(TaskComment(
                    task_id=task_id,
                    author="user",
                    content=f"⚠️ Merge conflict\n\n{comment}\n\nConflict details: {result.get('message', '')}".strip(),
                ))
                await session.commit()
                return {"ok": False, "column": target, "merged": False, "error": result.get("message", "Conflict")}
            else:
                raise HTTPException(500, f"Merge failed: {result.get('message', 'Unknown error')}")

    @app.post("/api/tasks/{task_id}/git-commit")
    async def git_commit_push(
        task_id: int,
        commit_message: str = Form(...),
        repo_override: str = Form(""),
        branch_override: str = Form("")
    ):
        """Commit and push task workdir changes to GitHub."""
        async with async_session() as session:
            # Get task
            task = await session.get(Task, task_id)
            if not task:
                raise HTTPException(404, "Task not found")
            
            if task.board_column != "done":
                raise HTTPException(400, "Task must be in 'done' column to commit")
            
            # Get git settings
            username = await _get_setting(session, "git_username")
            token = await _get_setting(session, "git_token")
            default_repo = await _get_setting(session, "git_default_repo")
            default_branch = await _get_setting(session, "git_default_branch", "main")
            
            if not username or not token:
                raise HTTPException(400, "Git username and token must be configured in Settings")
            
            # Determine repo and branch
            repo_name = repo_override if repo_override else default_repo
            branch = branch_override if branch_override else default_branch
            
            if not repo_name:
                raise HTTPException(400, "Repository must be specified or set as default in Settings")
            
            # Ensure repo exists
            ensure_result = await _ensure_github_repo(username, repo_name, token)
            if ensure_result["status"] == "error":
                raise HTTPException(400, ensure_result["message"])
            
            # Get or create task git state
            result = await session.execute(
                sa_select(TaskGitState).where(TaskGitState.task_id == task_id)
            )
            git_state = result.scalar_one_or_none()
            
            if not git_state:
                git_state = TaskGitState(task_id=task_id)
                session.add(git_state)
            
            # Setup local workdir
            workdir_base = Path("/tmp/soda-git-workdirs")
            workdir_base.mkdir(parents=True, exist_ok=True)
            workdir = workdir_base / f"task-{task_id}"
            
            repo_url = f"https://{username}:{token}@github.com/{username}/{repo_name}.git"
            
            try:
                # Clone or update repo
                if workdir.exists():
                    repo = git.Repo(workdir)
                    origin = repo.remotes.origin
                    origin.fetch()
                    
                    # Checkout branch
                    if branch in repo.branches:
                        repo.git.checkout(branch)
                    else:
                        repo.git.checkout('-b', branch)
                    
                    origin.pull()
                else:
                    repo = git.Repo.clone_from(repo_url, workdir)
                    
                    # Checkout or create branch
                    try:
                        repo.git.checkout(branch)
                    except git.exc.GitCommandError:
                        repo.git.checkout('-b', branch)
                
                # Create task info file
                task_info_file = workdir / f"task-{task_id}-info.md"
                task_info = f"""# Task {task_id}: {task.title}

**Status:** {task.board_column}
**Created:** {task.created_at}
**Description:**
{task.description or 'No description'}

---
*Auto-generated by Soda*
"""
                task_info_file.write_text(task_info)
                
                # Git operations
                repo.git.add(A=True)
                
                if repo.is_dirty() or repo.untracked_files:
                    commit = repo.index.commit(commit_message)
                    repo.git.push('origin', branch)
                    
                    git_state.repo = repo_name
                    git_state.branch = branch
                    git_state.workdir = str(workdir)
                    git_state.last_commit = commit.hexsha
                    git_state.last_pushed_at = datetime.utcnow()
                    
                    await session.commit()
                    
                    return {
                        "ok": True,
                        "commit": commit.hexsha[:8],
                        "repo": f"{username}/{repo_name}",
                        "branch": branch,
                        "message": "Successfully committed and pushed"
                    }
                else:
                    return {
                        "ok": True,
                        "commit": None,
                        "repo": f"{username}/{repo_name}",
                        "branch": branch,
                        "message": "No changes to commit"
                    }
                    
            except git.exc.GitCommandError as e:
                raise HTTPException(500, f"Git error: {str(e)}")
            except Exception as e:
                raise HTTPException(500, f"Unexpected error: {str(e)}")

    # ── API: Running processes ─────────────────────────────────────

    @app.get("/api/tasks/{task_id}/process")
    async def get_process_status(task_id: int):
        """Get the running process status for a task."""
        proc_info = running_processes.get(task_id)
        if not proc_info:
            return {"running": False, "exit_code": None}
        
        if isinstance(proc_info, tuple):
            proc, stdout_fd, stderr_fd = proc_info
        else:
            proc = proc_info
        
        return {
            "running": proc.returncode is None,
            "exit_code": proc.returncode,
            "pid": proc.pid if hasattr(proc, 'pid') else None,
        }

    @app.post("/api/tasks/{task_id}/kill")
    async def kill_task_process(task_id: int):
        """Kill a running process for a task and move it to blocked."""
        import logging
        logger = logging.getLogger(__name__)
        
        proc_info = running_processes.get(task_id)
        if not proc_info:
            raise HTTPException(404, "No running process for this task")
        
        if isinstance(proc_info, tuple):
            proc, stdout_fd, stderr_fd = proc_info
            try:
                stdout_fd.close()
                stderr_fd.close()
            except Exception:
                pass
        else:
            proc = proc_info
        
        if proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
                logger.info(f"Killed process for task {task_id}")
            except Exception as e:
                logger.error(f"Error killing process for task {task_id}: {e}")
        
        del running_processes[task_id]
        
        # Move task to blocked
        async with async_session() as session:
            task = await session.get(Task, task_id)
            if task:
                task.board_column = "blocked"
                comment = TaskComment(
                    task_id=task.id,
                    author="Soda",
                    content="⚠️ Process was killed — task moved to blocked. The AI process may have crashed or timed out."
                )
                session.add(comment)
                await session.commit()
                return {"ok": True, "message": "Process killed, task moved to blocked"}
        
        raise HTTPException(404, "Task not found")

    # ── API: Health check ──────────────────────────────────────────

    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok", "service": "soda"}

    # ── Background Watchdog ─────────────────────────────────────────
    import logging
    _watchdog_logger = logging.getLogger("watchdog")
    
    async def _watchdog_check():
        """Periodically check running tasks for stuck processes.
        Also cleans up stale ideas and running tasks."""
        # Grace period: tasks started less than this many seconds ago are not checked.
        # Must be > git clone timeout (180s) + a buffer for the AI process to start.
        GRACE_PERIOD_SEC = 300  # 5 minutes
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            try:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)

                # Clean up stale ideas: stuck in 'generating' for too long
                async with async_session() as session:
                    stale_ideas = await session.execute(
                        sa_select(Idea).where(
                            Idea.status == "generating",
                        )
                    )
                    for idea in stale_ideas.scalars().all():
                        is_running = idea.id in generation_tasks
                        if not is_running:
                            logger.warning(f"Stale idea {idea.id} in 'generating' with no background task — resetting")
                            idea.status = "active"
                            idea.generation_error = (
                                "Generation was interrupted (server restart or crash). You can try again."
                            )
                            idea.pending_questions = None
                    await session.commit()

                async with async_session() as session:
                    result = await session.execute(
                        sa_select(Task).where(Task.board_column == "running")
                    )
                    running_tasks = result.scalars().all()

                    for task in running_tasks:
                        proc_info = running_processes.get(task.id)

                        # Process already exited → post-process immediately (don't wait for grace period)
                        if proc_info and isinstance(proc_info, tuple) and len(proc_info) >= 1:
                            proc = proc_info[0]
                            if proc is not None and proc.returncode is not None:
                                _watchdog_logger.info(
                                    f"Task {task.id} process exited with code {proc.returncode}, running post-processing"
                                )
                                try:
                                    if len(proc_info) >= 3:
                                        proc_info[1].close()
                                        proc_info[2].close()
                                except Exception:
                                    pass
                                del running_processes[task.id]
                                asyncio.create_task(_post_process_task(task.id))
                                continue

                        # Grace period: skip stuck-with-no-process checks for recently started tasks
                        if task.updated_at and (now - task.updated_at).total_seconds() < GRACE_PERIOD_SEC:
                            continue

                        # Case 0: Task is in "starting" state (process not yet launched)
                        # Give it extra time: up to STARTING_TIMEOUT_SEC
                        STARTING_TIMEOUT_SEC = 600  # 10 minutes for git clone + process start
                        if proc_info and isinstance(proc_info, tuple) and len(proc_info) >= 2 and proc_info[1] == "starting":
                            try:
                                started_at = proc_info[2]
                                from datetime import datetime as _dtp
                                started_dt = _dtp.fromisoformat(started_at)
                                if started_dt.tzinfo is None:
                                    started_dt = started_dt.replace(tzinfo=timezone.utc)
                                elapsed = (now - started_dt).total_seconds()
                                if elapsed < STARTING_TIMEOUT_SEC:
                                    # Still within starting window — skip
                                    continue
                                # Exceeded starting timeout — process never started
                                _watchdog_logger.warning(f"Task {task.id} stuck in 'starting' for {elapsed:.0f}s, moving to blocked")
                                task.board_column = "blocked"
                                session.add(TaskComment(
                                    task_id=task.id,
                                    author="Soda",
                                    content=(
                                        f"⚠️ Watchdog: Task stuck in 'starting' for {elapsed:.0f}s. "
                                        f"The AI process never started (git clone may have timed out, "
                                        f"or the execute_command failed). Task moved to blocked."
                                    ),
                                ))
                                # Clean up the starting sentinel
                                running_processes.pop(task.id, None)
                                continue
                            except Exception as e:
                                logger.warning(f"Watchdog starting-state check failed for task {task.id}: {e}")
                                continue

                        # Case 1: Task is running but no process tracked
                        if not proc_info:
                            if task.id in _processing_tasks:
                                continue
                            if await _task_has_ai_output(task.id):
                                _watchdog_logger.info(
                                    f"Task {task.id} finished (AI output present) but not post-processed — recovering"
                                )
                                async with async_session() as recover_session:
                                    t = await recover_session.get(Task, task.id)
                                    if t and t.board_column == "running":
                                        pass  # keep running for post-process
                                asyncio.create_task(_post_process_task(task.id))
                                continue

                            # Try to read stderr for diagnostic info
                            workdir = Path(f"/tmp/soda-task-workdirs/task-{task.id}")
                            stderr_content = ""
                            stdout_content = ""
                            try:
                                err_log = workdir / ".soda-stderr.log"
                                out_log = workdir / ".soda-stdout.log"
                                if err_log.exists():
                                    stderr_content = err_log.read_text()[-2000:]  # last 2KB
                                if out_log.exists():
                                    stdout_content = out_log.read_text()[-2000:]
                            except Exception:
                                pass

                            diag = ""
                            if stderr_content:
                                diag = f"\n\n**Stderr (last 2KB):**\n```\n{stderr_content}\n```"
                            if stdout_content:
                                diag += f"\n\n**Stdout (last 2KB):**\n```\n{stdout_content}\n```"

                            _watchdog_logger.warning(f"Task {task.id} has no running process, moving to blocked")
                            task.board_column = "blocked"
                            session.add(TaskComment(
                                task_id=task.id,
                                author="Soda",
                                content=(
                                    f"⚠️ Watchdog: No running process found after {GRACE_PERIOD_SEC}s. "
                                    f"It may have crashed or failed to start. Task moved to blocked.{diag}"
                                ),
                            ))
                            continue
                        
                        # Case 2: Process has exited but task still running → stuck
                        if isinstance(proc_info, tuple):
                            proc = proc_info[0]
                        else:
                            proc = proc_info
                        
                        if proc.returncode is not None:
                            # Process exited — run post-processing (git/PR/callback)
                            _watchdog_logger.info(f"Task {task.id} process exited with code {proc.returncode}, running post-processing")
                            # Clean up process tracker
                            if isinstance(proc_info, tuple):
                                try:
                                    proc_info[1].close()
                                    proc_info[2].close()
                                except Exception:
                                    pass
                            del running_processes[task.id]
                            # Run post-processing in background (idempotency guard prevents duplicates)
                            asyncio.create_task(_post_process_task(task.id))
                    
                    await session.commit()
            except Exception as e:
                _watchdog_logger.error(f"Watchdog error: {e}")

    # ── Static files ───────────────────────────────────────────────

    autopilot.configure(_pipeline_start_task, _verify_project_completion)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


app = create_app()
