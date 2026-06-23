"""GitHub URL helpers — safe clone URLs and username resolution."""
from urllib.parse import quote

import httpx


def build_github_clone_url(repo_url: str, token: str) -> str:
    """Authenticated clone URL using x-access-token (safe when username is an email)."""
    base = (repo_url or "").rstrip("/")
    if not base:
        return base
    if not base.endswith(".git"):
        base += ".git"
    if "github.com/" not in base:
        return base
    token_q = quote(token, safe="")
    return base.replace("https://github.com/", f"https://x-access-token:{token_q}@github.com/")


async def resolve_github_username(token: str) -> str:
    """Resolve GitHub login from a PAT via API."""
    if not token:
        return ""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            "https://api.github.com/user",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
        if resp.status_code != 200:
            return ""
        return (resp.json().get("login") or "").strip()
