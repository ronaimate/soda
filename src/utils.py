"""
Utility functions and helpers for Soda application.
"""
import json
import logging
from pathlib import Path
from typing import Any, Optional, TypeVar, Sequence

from fastapi import HTTPException
from sqlalchemy import select as sa_select

from .database import GlobalSetting, UserDefaultSize, async_session

T = TypeVar("T")

logger = logging.getLogger("soda.utils")


async def get_or_404(session: Any, model: type[T], id: Any, entity_name: Optional[str] = None) -> T:
    """Get entity by ID or raise 404 if not found."""
    entity = await session.get(model, id)
    if not entity:
        name = entity_name or model.__name__
        raise HTTPException(404, f"{name} not found")
    return entity


async def get_setting(key: str, default: str = "") -> str:
    """Get a setting value from the database."""
    async with async_session() as session:
        result = await session.execute(
            sa_select(GlobalSetting).where(GlobalSetting.key == key)
        )
        setting = result.scalar_one_or_none()
        return setting.value if setting and setting.value else default


async def get_opencode_api_key() -> str:
    """Get OpenCode API key from settings."""
    return await get_setting("opencode_api_key", "")


def write_opencode_auth(api_key: str, provider: Optional[str] = None, model: Optional[str] = None) -> bool:
    """
    Write OpenCode authentication file.
    
    Args:
        api_key: API key for OpenCode
        provider: Optional provider name
        model: Optional model name
        
    Returns:
        True if auth file was written, False if no API key provided
    """
    if not api_key:
        logger.debug("No API key provided, skipping auth file write")
        return False
    
    auth_path = Path("/root/.local/share/opencode/auth.json")
    auth_path.parent.mkdir(parents=True, exist_ok=True)

    p = provider or "opencode"
    auth_data: dict = {
        "credentials": [{"provider": p, "key": api_key}],
    }
    if model:
        auth_data["model"] = model
    if p != "opencode":
        auth_data["provider"] = p
    
    try:
        with open(auth_path, 'w') as f:
            json.dump(auth_data, f, indent=2)
        logger.info(f"OpenCode auth file written to {auth_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write OpenCode auth file: {e}")
        return False


# ── User Default Sizes ────────────────────────────────────────────────

VALID_SIZES = ["XS", "S", "M", "L", "XL"]


async def get_user_default_sizes(session: Any, user_id: int) -> list[str]:
    """Get all default polo sizes for a user (from User.task_types array)."""
    from .database import User
    result = await session.execute(
        sa_select(User.task_types).where(User.id == user_id)
    )
    row = result.first()
    if not row or not row[0]:
        return []
    return [s for s in row[0] if s]


async def set_user_default_sizes(session: Any, user_id: int, sizes: Sequence[str]) -> None:
    """
    Set default polo sizes for a user.
    Updates the User.task_types ARRAY column (the v2 source of truth).
    Also keeps UserDefaultSize table in sync for backward compatibility.
    """
    from .database import User
    
    # Validate sizes
    for size in sizes:
        if size not in VALID_SIZES:
            raise HTTPException(400, f"Invalid size: {size}. Must be one of {VALID_SIZES}")
    
    # Lowercase all sizes (v2 stores them lowercase: xs, s, m, l, xl)
    sizes_lower = [s.lower() for s in sizes]
    
    # Check for conflicts with other users
    if sizes_lower:
        conflict_result = await session.execute(
            sa_select(User.id, User.task_types).where(
                User.id != user_id,
                User.type == "ai",
                User.task_types.overlap(sizes_lower),
            )
        )
        conflicts = conflict_result.all()
        if conflicts:
            conflict_desc = ", ".join(
                f"size '{s}' already assigned to user_id {uid}"
                for uid, user_types in conflicts
                for s in user_types
                if s in sizes_lower
            )
            raise HTTPException(400, f"Size conflict: {conflict_desc}")
    
    # Update User.task_types (the v2 source of truth)
    user = await session.get(User, user_id)
    if user:
        user.task_types = sizes_lower
    
    # Also keep UserDefaultSize in sync for backward compat
    await session.execute(
        UserDefaultSize.__table__.delete().where(
            UserDefaultSize.user_id == user_id,
        )
    )
    for size in sizes:
        session.add(UserDefaultSize(user_id=user_id, size=size))


async def find_user_by_size(session: Any, size: str) -> Optional[int]:
        """Find which user is assigned to a given task type/size. Returns user_id or None.
        Uses the user.task_types ARRAY column."""
        from .database import User
        size_lower = size.lower()
        result = await session.execute(
            sa_select(User).where(
                User.task_types.contains([size_lower]),
                User.type == "ai",
            )
        )
        user = result.scalars().first()
        return user.id if user else None
