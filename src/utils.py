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
    
    auth_path = Path("/root/.opencode/auth.json")
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    
    auth_data = {
        "api_key": api_key,
        "provider": provider or "openai",
        "model": model or "gpt-4"
    }
    
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
    """Get all default polo sizes for a user."""
    result = await session.execute(
        sa_select(UserDefaultSize.size).where(UserDefaultSize.user_id == user_id)
    )
    return [row[0] for row in result.all()]


async def set_user_default_sizes(session: Any, user_id: int, sizes: Sequence[str]) -> None:
    """
    Set default polo sizes for a user.
    - Removes old sizes not in the new list
    - Adds new sizes
    - Validates that sizes are valid (XS/S/M/L/XL)
    - Validates that no other user already has the same size
    """
    # Validate sizes
    for size in sizes:
        if size not in VALID_SIZES:
            raise HTTPException(400, f"Invalid size: {size}. Must be one of {VALID_SIZES}")
    
    # Check for conflicts with other users
    if sizes:
        conflict_result = await session.execute(
            sa_select(UserDefaultSize.user_id, UserDefaultSize.size).where(
                UserDefaultSize.size.in_(sizes),
                UserDefaultSize.user_id != user_id,
            )
        )
        conflicts = conflict_result.all()
        if conflicts:
            # Get user names for better error message
            user_result = await session.execute(
                sa_select(GlobalSetting).where(GlobalSetting.key == "users_by_id")
            )
            conflict_users = [(row[0], row[1]) for row in conflicts]
            conflict_desc = ", ".join(
                f"size '{size}' already assigned to user_id {uid}"
                for uid, size in conflict_users
            )
            raise HTTPException(400, f"Size conflict: {conflict_desc}")
    
    # Remove old sizes not in new list
    await session.execute(
        UserDefaultSize.__table__.delete().where(
            UserDefaultSize.user_id == user_id,
            ~UserDefaultSize.size.in_(sizes) if sizes else True,
        )
    )
    
    # Add new sizes
    existing_result = await session.execute(
        sa_select(UserDefaultSize.size).where(UserDefaultSize.user_id == user_id)
    )
    existing_sizes = {row[0] for row in existing_result.all()}
    
    for size in sizes:
        if size not in existing_sizes:
            session.add(UserDefaultSize(user_id=user_id, size=size))


async def find_user_by_size(session: Any, size: str) -> Optional[int]:
        """Find which user is assigned to a given task type/size. Returns user_id or None.
        Uses the user.task_types ARRAY column."""
        from .database import User
        size_lower = size.lower()
        result = await session.execute(
            sa_select(User).where(User.task_types.contains([size_lower]))
        )
        user = result.scalars().first()
        return user.id if user else None
