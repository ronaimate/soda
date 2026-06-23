"""
Operation commands and prompt generation for Soda.

Each operation (task_run, task_master, merge) has its own command template
stored in GlobalSetting. The system generates the prompt dynamically from
the task context — the user/AI assignee does NOT need to define a command.
"""
import json
import logging
from typing import Optional

from sqlalchemy import select as sa_select

from .database import (
    GlobalSetting,
    Project,
    Task,
    TaskComment,
    TaskDependency,
    User,
    async_session,
)

logger = logging.getLogger("soda.operations")


# ── Default operation commands ────────────────────────────────────────

# Callback URL template
_CALLBACK_TPL = "{{callback.url}}?taskId={{task.id}}"

# task_run: run a regular task with OpenCode
DEFAULT_TASK_RUN_CMD = (
    "opencode run '{{task.prompt}}' && "
    "curl -s -X POST '" + _CALLBACK_TPL + "&status=review' || true"
)

# task_master: architect a project (used for idea generation, not task running)
# Note: in current implementation, task_master uses _call_architect (OpenRouter/OpenCode direct),
# not a shell command. This is here for future use.
DEFAULT_TASK_MASTER_CMD = (
    "opencode run '{{task.prompt}}'"
)

# merge: merge a task's branch into the default branch
DEFAULT_MERGE_CMD = (
    "cd {{task.workdir}} && "
    "git fetch origin && "
    "git checkout {{default_branch}} && "
    "git pull origin {{default_branch}} && "
    "git merge --no-ff {{task.branch}} -m '{{merge.message}}' && "
    "git push origin {{default_branch}}"
)


# ── Operation command storage ──────────────────────────────────────────

OPERATION_KEYS = {
    "task_run": ("op_cmd_task_run", DEFAULT_TASK_RUN_CMD),
    "task_master": ("op_cmd_task_master", DEFAULT_TASK_MASTER_CMD),
    "merge": ("op_cmd_merge", DEFAULT_MERGE_CMD),
}


async def get_operation_command(op: str) -> str:
    """Get the command template for an operation. Falls back to default."""
    if op not in OPERATION_KEYS:
        raise ValueError(f"Unknown operation: {op}")
    key, default = OPERATION_KEYS[op]
    async with async_session() as session:
        result = await session.execute(
            sa_select(GlobalSetting).where(GlobalSetting.key == key)
        )
        setting = result.scalar_one_or_none()
        if setting and setting.value:
            return setting.value
    return default


async def set_operation_command(op: str, value: str) -> None:
    """Set the command template for an operation."""
    if op not in OPERATION_KEYS:
        raise ValueError(f"Unknown operation: {op}")
    key, default = OPERATION_KEYS[op]
    async with async_session() as session:
        result = await session.execute(
            sa_select(GlobalSetting).where(GlobalSetting.key == key)
        )
        setting = result.scalar_one_or_none()
        if setting:
            setting.value = value
        else:
            session.add(GlobalSetting(key=key, value=value))
        await session.commit()


async def get_all_operation_commands() -> dict:
    """Get all operation commands (for Settings UI)."""
    out = {}
    for op, (key, default) in OPERATION_KEYS.items():
        async with async_session() as session:
            result = await session.execute(
                sa_select(GlobalSetting).where(GlobalSetting.key == key)
            )
            setting = result.scalar_one_or_none()
            out[op] = {
                "key": key,
                "value": setting.value if setting and setting.value else default,
                "default": default,
            }
    return out


# ── Prompt generation ─────────────────────────────────────────────────


async def generate_task_prompt(
    task: Task,
    project: Project,
    comments: list[dict],
    depends_on_ids: Optional[list] = None,
) -> str:
    """
    Generate a full prompt for a task run.
    Includes: project context, task details, other tasks (to prevent overlap), comments.

    Template variables available in the command:
    - {{task.id}}, {{task.title}}, {{task.description}}, {{task.complexity}}
    - {{task.workdir}}
    - {{project.name}}, {{project.description}}
    - {{task.comments}} (JSON string)
    - {{callback.url}}
    - {{task.prompt}} (this full prompt, as a single string)
    """
    async with async_session() as session:
        # Fetch remaining tasks in the same project
        result = await session.execute(
            sa_select(Task).where(
                Task.project_id == task.project_id,
                Task.id != task.id,
            ).order_by(Task.id)
        )
        remaining_tasks = result.scalars().all()
        remaining_task_ids = [t.id for t in remaining_tasks]

        # Fetch dependencies among remaining tasks
        remaining_deps = {}
        if remaining_task_ids:
            dep_result = await session.execute(
                sa_select(TaskDependency.task_id, TaskDependency.depends_on_id)
                .where(TaskDependency.task_id.in_(remaining_task_ids))
            )
            for tid, dep_id in dep_result.all():
                remaining_deps.setdefault(tid, []).append(dep_id)

        task_titles = {t.id: t.title for t in remaining_tasks}

        def _status_icon(col: str) -> str:
            return {
                "backlog": "📋",
                "running": "▶️",
                "blocked": "❓",
                "review": "👁️",
                "done": "✅",
            }.get(col, "•")

        def _deps_str(t: Task) -> str:
            deps = remaining_deps.get(t.id, [])
            if not deps:
                return ""
            dep_titles = [task_titles.get(d, f"#{d}") for d in deps]
            return f" *(depends on: {', '.join(dep_titles)})*"

        remaining_summary = "\n".join(
            f"- {_status_icon(t.board_column)} [{t.board_column}] **{t.title}**: {t.description or '(no description)'}{_deps_str(t)}"
            for t in remaining_tasks
        )

    return f"""You are working on a software project.

## Project: {project.name}
{project.description or ''}

## Your Task (ONLY work on this):
**Title:** {task.title}
**Description:** {task.description or '(no description)'}
**Complexity:** {task.complexity or 'not specified'}

## Other tasks in this project (DO NOT work on these — they are separate tasks):
{remaining_summary if remaining_summary else '(none)'}

## Existing comments on this task:
{json.dumps(comments, indent=2) if comments else '(none)'}

## Work Instructions:
- ONLY implement what is described in "Your Task" above
- Do NOT work on any of the other tasks listed above
- If a task in the list above depends on yours (i.e. your task must complete first), still focus on your own work — do not start the dependent task
- If a task above is already done (✅), review what was implemented to keep your work consistent
- Work ONLY in the current directory
- Create/edit files directly in this directory
- Do NOT call any callback URL — the system handles that automatically
- If you cannot complete the task, describe what is blocking you as the last line of your output
"""
