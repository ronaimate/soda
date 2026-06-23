"""Project pipeline: auto and step-by-step task execution."""
from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

from sqlalchemy import select as sa_select

from .database import Project, Task, TaskDependency, async_session

logger = logging.getLogger("soda.autopilot")

_start_task: Optional[Callable[[int], Awaitable[None]]] = None


_verify_completion: Optional[Callable[[int], Awaitable[bool]]] = None


def configure(
    start_task: Callable[[int], Awaitable[None]],
    verify_completion: Optional[Callable[[int], Awaitable[bool]]] = None,
) -> None:
    global _start_task, _verify_completion
    _start_task = start_task
    _verify_completion = verify_completion


async def find_next_runnable_task(project_id: int) -> Optional[Task]:
    """First backlog task whose dependencies are all done."""
    async with async_session() as session:
        result = await session.execute(
            sa_select(Task)
            .where(Task.project_id == project_id, Task.board_column == "backlog")
            .order_by(Task.position, Task.created_at)
        )
        candidates = result.scalars().all()
        if not candidates:
            return None

        done_result = await session.execute(
            sa_select(Task.id).where(
                Task.project_id == project_id,
                Task.board_column == "done",
            )
        )
        done_ids = {row[0] for row in done_result.all()}

        for task in candidates:
            dep_result = await session.execute(
                sa_select(TaskDependency.depends_on_id).where(
                    TaskDependency.task_id == task.id
                )
            )
            dep_ids = [row[0] for row in dep_result.all()]
            if all(d in done_ids for d in dep_ids):
                return task
        return None


async def count_progress(project_id: int) -> dict:
    async with async_session() as session:
        result = await session.execute(
            sa_select(Task.board_column).where(Task.project_id == project_id)
        )
        cols = [row[0] for row in result.all()]
        total = len(cols)
        done = sum(1 for c in cols if c == "done")
        return {"total": total, "done": done, "remaining": total - done}


async def get_pipeline_status(project_id: int) -> dict:
    async with async_session() as session:
        project = await session.get(Project, project_id)
        if not project:
            return {}
        progress = await count_progress(project_id)
        next_task = await find_next_runnable_task(project_id)
        return {
            "project_id": project_id,
            "run_mode": project.run_mode or "step",
            "pipeline_state": project.pipeline_state or "idle",
            "current_task_id": project.current_task_id,
            "advanced_mode": bool(project.advanced_mode),
            "progress": progress,
            "next_task_id": next_task.id if next_task else None,
            "next_task_title": next_task.title if next_task else None,
        }


async def init_pipeline(project_id: int, run_mode: str = "step") -> None:
    async with async_session() as session:
        project = await session.get(Project, project_id)
        if not project:
            return
        project.run_mode = run_mode if run_mode in ("auto", "step") else "step"
        project.pipeline_state = "paused"
        project.current_task_id = None
        await session.commit()


async def pipeline_pause(project_id: int) -> None:
    async with async_session() as session:
        project = await session.get(Project, project_id)
        if not project:
            return
        if project.pipeline_state == "complete":
            return
        project.pipeline_state = "paused"
        await session.commit()


async def pipeline_set_mode(project_id: int, run_mode: str) -> None:
    async with async_session() as session:
        project = await session.get(Project, project_id)
        if not project:
            return
        if run_mode in ("auto", "step"):
            project.run_mode = run_mode
        await session.commit()


async def _start_next_task(project_id: int) -> bool:
    if not _start_task:
        logger.error("Pipeline start_task hook not configured")
        return False

    next_task = await find_next_runnable_task(project_id)
    if not next_task:
        if _verify_completion:
            try:
                added = await _verify_completion(project_id)
                if added:
                    next_task = await find_next_runnable_task(project_id)
            except Exception as e:
                logger.warning(f"Completion verification failed for project {project_id}: {e}")
        if not next_task:
            async with async_session() as session:
                project = await session.get(Project, project_id)
                if project:
                    project.pipeline_state = "complete"
                    project.current_task_id = None
                    await session.commit()
            return False

    async with async_session() as session:
        project = await session.get(Project, project_id)
        if not project or project.pipeline_state == "paused":
            return False
        running = await session.execute(
            sa_select(Task).where(
                Task.project_id == project_id,
                Task.board_column == "running",
            )
        )
        if running.scalar_one_or_none():
            return False
        project.pipeline_state = "running"
        project.current_task_id = next_task.id
        await session.commit()

    await _start_task(next_task.id)
    return True


async def pipeline_next(project_id: int) -> dict:
    """Step mode: user triggers next task."""
    async with async_session() as session:
        project = await session.get(Project, project_id)
        if not project:
            raise ValueError("Project not found")
        if project.pipeline_state == "complete":
            return await get_pipeline_status(project_id)
        if project.pipeline_state == "running":
            return await get_pipeline_status(project_id)
        project.pipeline_state = "idle"

    started = await _start_next_task(project_id)
    status = await get_pipeline_status(project_id)
    status["started"] = started
    return status


async def pipeline_resume(project_id: int) -> dict:
    async with async_session() as session:
        project = await session.get(Project, project_id)
        if not project:
            raise ValueError("Project not found")
        if project.pipeline_state == "complete":
            return await get_pipeline_status(project_id)
        run_mode = project.run_mode or "step"
        project.pipeline_state = "idle"
        await session.commit()

    if run_mode == "auto":
        await _start_next_task(project_id)
    else:
        async with async_session() as session:
            p = await session.get(Project, project_id)
            if p:
                p.pipeline_state = "paused"
                await session.commit()

    return await get_pipeline_status(project_id)


async def on_task_blocked(task_id: int, reason: str = "") -> None:
    async with async_session() as session:
        task = await session.get(Task, task_id)
        if not task:
            return
        project = await session.get(Project, task.project_id)
        if not project:
            return
        project.pipeline_state = "waiting_user"
        project.current_task_id = task_id
        await session.commit()


async def on_task_completed(task_id: int) -> None:
    """Schedule next task after post-process + merge finished."""
    async with async_session() as session:
        task = await session.get(Task, task_id)
        if not task or task.board_column != "done":
            return
        project = await session.get(Project, task.project_id)
        if not project:
            return
        run_mode = project.run_mode or "step"
        state = project.pipeline_state
        project_id = project.id
        if state == "paused":
            project.current_task_id = None
            await session.commit()
            return

    if run_mode == "auto":
        await _start_next_task(project_id=project_id)
    else:
        async with async_session() as session:
            project = await session.get(Project, project_id)
            if project:
                project.pipeline_state = "paused"
                project.current_task_id = None
                await session.commit()


async def on_pipeline_task_failed(task_id: int) -> None:
    await on_task_blocked(task_id)


async def maybe_auto_start_after_generation(project_id: int) -> None:
    """Called when project is created with run_mode=auto."""
    async with async_session() as session:
        project = await session.get(Project, project_id)
        if not project:
            return
        if project.run_mode != "auto":
            project.pipeline_state = "paused"
            await session.commit()
            return
        project.pipeline_state = "idle"
        await session.commit()
    await _start_next_task(project_id)
