import asyncio
import json
import os
import re
import subprocess
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import git
import httpx
from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .database import (
    GlobalSetting, Idea, Project, Task, TaskComment, TaskDependency, User,
    TaskGitState, async_session, init_db, sa_select,
)

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
OPENCODE_AUTH = Path("/root/.local/share/opencode/auth.json")

# In-memory process tracker: task_id -> asyncio.subprocess.Process
running_processes: dict[int, asyncio.subprocess.Process] = {}


# ─── Pydantic models ────────────────────────────────────────────────

class CallbackPayload(BaseModel):
    taskId: int
    status: str  # "blocked" | "review"
    question: Optional[str] = None
    summary: Optional[str] = None


class TaskMovePayload(BaseModel):
    column: str


class CommentPayload(BaseModel):
    content: str
    author: str = "user"


# ─── Lifespan ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


# ─── App factory ────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(title="Soda", lifespan=lifespan)

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

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
        """Write AI user's API key and model to OpenCode auth.json.
        Falls back to global OpenCode API key if user has no key set."""
        auth_dir = OPENCODE_AUTH.parent
        auth_dir.mkdir(parents=True, exist_ok=True)
        auth_data = {}
        if user.api_key:
            auth_data["apiKey"] = user.api_key
        if user.provider:
            auth_data["provider"] = user.provider
        if user.model:
            auth_data["model"] = user.model
        # If user has no API key, don't overwrite auth.json — let the
        # global OPENCODE_API_KEY env var (set by callers) handle auth.
        if not user.api_key:
            return
        with open(OPENCODE_AUTH, "w") as f:
            json.dump(auth_data, f)

    # ── Helper: get OpenCode API key from settings ─────────────────

    async def _get_opencode_api_key() -> str:
        """Get OpenCode API key from global settings."""
        async with async_session() as session:
            result = await session.execute(
                sa_select(GlobalSetting).where(GlobalSetting.key == "opencode_api_key")
            )
            setting = result.scalar_one_or_none()
            return (setting.value or "").strip() if setting else ""

    # ── Helper: run execute command ─────────────────────────────────

    async def _run_execute_command(task: Task, assignee: User) -> None:
        """Run the AI user's execute command as a subprocess."""
        if not assignee.execute_command:
            return

        # Write AI user's auth to OpenCode config
        _write_opencode_auth(assignee)

        # Get OpenCode API key from settings
        opencode_api_key = await _get_opencode_api_key()

        # Build comments JSON
        async with async_session() as session:
            result = await session.execute(
                sa_select(TaskComment).where(TaskComment.task_id == task.id).order_by(TaskComment.created_at)
            )
            comments = [
                {"author": c.author, "content": c.content, "created_at": str(c.created_at)}
                for c in result.scalars().all()
            ]

        # Get callback URL and project name from settings
        async with async_session() as session:
            result = await session.execute(
                sa_select(GlobalSetting).where(GlobalSetting.key == "callback_url")
            )
            setting = result.scalar_one_or_none()
            callback_url = setting.value if setting else "http://localhost:8000/api/callback"

        async with async_session() as session:
            project = await session.get(Project, task.project_id)
            project_name = project.name if project else ""

        # Create task workdir
        workdir_base = Path("/tmp/soda-task-workdirs")
        workdir_base.mkdir(parents=True, exist_ok=True)
        workdir = workdir_base / f"task-{task.id}"
        workdir.mkdir(parents=True, exist_ok=True)

        # Resolve template variables
        cmd = assignee.execute_command
        cmd = cmd.replace("{{task.id}}", str(task.id))
        cmd = cmd.replace("{{task.title}}", task.title or "")
        cmd = cmd.replace("{{task.description}}", task.description or "")
        cmd = cmd.replace("{{task.complexity}}", task.complexity or "")
        cmd = cmd.replace("{{task.comments}}", json.dumps(comments))
        cmd = cmd.replace("{{project.name}}", project_name)
        cmd = cmd.replace("{{callback.url}}", callback_url)
        cmd = cmd.replace("{{task.workdir}}", str(workdir))

        # Build environment with OpenCode API key injected
        env = os.environ.copy()
        if opencode_api_key:
            env["OPENCODE_API_KEY"] = opencode_api_key

        # Run the command in the task workdir, capture output to file
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

        # Parse pending_questions for each idea
        ideas = []
        for i in ideas_raw:
            questions = []
            if i.pending_questions:
                try:
                    questions = json.loads(i.pending_questions)
                except Exception:
                    questions = []
            from types import SimpleNamespace
            iv = SimpleNamespace()
            for attr in ["id", "title", "description", "system_prompt", "architect_user_id", "status"]:
                setattr(iv, attr, getattr(i, attr))
            iv.questions = questions
            ideas.append(iv)

        return templates.TemplateResponse(
            "ideas.html",
            {"request": request, "ideas": ideas, "ai_users": ai_users, "projects": projects},
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
            if task_ids:
                dep_result = await session.execute(
                    sa_select(TaskDependency.task_id, TaskDependency.depends_on_id)
                    .where(TaskDependency.task_id.in_(task_ids))
                )
                for task_id, dep_id in dep_result.all():
                    deps_map.setdefault(task_id, []).append(dep_id)
                
                # Determine which tasks have unmet dependencies
                # A dependency is "unmet" if the dependency task is not in "done" column
                done_ids = {t.id for t in tasks if t.board_column == "done"}
                unmet_ids = set()
                for tid, dep_ids in deps_map.items():
                    if any(d not in done_ids for d in dep_ids):
                        unmet_ids.add(tid)

            for t in tasks:
                cr = await session.execute(
                    sa_select(TaskComment).where(TaskComment.task_id == t.id).order_by(TaskComment.created_at)
                )
                comments_map[t.id] = cr.scalars().all()
                # Attach has_unmet_deps flag
                t.has_unmet_deps = t.id in unmet_ids

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
            "users.html", {"request": request, "users": users, "projects": projects}
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
            {"request": request, "settings": settings, "projects": projects},
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
                "review_user_id": project.review_user_id,
            }

    @app.patch("/api/projects/{project_id}")
    async def update_project(project_id: int, name: Optional[str] = Form(None), description: Optional[str] = Form(None), review_user_id: Optional[int] = Form(None)):
        async with async_session() as session:
            project = await session.get(Project, project_id)
            if not project:
                raise HTTPException(404)
            if name:
                project.name = name
            if description is not None:
                project.description = description
            if review_user_id is not None:
                project.review_user_id = review_user_id if review_user_id > 0 else None
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
    ):
        async with async_session() as session:
            project = await session.get(Project, project_id)
            if not project:
                raise HTTPException(404, "Project not found")
            # Get max position
            result = await session.execute(
                sa_select(Task).where(Task.project_id == project_id).order_by(Task.position.desc()).limit(1)
            )
            last = result.scalar_one_or_none()
            pos = (last.position + 1) if last else 0
            task = Task(
                project_id=project_id,
                title=title,
                description=description,
                assignee_id=assignee_id,
                complexity=complexity,
                position=pos,
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)
            return {"id": task.id, "title": task.title, "column": task.board_column}

    @app.get("/api/tasks/{task_id}")
    async def get_task(task_id: int):
        async with async_session() as session:
            task = await session.get(Task, task_id)
            if not task:
                raise HTTPException(404)
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
                # Delete task comments
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
                    pass  # Table might not exist or have different structure
                # Delete the tasks
                for task in tasks:
                    await session.delete(task)
            
            # Delete the project
            await session.delete(project)
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

            # If moving to Running and assignee is an AI user, execute command
            if new_column == "running":
                if task.assignee_id:
                    assignee = await session.get(User, task.assignee_id)
                    if assignee and assignee.type == "ai":
                        await _run_execute_command(task, assignee)

            # If moving from blocked to running, kill old process if any
            if old_column == "blocked" and new_column == "running":
                if task.id in running_processes:
                    proc = running_processes[task.id]
                    if proc.returncode is None:
                        proc.kill()
                    del running_processes[task.id]

            # If moving to backlog, kill process
            if new_column == "backlog" and task.id in running_processes:
                proc = running_processes[task.id]
                if proc.returncode is None:
                    proc.kill()
                del running_processes[task.id]

            await session.commit()
            return {"ok": True, "column": new_column}

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
        questions = []
        if i.pending_questions:
            import json as _json
            try:
                questions = _json.loads(i.pending_questions)
            except Exception:
                questions = []
        return {
            "id": i.id,
            "title": i.title,
            "description": i.description,
            "system_prompt": i.system_prompt,
            "architect_user_id": i.architect_user_id,
            "status": i.status,
            "questions": questions,
            "created_at": str(i.created_at),
        }

    @app.get("/api/ideas")
    async def list_ideas():
        async with async_session() as session:
            result = await session.execute(
                sa_select(Idea).order_by(Idea.created_at.desc())
            )
            return [_idea_to_dict(i) for i in result.scalars().all()]

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
        """Call the architect AI via OpenCode and return parsed JSON result."""
        import logging
        logger = logging.getLogger(__name__)
        
        _write_opencode_auth(architect)
        opencode_api_key = await _get_opencode_api_key()
        env = os.environ.copy()
        if opencode_api_key:
            env["OPENCODE_API_KEY"] = opencode_api_key

        logger.info(f"Calling architect {architect.name} with prompt length: {len(prompt)}")
        
        proc = await asyncio.create_subprocess_shell(
            f'opencode run {json.dumps(prompt)}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.error(f"Architect timed out after 600s")
            raise HTTPException(504, "Architect AI timed out. Try again with a shorter prompt.")
        output = stdout.decode().strip()
        error_output = stderr.decode().strip()
        
        logger.info(f"Architect stdout length: {len(output)}")
        logger.info(f"Architect stderr length: {len(error_output)}")
        if error_output:
            logger.error(f"Architect stderr: {error_output}")
        
        if proc.returncode != 0:
            logger.error(f"Architect process failed with code {proc.returncode}")
            raise HTTPException(500, f"Architect process failed (code {proc.returncode}): {error_output[:500]}")
        
        json_match = re.search(r'\{[\s\S]*\}', output)
        if not json_match:
            logger.error(f"No JSON found in architect output: {output[:500]}")
            raise HTTPException(500, f"Architect did not return valid JSON. Stderr: {error_output[:200]}, Output: {output[:200]}")
        try:
            result = json.loads(json_match.group())
            logger.info(f"Successfully parsed architect response: {result.get('type')}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse architect JSON: {str(e)}, Output: {output[:500]}")
            raise HTTPException(500, f"Failed to parse architect JSON response: {str(e)}")

    async def _create_project_from_result(
        idea: "Idea",
        result: dict,
        repo_name: str = None,
        repo_private: bool = True,
    ) -> dict:
        """Create project and tasks from architect generate response.
        Also creates a GitHub repo for the project.
        If repo creation fails, the entire project generation fails (no tasks created)."""
        # First, verify GitHub auth is configured
        async with async_session() as session:
            username = await _get_setting(session, "git_username")
            token = await _get_setting(session, "git_token")

        if not username or not token:
            raise HTTPException(400,
                "⚠️ GitHub auth is required to generate projects. "
                "Please configure git_username and git_token in Settings → GitHub. "
                "The token needs 'repo' scope to create repositories."
            )

        # Determine repo name: use provided name or sanitize from project name
        if repo_name:
            repo_name = re.sub(r'[^a-z0-9-]', '', repo_name.lower().replace(' ', '-'))[:100]
        else:
            repo_name = result.get("project_name", idea.title).lower().replace(" ", "-")
            repo_name = re.sub(r'[^a-z0-9-]', '', repo_name)[:100]

        if not repo_name:
            repo_name = f"soda-{idea.id}"

        # Create GitHub repo for the project — if this fails, everything fails
        ensure_result = await _ensure_github_repo(username, repo_name, token, private=repo_private)
        if ensure_result["status"] == "error":
            raise HTTPException(400,
                f"⚠️ Failed to create GitHub repo '{repo_name}': {ensure_result['message']}. "
                "Please check your git_token has 'repo' scope and the repo name is valid. "
                "Project generation was cancelled."
            )

        repo_url = ensure_result["data"].get("html_url", f"https://github.com/{username}/{repo_name}")

        # Create project and tasks in DB
        async with async_session() as session:
            project = Project(
                name=result.get("project_name", idea.title),
                description=result.get("project_description", idea.description),
                repo_name=repo_name,
                repo_url=repo_url,
            )
            session.add(project)
            await session.commit()
            await session.refresh(project)

            for i, t in enumerate(result.get("tasks", [])):
                task = Task(
                    project_id=project.id,
                    title=t.get("title", "Untitled"),
                    description=t.get("description", ""),
                    complexity=t.get("complexity"),
                    board_column="backlog",
                    position=i,
                )
                session.add(task)

            idea_obj = await session.get(Idea, idea.id)
            idea_obj.status = "generated"
            idea_obj.pending_questions = None
            await session.commit()

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
    ):
        """Start generating a project from an idea using the architect AI.
        repo_name: optional custom repo name (sanitized from project name if not provided)
        repo_private: 'true' or 'false' (default: true)"""
        async with async_session() as session:
            idea = await session.get(Idea, idea_id)
            if not idea:
                raise HTTPException(404, "Idea not found")
            
            # Use provided architect_user_id or fall back to idea's architect
            arch_id = architect_user_id or idea.architect_user_id
            if not arch_id:
                raise HTTPException(400, "No architect user selected")
            
            architect = await session.get(User, arch_id)
            if not architect or architect.type != "ai":
                raise HTTPException(400, "Architect must be an AI user")
            
            # Save the architect to the idea for future reference
            idea.architect_user_id = arch_id
            idea.status = "generating"
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
  "tasks": [
    {{"title": "...", "description": "...", "complexity": "S|M|L|XL"}}
  ]
}}

Return ONLY valid JSON, no other text."""

        try:
            result = await _call_architect(architect, prompt)
        except Exception:
            async with async_session() as session:
                idea_obj = await session.get(Idea, idea_id)
                idea_obj.status = "active"
                await session.commit()
            raise

        if result.get("type") == "questions":
            questions = result.get("questions", [])
            async with async_session() as session:
                idea_obj = await session.get(Idea, idea_id)
                idea_obj.status = "active"
                idea_obj.pending_questions = json.dumps(questions)
                await session.commit()
            return {"status": "questions", "questions": questions, "idea_id": idea_id}

        if result.get("type") == "generate":
            async with async_session() as session:
                idea = await session.get(Idea, idea_id)
            return await _create_project_from_result(
                idea, result,
                repo_name=repo_name,
                repo_private=repo_private == "true",
            )

        async with async_session() as session:
            idea_obj = await session.get(Idea, idea_id)
            idea_obj.status = "active"
            await session.commit()
        raise HTTPException(500, "Unexpected architect response type")

    @app.post("/api/ideas/{idea_id}/answer")
    async def answer_idea_questions(idea_id: int, answers: str = Form(...)):
        """Submit answers to architect questions and continue generation."""
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
            await session.commit()

        try:
            answers_list = json.loads(answers)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid answers format")

        sys_prompt = architect.system_prompt or ""
        if idea.system_prompt:
            sys_prompt += "\n\n" + idea.system_prompt

        qa_pairs = "\n".join([
            f"Q: {q}\nA: {a}"
            for q, a in zip(pending_questions, answers_list)
        ])

        prompt = f"""You are an Architect AI. Generate a project plan from this idea.

Title: {idea.title}
Description: {idea.description}

{sys_prompt}

You previously asked questions and received these answers:
{qa_pairs}

Now generate the project. Return ONLY this JSON:
{{
  "type": "generate",
  "project_name": "...",
  "project_description": "...",
  "tasks": [
    {{"title": "...", "description": "...", "complexity": "S|M|L|XL"}}
  ]
}}

If you still have questions, return:
{{
  "type": "questions",
  "questions": ["Question?"]
}}

Return ONLY valid JSON, no other text."""

        try:
            result = await _call_architect(architect, prompt)
        except Exception:
            async with async_session() as session:
                idea_obj = await session.get(Idea, idea_id)
                idea_obj.status = "active"
                await session.commit()
            raise

        if result.get("type") == "questions":
            questions = result.get("questions", [])
            async with async_session() as session:
                idea_obj = await session.get(Idea, idea_id)
                idea_obj.status = "active"
                idea_obj.pending_questions = json.dumps(questions)
                await session.commit()
            return {"status": "questions", "questions": questions, "idea_id": idea_id}

        if result.get("type") == "generate":
            async with async_session() as session:
                idea = await session.get(Idea, idea_id)
            return await _create_project_from_result(idea, result)

        async with async_session() as session:
            idea_obj = await session.get(Idea, idea_id)
            idea_obj.status = "active"
            await session.commit()
        raise HTTPException(500, "Unexpected architect response type")

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
                    "role": u.role,
                    "type": u.type,
                    "provider": u.provider,
                    "model": u.model,
                    "system_prompt": u.system_prompt,
                    "execute_command": u.execute_command,
                }
                for u in users
            ]

    @app.post("/api/users")
    async def create_user(
        name: str = Form(...),
        role: str = Form(""),
        type: str = Form(...),
        provider: str = Form(""),
        api_key: str = Form(""),
        model: str = Form(""),
        system_prompt: str = Form(""),
        execute_command: str = Form(""),
    ):
        async with async_session() as session:
            user = User(
                name=name,
                role=role or None,
                type=type,
                provider=provider or None,
                api_key=api_key or None,
                model=model or None,
                system_prompt=system_prompt or None,
                execute_command=execute_command or None,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)

            # If AI user with API key, update OpenCode auth
            if type == "ai" and api_key:
                _write_opencode_auth(user)

            return {"id": user.id, "name": user.name, "type": user.type}

    @app.patch("/api/users/{user_id}")
    async def update_user(
        user_id: int,
        name: Optional[str] = Form(None),
        role: Optional[str] = Form(None),
        type: Optional[str] = Form(None),
        provider: Optional[str] = Form(None),
        api_key: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
        system_prompt: Optional[str] = Form(None),
        execute_command: Optional[str] = Form(None),
    ):
        async with async_session() as session:
            user = await session.get(User, user_id)
            if not user:
                raise HTTPException(404)
            if name:
                user.name = name
            if role is not None:
                user.role = role or None
            if api_key is not None:
                user.api_key = api_key or None
            if provider is not None:
                user.provider = provider or None
            if model is not None:
                user.model = model or None
            if system_prompt is not None:
                user.system_prompt = system_prompt or None
            if execute_command is not None:
                user.execute_command = execute_command or None
            await session.commit()

            # Update OpenCode auth if this AI user has API key
            if user.type == "ai" and user.api_key:
                _write_opencode_auth(user)

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
        
        async with async_session() as session:
            for key, value in data.items():
                result = await session.execute(
                    sa_select(GlobalSetting).where(GlobalSetting.key == key)
                )
                setting = result.scalar_one_or_none()
                
                if setting:
                    setting.value = value
                else:
                    session.add(GlobalSetting(key=key, value=value))
            
            await session.commit()
        
        return {"ok": True, "updated": list(data.keys())}

    # ── API: Models ────────────────────────────────────────────────

    @app.get("/api/models")
    async def list_models():
        """List available AI models from OpenCode"""
        try:
            # Write global OpenCode API key to auth.json so the CLI can authenticate
            opencode_api_key = await _get_opencode_api_key()
            auth_dir = OPENCODE_AUTH.parent
            auth_dir.mkdir(parents=True, exist_ok=True)
            auth_data = {}
            if opencode_api_key:
                auth_data["apiKey"] = opencode_api_key
            with open(OPENCODE_AUTH, "w") as f:
                json.dump(auth_data, f)

            # Build env with API key
            env = os.environ.copy()
            if opencode_api_key:
                env["OPENCODE_API_KEY"] = opencode_api_key

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
            
            # OpenCode returns plain text, one model per line
            models = []
            for line in output.split("\n"):
                line = line.strip()
                if line:
                    models.append({"id": line})
            return models
            
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []

    # ── API: Callback ──────────────────────────────────────────────

    @app.post("/api/callback")
    async def callback(
        request: Request,
        taskId: int = Query(None),
        status: str = Query(None),
        question: Optional[str] = Query(None),
        summary: Optional[str] = Query(None),
    ):
        """Callback endpoint for the execute command to report status.
        Accepts both JSON body and query parameters (OpenCode uses query params)."""
        # If query params are present, use them; otherwise parse JSON body
        if taskId is not None and status is not None:
            payload = CallbackPayload(taskId=taskId, status=status, question=question, summary=summary)
        else:
            try:
                body = await request.json()
                payload = CallbackPayload(**body)
            except Exception:
                raise HTTPException(422, "Invalid callback payload: provide taskId and status as query params or JSON body")
        async with async_session() as session:
            task = await session.get(Task, payload.taskId)
            if not task:
                raise HTTPException(404, "Task not found")

            if payload.status == "blocked" and payload.question:
                task.board_column = "blocked"
                comment = TaskComment(
                    task_id=task.id,
                    author="AI",
                    content=payload.question,
                )
                session.add(comment)

            elif payload.status == "review":
                # Collect AI output first (before any git operations)
                ai_output = ""
                if payload.summary:
                    ai_output = f"**Summary:** {payload.summary}"
                workdir = Path(f"/tmp/soda-task-workdirs/task-{payload.taskId}")
                stdout_log = workdir / ".soda-stdout.log"
                stderr_log = workdir / ".soda-stderr.log"
                if stdout_log.exists():
                    try:
                        stdout_text = stdout_log.read_text().strip()
                        if stdout_text:
                            ai_output += f"\n\n**AI Output:**\n{stdout_text[:3000]}"
                    except Exception:
                        pass
                if stderr_log.exists():
                    try:
                        stderr_text = stderr_log.read_text().strip()
                        if stderr_text:
                            ai_output += f"\n\n**Stderr:**\n{stderr_text[:500]}"
                    except Exception:
                        pass
                # Close process fds and clean up
                if payload.taskId in running_processes:
                    proc_info = running_processes[payload.taskId]
                    if isinstance(proc_info, tuple):
                        proc, stdout_fd, stderr_fd = proc_info
                        try:
                            stdout_fd.close()
                            stderr_fd.close()
                        except Exception:
                            pass
                    try:
                        if isinstance(proc_info, tuple):
                            await asyncio.wait_for(proc_info[0].wait(), timeout=10)
                        else:
                            await asyncio.wait_for(proc_info.wait(), timeout=10)
                    except Exception:
                        pass
                # List files in task workdir
                if workdir.exists():
                    files = [f for f in workdir.iterdir() if not f.name.startswith(".soda-")]
                    if files:
                        file_list = "\n".join(f"  • {f.name}{'/' if f.is_dir() else ''}" for f in sorted(files)[:30])
                        ai_output += f"\n\n**Files created:**\n{file_list}"
                if ai_output:
                    comment = TaskComment(task_id=task.id, author="AI", content=ai_output)
                    session.add(comment)

                # Try to create GitHub PR — if it fails, move to blocked instead of review
                pr_created = False
                pr_error = None
                try:
                    project = await session.get(Project, task.project_id)
                    pr_username = await _get_setting(session, "git_username")
                    pr_token = await _get_setting(session, "git_token")
                    pr_default_branch = await _get_setting(session, "git_default_branch", "main")
                    if pr_username and pr_token:
                        # Use the project's repo name (set during project generation)
                        repo_name = project.repo_name if project and project.repo_name else f"soda-{task.project_id}"
                        pr_url = await _create_github_pr(
                            session=session, task=task, project=project, workdir=workdir,
                            username=pr_username, token=pr_token,
                            repo_name=repo_name,
                            default_branch=pr_default_branch,
                        )
                        if pr_url:
                            pr_comment = TaskComment(task_id=task.id, author="Soda",
                                content=f"📦 **Pull Request created:** {pr_url}")
                            session.add(pr_comment)
                            pr_created = True
                    else:
                        pr_error = "⚠️ GitHub auth not configured. Set git_username and git_token in Settings to auto-create PRs."
                except Exception as e:
                    pr_error = f"⚠️ Failed to create PR: {str(e)}"

                if pr_error:
                    # PR creation failed — move to blocked with error info
                    task.board_column = "blocked"
                    error_comment = TaskComment(task_id=task.id, author="Soda", content=pr_error)
                    session.add(error_comment)
                elif pr_created:
                    # PR created successfully — move to review
                    task.board_column = "review"
                else:
                    # No git auth configured — still move to review but with a note
                    task.board_column = "review"
                    no_auth_comment = TaskComment(task_id=task.id, author="Soda",
                        content="⚠️ GitHub auth not configured. Set git_username and git_token in Settings to auto-create PRs.")
                    session.add(no_auth_comment)

                # Auto-review if review user configured (only if we made it to review)
                if task.board_column == "review" and project and project.review_user_id:
                    reviewer = await session.get(User, project.review_user_id)
                    if reviewer and reviewer.type == "ai":
                        _write_opencode_auth(reviewer)
                        # Run review via OpenCode
                        review_prompt = f"""Review this task:
Title: {task.title}
Description: {task.description}

Provide a concise review. Focus on: code quality, completeness, and any issues.
Return your review as JSON:
{{"approved": true/false, "comments": "..."}}"""

                        proc = await asyncio.create_subprocess_shell(
                            f'opencode run {json.dumps(review_prompt)}',
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
                        output = stdout.decode().strip()
                        jm = re.search(r'\{[\s\S]*\}', output)
                        if jm:
                            try:
                                review_result = json.loads(jm.group())
                                review_comment = TaskComment(
                                    task_id=task.id,
                                    author=reviewer.name,
                                    content=review_result.get("comments", "Review completed."),
                                )
                                session.add(review_comment)
                            except json.JSONDecodeError:
                                pass

            # Clean up process tracker
            if payload.taskId in running_processes:
                proc_info = running_processes[payload.taskId]
                if isinstance(proc_info, tuple):
                    proc = proc_info[0]
                else:
                    proc = proc_info
                if proc.returncode is None:
                    try:
                        proc.kill()
                        await proc.wait()
                    except Exception:
                        pass
                del running_processes[payload.taskId]

            await session.commit()
            return {"ok": True, "column": task.board_column}

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

                # Create PR via GitHub API
                pr_url = f"https://api.github.com/repos/{username}/{target_repo}/pulls"
                pr_data = {
                    "title": f"Task {task.id}: {task.title}",
                    "head": feature_branch,
                    "base": target_branch,
                    "body": f"## Task {task.id}: {task.title}\n\n{task.description or 'No description'}\n\n**Complexity:** {task.complexity or 'N/A'}\n\n*Created by Soda*",
                }
                async with httpx.AsyncClient() as client:
                    pr_resp = await client.post(pr_url, headers=headers, json=pr_data)
                    if pr_resp.status_code in [200, 201]:
                        pr_json = pr_resp.json()
                        pr_html_url = pr_json.get("html_url", "")
                        logger.info(f"Created PR: {pr_html_url}")
                        return pr_html_url
                    else:
                        logger.error(f"PR creation failed: {pr_resp.text}")
                        return None
            else:
                logger.info(f"No changes to commit for task {task.id}")
                return None

        except Exception as e:
            logger.error(f"Error creating PR for task {task.id}: {e}")
            return None

    async def _ensure_github_repo(owner: str, repo_name: str, token: str, private: bool = True) -> dict:
        """Ensure the GitHub repository exists, create if it doesn't."""
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Check if repo exists
        async with httpx.AsyncClient() as client:
            check_url = f"https://api.github.com/repos/{owner}/{repo_name}"
            check_resp = await client.get(check_url, headers=headers)
            
            if check_resp.status_code == 200:
                return {"status": "exists", "data": check_resp.json()}
            
            if check_resp.status_code != 404:
                return {"status": "error", "message": f"GitHub API error: {check_resp.text}"}
            
            # Create the repository
            create_url = "https://api.github.com/user/repos"
            create_data = {
                "name": repo_name,
                "private": private,
                "auto_init": True,  # Create with README
                "description": f"Created by Soda"
            }
            create_resp = await client.post(create_url, headers=headers, json=create_data)
            
            if create_resp.status_code in [200, 201]:
                return {"status": "created", "data": create_resp.json()}
            else:
                return {"status": "error", "message": f"Failed to create repo: {create_resp.text}"}

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
        proc = running_processes.get(task_id)
        if not proc:
            return {"running": False}
        return {"running": proc.returncode is None, "exit_code": proc.returncode}

    # ── API: Health check ──────────────────────────────────────────

    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok", "service": "soda"}

    # ── Static files ───────────────────────────────────────────────

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


app = create_app()
