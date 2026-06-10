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
from fastapi.responses import HTMLResponse, JSONResponse
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

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
OPENCODE_AUTH = Path("/root/.local/share/opencode/auth.json")

# In-memory process tracker: task_id -> (asyncio.subprocess.Process, stdout_fd, stderr_fd)
running_processes: dict[int, Tuple[asyncio.subprocess.Process, Any, Any]] = {}

# Idempotency guard: track tasks currently being post-processed
_processing_tasks: set[int] = set()

# Context for post-processing after AI completes
_post_process_ctx: dict[int, dict] = {}

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
        # Start watchdog in background
        watchdog_task = asyncio.create_task(_watchdog_check())
        yield
        watchdog_task.cancel()

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
        Falls back to global API key based on ai_provider setting."""
        auth_dir = OPENCODE_AUTH.parent
        auth_dir.mkdir(parents=True, exist_ok=True)
        auth_data = {}
        if user.api_key:
            auth_data["apiKey"] = user.api_key
        if user.provider:
            auth_data["provider"] = user.provider
        else:
            # Infer provider from user model or global setting
            if user.model and user.model.startswith("openrouter/"):
                auth_data["provider"] = "openrouter"
        if user.model:
            auth_data["model"] = user.model
        # If user has no API key, don't overwrite auth.json — let the
        # global OPENCODE_API_KEY env var (set by callers) handle auth.
        if not user.api_key:
            return
        with open(OPENCODE_AUTH, "w") as f:
            json.dump(auth_data, f)

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

    async def _get_effective_api_key() -> str:
        """Get the API key for the currently configured provider."""
        provider = await get_setting("ai_provider", "opencode")
        if provider == "openrouter":
            return await _get_openrouter_api_key()
        return await _get_opencode_api_key()

    # ── Helper: run execute command ─────────────────────────────────

    async def _run_execute_command(task: Task, assignee: User) -> None:
        """Run the AI user's execute command as a subprocess.
        Clones the project repo, checks out main, provides context,
        then after AI completes: git commit/push, create PR, send callback."""
        if not assignee.execute_command:
            return

        # Write AI user's auth to OpenCode config
        _write_opencode_auth(assignee)

        # Get effective API key based on provider setting
        api_key = await _get_effective_api_key()

        # Build comments JSON and collect context
        async with async_session() as session:
            result = await session.execute(
                sa_select(TaskComment).where(TaskComment.task_id == task.id).order_by(TaskComment.created_at)
            )
            comments = [
                {"author": c.author, "content": c.content, "created_at": str(c.created_at)}
                for c in result.scalars().all()
            ]

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

            remaining_result = await session.execute(
                sa_select(Task).where(
                    Task.project_id == task.project_id,
                    Task.id != task.id,
                ).order_by(Task.id)
            )
            remaining_tasks = remaining_result.scalars().all()
            remaining_summary = "\n".join(
                f"- [{t.board_column}] {t.title}: {t.description or '(no description)'}"
                for t in remaining_tasks
            )

        # Build authenticated repo URL
        auth_repo_url = repo_url
        if repo_url and git_username and git_token:
            auth_repo_url = repo_url.replace("https://github.com/", f"https://{git_username}:{git_token}@github.com/")
            auth_repo_url = auth_repo_url.replace("http://github.com/", f"https://{git_username}:{git_token}@github.com/")

        # Create task workdir and clone repo
        workdir_base = Path("/tmp/soda-task-workdirs")
        workdir_base.mkdir(parents=True, exist_ok=True)
        workdir = workdir_base / f"task-{task.id}"
        workdir.mkdir(parents=True, exist_ok=True)

        if auth_repo_url:
            import shutil
            import subprocess as sp
            try:
                for item in workdir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                sp.run(["git", "clone", auth_repo_url, str(workdir)], check=True, capture_output=True, timeout=60)
                sp.run(["git", "checkout", "main"], cwd=str(workdir), check=True, capture_output=True, timeout=30)
                sp.run(["git", "pull", "origin", "main"], cwd=str(workdir), check=True, capture_output=True, timeout=30)
            except Exception:
                pass

        # Build the full prompt with context
        full_prompt = f"""You are working on a software project.

## Project: {project_name}

## Your Task (ONLY work on this):
**Title:** {task.title}
**Description:** {task.description or '(no description)'}
**Complexity:** {task.complexity or 'not specified'}

## Other tasks in this project (DO NOT work on these):
{remaining_summary if remaining_summary else '(none)'}

## Existing comments on this task:
{json.dumps(comments, indent=2) if comments else '(none)'}

## Work Instructions:
- ONLY implement what is described in "Your Task" above
- Do NOT work on any of the other tasks listed above
- Work ONLY in the current directory: {workdir}
- Create/edit files directly in this directory
- Do NOT call any callback URL — the system handles that automatically
- If you cannot complete the task, describe what is blocking you as the last line of your output
"""

        # Write prompt to file (prevents shell quoting issues with special chars)
        prompt_file = workdir / ".soda-prompt.txt"
        prompt_file.write_text(full_prompt)

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
        # Replace {{task.prompt}} with file contents via shell substitution
        # Removes surrounding single quotes too, so shell quoting doesn't break
        cmd = cmd.replace("'{{task.prompt}}'", f'"$(cat {prompt_file})"')
        # Fallback: if no quotes were used, replace with file path
        cmd = cmd.replace("{{task.prompt}}", str(prompt_file))

        # Save the full prompt as a comment
        try:
            async with async_session() as prompt_session:
                prompt_session.add(TaskComment(
                    task_id=task.id,
                    author="Soda",
                    content=f"📋 **Prompt sent to AI:**\n\n{full_prompt}",
                ))
                await prompt_session.commit()
        except Exception:
            pass

        # Build env
        env = os.environ.copy()
        if api_key:
            env["OPENCODE_API_KEY"] = api_key
            env["OPENROUTER_API_KEY"] = api_key
            # Ensure auth.json has the key if user has no individual key
            auth_dir = OPENCODE_AUTH.parent
            auth_dir.mkdir(parents=True, exist_ok=True)
            if not assignee.api_key:
                auth_data = {}
                if assignee.model:
                    auth_data["model"] = assignee.model
                    if assignee.model.startswith("openrouter/"):
                        auth_data["provider"] = "openrouter"
                auth_data["apiKey"] = api_key
                with open(OPENCODE_AUTH, "w") as f:
                    json.dump(auth_data, f)

        # Run OpenCode
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

        # Store context for post-processing
        _post_process_ctx[task.id] = {
            "callback_url": callback_url,
            "workdir": str(workdir),
            "auth_repo_url": auth_repo_url,
            "repo_name": repo_name,
            "git_username": git_username,
            "git_token": git_token,
            "default_branch": default_branch,
            "project_id": task.project_id,
        }


    # Context for post-processing after AI completes
    _post_process_ctx: dict[int, dict] = {}

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
                logger.warning(f"Task {task_id}: No context found for post-processing")
                return

            workdir = Path(ctx["workdir"])
            auth_repo_url = ctx["auth_repo_url"]
            repo_name = ctx["repo_name"]
            git_username = ctx["git_username"]
            git_token = ctx["git_token"]
            default_branch = ctx["default_branch"]

            # Check for AI blocking message in stdout
            # The prompt tells AI: "If you cannot complete the task, describe what is blocking you as the last line of your output"
            blocked_reason = ""
            stdout_file = workdir / ".soda-stdout.log"
            if stdout_file.exists():
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

            if not stdout_text:
                # No AI output at all — indicates execution error
                error_detail = stderr_text[:2000] if stderr_text else "Unknown error (no output)"
                async with async_session() as session:
                    task = await session.get(Task, task_id)
                    if task:
                        task.board_column = "blocked"
                        session.add(TaskComment(task_id=task_id, author="Soda",
                            content=f"⚠️ **Execution error:** OpenCode did not produce any output.\n\n```\n{error_detail}\n```"))
                        await session.commit()
                return

            # Git commit + push + PR
            pr_url = await _git_commit_push_and_pr(
                task_id=task_id,
                workdir=workdir,
                auth_repo_url=auth_repo_url,
                repo_name=repo_name,
                username=git_username,
                token=git_token,
                default_branch=default_branch,
            )

            # Update task status based on PR result
            async with async_session() as session:
                task = await session.get(Task, task_id)
                if not task:
                    return
                if pr_url:
                    task.board_column = "review"
                    session.add(TaskComment(task_id=task_id, author="Soda",
                        content=f"📦 **Pull Request created:** {pr_url}"))
                elif git_username and git_token:
                    task.board_column = "blocked"
                    session.add(TaskComment(task_id=task_id, author="Soda",
                        content="⚠️ Failed to create PR"))
                else:
                    task.board_column = "blocked"
                    session.add(TaskComment(task_id=task_id, author="Soda",
                        content="⚠️ GitHub auth not configured. Set git_username and git_token in Settings to auto-create PRs."))
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
    ) -> Optional[str]:
        """Commit task workdir changes, push feature branch, create PR. Returns PR URL or None."""
        if not username or not token or not auth_repo_url:
            return None
        
        try:
            feature_branch = f"task-{task_id}"
            repo_workdir = Path(f"/tmp/soda-pr-workdirs/task-{task_id}")
            repo_workdir.parent.mkdir(parents=True, exist_ok=True)
            
            # Remove old and clone
            import shutil
            if repo_workdir.exists():
                shutil.rmtree(repo_workdir)
            
            repo = git.Repo.clone_from(auth_repo_url, repo_workdir)
            
            # Checkout default branch
            try:
                repo.git.checkout(default_branch)
            except Exception:
                repo.git.checkout("-b", default_branch)
            
            # Create feature branch
            repo.git.checkout("-b", feature_branch)
            
            # Copy workdir contents (exclude .soda- files)
            for item in workdir.iterdir():
                if item.name.startswith(".soda-"):
                    continue
                dest = repo_workdir / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
            
            # Commit + push
            repo.git.add(A=True)
            if repo.is_dirty() or repo.untracked_files:
                repo.index.commit(f"feat: task {task_id}")
                repo.git.push("origin", feature_branch)
                
                # Create PR using GitHubService
                gh_service = GitHubService(username, token)
                pr_result = await gh_service.create_pull_request(
                    repo_name=repo_name,
                    title=f"Task {task_id}",
                    head=feature_branch,
                    base=default_branch,
                    body=f"Task {task_id} — created by Soda"
                )
                
                if pr_result["success"]:
                    logger.info(f"PR created: {pr_result['pr_url']}")
                    return pr_result["pr_url"]
                else:
                    logger.error(f"PR failed: {pr_result['error']}")
                    return None
            
            return None
        except Exception as e:
            logger.error(f"git/pr error for task {task_id}: {e}")
            return None

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
                # Attach is_running flag for animation
                t.is_running = t.board_column == "running"

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
                # Find user assigned to this size
                assigned_user_id = await find_user_by_size(session, complexity)
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
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)
            return {"id": task.id, "title": task.title, "column": task.board_column, "assignee_id": task.assignee_id}

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

            # If moving to Running and assignee is an AI user, execute command
            if new_column == "running":
                if task.assignee_id:
                    assignee = await session.get(User, task.assignee_id)
                    if assignee and assignee.type == "ai":
                        await _run_execute_command(task, assignee)

            # If moving from blocked to running, kill old process if any
            if old_column == "blocked" and new_column == "running":
                if task.id in running_processes:
                    proc, stdout_fd, stderr_fd = running_processes[task.id]
                    if proc.returncode is None:
                        proc.kill()
                    del running_processes[task.id]

            # If moving to backlog, kill process
            if new_column == "backlog" and task.id in running_processes:
                proc, stdout_fd, stderr_fd = running_processes[task.id]
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
        api_key = await _get_effective_api_key()
        env = os.environ.copy()
        if api_key:
            env["OPENCODE_API_KEY"] = api_key
            env["OPENROUTER_API_KEY"] = api_key
            # Ensure auth.json always has the key, even if user has no individual key
            auth_dir = OPENCODE_AUTH.parent
            auth_dir.mkdir(parents=True, exist_ok=True)
            if not architect.api_key:
                # User has no individual key — write global key to auth.json
                auth_data = {}
                if architect.model:
                    auth_data["model"] = architect.model
                    if architect.model.startswith("openrouter/"):
                        auth_data["provider"] = "openrouter"
                auth_data["apiKey"] = api_key
                with open(OPENCODE_AUTH, "w") as f:
                    json.dump(auth_data, f)

        logger.info(f"Calling architect {architect.name} with prompt length: {len(prompt)}")
        
        proc = await asyncio.create_subprocess_shell(
            f'opencode run --pure {json.dumps(prompt)}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.error(f"Architect timed out after 60s")
            raise HTTPException(504, "Architect AI timed out after 60s. The prompt may be too complex or the AI service is slow. Try again or use a shorter description.")
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
        username = await get_setting("git_username")
        token = await get_setting("git_token")

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

            # Resolve assignee role to user ID
            def _resolve_assignee_id(assignee_role: str) -> Optional[int]:
                """Map assignee_role (junior/medior/senior) to an existing AI user ID."""
                role_map = {
                    "junior": "Junior Developer",
                    "medior": "Medior Developer",
                    "medior": "Medior Developer",
                    "senior": "Senior Developer",
                }
                target_name = role_map.get(assignee_role.lower() if assignee_role else "")
                if not target_name:
                    return None
                # We'll look this up in the caller; for now return the name
                return target_name

            # Pre-resolve assignee names to user IDs
            assignee_name_to_id: dict[str, int] = {}
            for t in result.get("tasks", []):
                role = t.get("assignee_role", "")
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
                assignee_id = None
                role = t.get("assignee_role", "")
                resolved_name = _resolve_assignee_id(role)
                if resolved_name:
                    assignee_id = assignee_name_to_id.get(resolved_name)

                task = Task(
                    project_id=project.id,
                    title=t.get("title", "Untitled"),
                    description=t.get("description", ""),
                    complexity=t.get("complexity"),
                    board_column="backlog",
                    position=i,
                    assignee_id=assignee_id,
                )
                session.add(task)
                await session.flush()  # Get the task ID
                task_db_ids.append(task.id)

            # Create dependencies based on depends_on indices from architect
            for i, t in enumerate(result.get("tasks", [])):
                depends_on_indices = t.get("depends_on", [])
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
        Returns immediately with status=generating. Frontend should poll for status.
        repo_name: optional custom repo name (sanitized from project name if not provided)
        repo_private: 'true' or 'false' (default: true)"""
        async with async_session() as session:
            idea = await session.get(Idea, idea_id)
            if not idea:
                raise HTTPException(404, "Idea not found")
            
            # Use provided architect_user_id, fall back to idea's architect, then Task Master
            arch_id = architect_user_id or idea.architect_user_id
            if not arch_id:
                task_master = await session.execute(
                    sa_select(User).where(User.name == "Task Master")
                )
                tm = task_master.scalar_one_or_none()
                if tm:
                    arch_id = tm.id
                    idea.architect_user_id = tm.id
            if not arch_id:
                raise HTTPException(400, "No architect user available. Please create a Task Master user first.")
            
            architect = await session.get(User, arch_id)
            if not architect or architect.type != "ai":
                raise HTTPException(400, "Architect must be an AI user")
            
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
    {{"title": "...", "description": "...", "complexity": "S|M|L|XL", "depends_on": []}}
  ]
}}

IMPORTANT: Each task can have a "depends_on" field with indices of previous tasks it depends on.
- tasks[0] should always have "depends_on": [] (no dependencies, can start immediately)
- Subsequent tasks should depend on earlier tasks that must be completed first
- Use task indices (0-based) for dependencies, e.g., "depends_on": [0, 1]
- Create a logical dependency chain: setup → core → features → tests → deploy
- A task can depend on multiple previous tasks if needed

Return ONLY valid JSON, no other text."""

        # Start background task for architect call + project creation
        asyncio.create_task(_generate_project_background(
            idea_id=idea_id,
            architect=architect,
            prompt=prompt,
            repo_name=repo_name,
            repo_private=repo_private == "true",
        ))

        return {"status": "generating", "idea_id": idea_id}

    async def _generate_project_background(
        idea_id: int,
        architect: User,
        prompt: str,
        repo_name: Optional[str] = None,
        repo_private: bool = True,
    ):
        """Background task: call architect, create project, handle errors."""
        try:
            result = await _call_architect(architect, prompt)
        except Exception as e:
            logger.error(f"Architect call failed for idea {idea_id}: {type(e).__name__}: {e}")
            async with async_session() as session:
                idea_obj = await session.get(Idea, idea_id)
                if idea_obj:
                    idea_obj.status = "error"
                    idea_obj.pending_questions = json.dumps({"error": str(e)})
                    await session.commit()
            return

        if result.get("type") == "questions":
            questions = result.get("questions", [])
            async with async_session() as session:
                idea_obj = await session.get(Idea, idea_id)
                if idea_obj:
                    idea_obj.status = "active"
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
                )
            except HTTPException as e:
                logger.error(f"Project creation failed for idea {idea_id}: {e}")
                async with async_session() as session:
                    idea_obj = await session.get(Idea, idea_id)
                    if idea_obj:
                        idea_obj.status = "error"
                        idea_obj.pending_questions = json.dumps({"error": str(e.detail)})
                        await session.commit()
            return

        # Unexpected response type
        async with async_session() as session:
            idea_obj = await session.get(Idea, idea_id)
            if idea_obj:
                idea_obj.status = "error"
                idea_obj.pending_questions = json.dumps({"error": "Unexpected architect response type"})
                await session.commit()

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
    {{"title": "...", "description": "...", "complexity": "S|M|L|XL", "depends_on": []}}
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

Return ONLY valid JSON, no other text."""

        # Start background task
        asyncio.create_task(_generate_project_background(
            idea_id=idea_id,
            architect=architect,
            prompt=prompt,
            repo_name=None,
            repo_private=True,
        ))

        return {"status": "generating", "idea_id": idea_id}

    # ── API: Users ─────────────────────────────────────────────────

    @app.get("/api/users")
    async def list_users():
        async with async_session() as session:
            result = await session.execute(sa_select(User).order_by(User.name))
            users = result.scalars().all()
            # Get all default sizes in one query
            sizes_result = await session.execute(sa_select(UserDefaultSize.user_id, UserDefaultSize.size))
            sizes_map: dict[int, list[str]] = {}
            for uid, size in sizes_result.all():
                sizes_map.setdefault(uid, []).append(size)
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
                    "default_sizes": sizes_map.get(u.id, []),
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

    @app.put("/api/users/{user_id}/default-sizes")
    async def update_user_sizes(user_id: int, payload: dict):
        """
        Set default polo sizes for a user.
        Body: {"sizes": ["XS", "S"]}
        Each size can only be assigned to one user.
        """
        sizes = payload.get("sizes", [])
        if not isinstance(sizes, list):
            raise HTTPException(400, "sizes must be a list")
        
        async with async_session() as session:
            user = await get_or_404(session, User, user_id)
            try:
                await set_user_default_sizes(session, user_id, sizes)
                await session.commit()
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error setting default sizes for user {user_id}: {e}")
                raise HTTPException(500, f"Failed to set default sizes: {str(e)}")
            
            return {"user_id": user_id, "default_sizes": sizes}

    @app.get("/api/sizes")
    async def list_sizes():
        """List all polo sizes and their assigned users."""
        async with async_session() as session:
            result = await session.execute(
                sa_select(UserDefaultSize.size, UserDefaultSize.user_id, User.name)
                .join(User, User.id == UserDefaultSize.user_id)
                .order_by(UserDefaultSize.size)
            )
            assignments = {}
            for size, uid, uname in result.all():
                assignments[size] = {"user_id": uid, "user_name": uname}
            
            return {
                "valid_sizes": VALID_SIZES,
                "assignments": assignments,
            }

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
        """List available AI models from configured provider (OpenCode or OpenRouter)"""
        try:
            # Get provider setting
            provider = await get_setting("ai_provider", "opencode")
            opencode_api_key = await get_setting("opencode_api_key", "")
            openrouter_api_key = await get_setting("openrouter_api_key", "")

            if provider == "openrouter":
                return await _fetch_openrouter_models(openrouter_api_key)
            else:
                return await _fetch_opencode_models(opencode_api_key)
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return []

    async def _fetch_openrouter_models(api_key: str) -> list[dict]:
        """Fetch models from OpenRouter API."""
        if not api_key:
            return []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if resp.status_code != 200:
                    logger.error(f"OpenRouter API error: {resp.status_code}")
                    return []
                data = resp.json()
                models = []
                for m in data.get("data", []):
                    models.append({"id": m["id"], "name": m.get("name", m["id"])})
                return models
        except Exception as e:
            logger.error(f"OpenRouter fetch error: {e}")
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

            # Close process file descriptors
            if payload.taskId in running_processes:
                proc_info = running_processes[payload.taskId]
                if isinstance(proc_info, tuple):
                    proc, stdout_fd, stderr_fd = proc_info
                    try:
                        stdout_fd.close()
                        stderr_fd.close()
                    except Exception:
                        pass
                    
                    # Check if process has exited
                    if proc.returncode is not None:
                        # Process exited - remove from tracker and trigger post-processing
                        del running_processes[payload.taskId]
                        logger.info(f"Task {payload.taskId}: Process exited with code {proc.returncode}, triggering post-processing")
                        # Trigger post-processing (git/PR/auto-review)
                        asyncio.create_task(_post_process_task(payload.taskId))
                    else:
                        # Process still running - watchdog will handle it
                        logger.info(f"Task {payload.taskId}: Process still running, watchdog will trigger post-processing")
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
            return {"status": "exists", "data": {"name": repo_name}}
        
        # Create the repository
        result = await gh_service.create_repository(repo_name, private=private, auto_init=True)
        
        if result["success"]:
            # Commit .gitignore to the newly created repo
            await gh_service.commit_gitignore(repo_name)
            return {"status": "created", "data": result["repo_data"]}
        else:
            return {"status": "error", "message": result["error"]}

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
        """Periodically check running tasks for stuck processes."""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            try:
                async with async_session() as session:
                    # Find all tasks in running column
                    result = await session.execute(
                        sa_select(Task).where(Task.board_column == "running")
                    )
                    running_tasks = result.scalars().all()
                    
                    for task in running_tasks:
                        proc_info = running_processes.get(task.id)
                        
                        # Case 1: Task is running but no process tracked → stuck
                        if not proc_info:
                            _watchdog_logger.warning(f"Task {task.id} has no running process, moving to blocked")
                            task.board_column = "blocked"
                            comment = TaskComment(
                                task_id=task.id,
                                author="Soda",
                                content="⚠️ Watchdog: No running process found for this task. It may have crashed or failed to start. Task moved to blocked."
                            )
                            session.add(comment)
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

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


app = create_app()
