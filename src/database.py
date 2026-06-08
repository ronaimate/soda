import os
from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Boolean, CheckConstraint, Column, Date, DateTime, ForeignKey,
    Integer, String, Text, func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import select as sa_select


_db_url = os.getenv("DATABASE_URL", "").strip()
DATABASE_URL = _db_url if _db_url else "postgresql+asyncpg://soda:soda@localhost:5432/soda"

engine = create_async_engine(DATABASE_URL, echo=False, pool_size=5, max_overflow=10)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[Optional[str]] = mapped_column(String(255))
    type: Mapped[str] = mapped_column(String(10), nullable=False)  # 'human' or 'ai'
    provider: Mapped[Optional[str]] = mapped_column(String(100))
    api_key: Mapped[Optional[str]] = mapped_column(Text)
    model: Mapped[Optional[str]] = mapped_column(String(255))
    system_prompt: Mapped[Optional[str]] = mapped_column(Text)
    execute_command: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("type IN ('human', 'ai')", name="user_type_check"),
    )


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    repo_name: Mapped[Optional[str]] = mapped_column(String(255))
    repo_url: Mapped[Optional[str]] = mapped_column(String(500))
    review_user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Idea(Base):
    __tablename__ = "ideas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    system_prompt: Mapped[Optional[str]] = mapped_column(Text)
    architect_user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    status: Mapped[str] = mapped_column(
        String(20), default="active"
    )  # active, generating, generated, archived
    pending_questions: Mapped[Optional[str]] = mapped_column(Text)  # JSON array of pending questions
    created_by: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("status IN ('active', 'generating', 'generated', 'archived')", name="idea_status_check"),
    )


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    board_column: Mapped[str] = mapped_column(
        String(20), default="backlog"
    )
    assignee_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    complexity: Mapped[Optional[str]] = mapped_column(String(3))
    position: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
                    "board_column IN ('backlog', 'running', 'blocked', 'review', 'done')",
                    name="task_column_check",
                ),
        CheckConstraint(
            "complexity IN ('XS', 'S', 'M', 'L', 'XL') OR complexity IS NULL",
            name="task_complexity_check",
        ),
    )


class TaskDependency(Base):
    __tablename__ = "task_dependencies"

    task_id: Mapped[int] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), primary_key=True
    )
    depends_on_id: Mapped[int] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), primary_key=True
    )


class TaskComment(Base):
    __tablename__ = "task_comments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    author: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class GlobalSetting(Base):
    __tablename__ = "global_settings"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    value: Mapped[Optional[str]] = mapped_column(Text)


class TaskGitState(Base):
    __tablename__ = "task_git_state"

    task_id: Mapped[int] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), primary_key=True
    )
    repo: Mapped[Optional[str]] = mapped_column(String(255))
    branch: Mapped[Optional[str]] = mapped_column(String(255))
    workdir: Mapped[Optional[str]] = mapped_column(Text)
    last_commit: Mapped[Optional[str]] = mapped_column(String(40))
    last_pushed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


async def init_db():
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Auto-migrate: add missing columns to existing tables
    async with engine.begin() as conn:
        # Check if ideas table exists and add pending_questions if missing
        result = await conn.exec_driver_sql("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'ideas' AND column_name = 'pending_questions'
        """)
        if result.first() is None:
            await conn.exec_driver_sql("ALTER TABLE ideas ADD COLUMN pending_questions TEXT")
        
        # Check if ideas table exists and add created_by if missing
        result = await conn.exec_driver_sql("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'ideas' AND column_name = 'created_by'
        """)
        if result.first() is None:
            await conn.exec_driver_sql("ALTER TABLE ideas ADD COLUMN created_by VARCHAR(255)")
        
        # Check if projects table has repo_name column
        result = await conn.exec_driver_sql("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'projects' AND column_name = 'repo_name'
        """)
        if result.first() is None:
            await conn.exec_driver_sql("ALTER TABLE projects ADD COLUMN repo_name VARCHAR(255)")
            await conn.exec_driver_sql("ALTER TABLE projects ADD COLUMN repo_url VARCHAR(500)")
        
        # Fix ForeignKey constraints for CASCADE delete
        # Drop and recreate tasks.project_id FK with CASCADE
        try:
            await conn.exec_driver_sql("""
                DO $$
                BEGIN
                    ALTER TABLE tasks DROP CONSTRAINT IF EXISTS tasks_project_id_fkey;
                    ALTER TABLE tasks ADD CONSTRAINT tasks_project_id_fkey 
                        FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE;
                EXCEPTION WHEN duplicate_object THEN
                    NULL;
                END $$;
            """)
        except Exception:
            pass
        
        # Drop and recreate task_comments.task_id FK with CASCADE
        try:
            await conn.exec_driver_sql("""
                DO $$
                BEGIN
                    ALTER TABLE task_comments DROP CONSTRAINT IF EXISTS task_comments_task_id_fkey;
                    ALTER TABLE task_comments ADD CONSTRAINT task_comments_task_id_fkey 
                        FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE;
                EXCEPTION WHEN duplicate_object THEN
                    NULL;
                END $$;
            """)
        except Exception:
            pass
        
        # Drop and recreate task_git_state.task_id FK with CASCADE
        try:
            await conn.exec_driver_sql("""
                DO $$
                BEGIN
                    ALTER TABLE task_git_state DROP CONSTRAINT IF EXISTS task_git_state_task_id_fkey;
                    ALTER TABLE task_git_state ADD CONSTRAINT task_git_state_task_id_fkey 
                        FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE;
                EXCEPTION WHEN duplicate_object THEN
                    NULL;
                END $$;
            """)
        except Exception:
            pass
        
        # Drop and recreate task_dependencies FKs with CASCADE
        try:
            await conn.exec_driver_sql("""
                DO $$
                BEGIN
                    ALTER TABLE task_dependencies DROP CONSTRAINT IF EXISTS task_dependencies_task_id_fkey;
                    ALTER TABLE task_dependencies ADD CONSTRAINT task_dependencies_task_id_fkey 
                        FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE;
                    ALTER TABLE task_dependencies DROP CONSTRAINT IF EXISTS task_dependencies_depends_on_id_fkey;
                    ALTER TABLE task_dependencies ADD CONSTRAINT task_dependencies_depends_on_id_fkey 
                        FOREIGN KEY (depends_on_id) REFERENCES tasks(id) ON DELETE CASCADE;
                EXCEPTION WHEN duplicate_object THEN
                    NULL;
                END $$;
            """)
        except Exception:
            pass
    
    # Initialize default settings
    async with async_session() as session:
        for key, default in [
            ("callback_url", "http://localhost:8000/api/callback"),
            ("opencode_api_key", ""),
        ]:
            existing = await session.execute(
                sa_select(GlobalSetting).where(GlobalSetting.key == key)
            )
            if not existing.scalar_one_or_none():
                session.add(GlobalSetting(key=key, value=default))
        await session.commit()

    # Seed default users if none exist
    async with async_session() as session:
        result = await session.execute(sa_select(User))
        if not result.scalars().all():
            _default_model = "anthropic/claude-sonnet-4"

            # Execute commands: run OpenCode, then auto-send callback
            _callback_tpl = "{{callback.url}}?taskId={{task.id}}"
            _junior_exec = (
                "opencode run '{{task.prompt}}' && "
                "curl -s -X POST '" + _callback_tpl + "&status=review' "
                "|| true"
            )
            _medior_exec = _junior_exec
            _senior_exec = _junior_exec
            _taskmaster_exec = (
                "opencode run 'You are the Task Master. Analyze the following project idea and break it down into actionable tasks.\n\n"
                "Project: {{project.name}}\n"
                "Description: {{project.description}}\n\n"
                "Create a structured task list with dependencies. Output as JSON with project_name, project_description, and tasks array. "
                "Each task should have: title, description, complexity (low/medium/high), assignee_role (junior/medior/senior), "
                "and depends_on (array of task indices, empty if none).\n\n"
                "IMPORTANT: Output ONLY valid JSON, no markdown formatting or code blocks.'"
            )

            seed_users = [
                User(
                    name="Project Owner",
                    role="Project Owner",
                    type="human",
                ),
                User(
                    name="Task Master",
                    role="Task Master",
                    type="ai",
                    model=_default_model,
                    system_prompt=(
                        "You are the Task Master. Your job is to analyze project ideas and break them down into well-defined, actionable tasks.\n\n"
                        "When given a project idea, you must:\n"
                        "1. Analyze the idea thoroughly\n"
                        "2. Create a list of specific, actionable tasks\n"
                        "3. For each task, provide: title, description, complexity (low/medium/high), and assignee role (junior/medior/senior)\n"
                        "4. Define task dependencies (which tasks must be completed before others)\n"
                        "5. Output everything as a structured JSON response\n\n"
                        "Always think about:\n"
                        "- What needs to be built first (foundation/infrastructure)\n"
                        "- What can be parallelized\n"
                        "- What depends on what\n"
                        "- Appropriate complexity for each task\n\n"
                        'Output format:\n'
                        '{\n'
                        '  "project_name": "...",\n'
                        '  "project_description": "...",\n'
                        '  "tasks": [\n'
                        '    {\n'
                        '      "title": "...",\n'
                        '      "description": "...",\n'
                        '      "complexity": "low|medium|high",\n'
                        '      "assignee_role": "junior|medior|senior",\n'
                        '      "depends_on": [task_index, ...]\n'
                        '    }\n'
                        '  ]\n'
                        '}'
                    ),
                    execute_command=_taskmaster_exec,
                ),
                User(
                    name="Junior Developer",
                    role="Junior Developer",
                    type="ai",
                    model=_default_model,
                    system_prompt=(
                        "You are a Junior Developer. You handle straightforward, well-defined tasks with clear requirements.\n\n"
                        "Your characteristics:\n"
                        "- You follow instructions precisely\n"
                        "- You ask questions when requirements are unclear\n"
                        "- You write clean, simple code\n"
                        "- You focus on one thing at a time\n"
                        "- You ask for help when stuck\n\n"
                        "Rules:\n"
                        "- ONLY work on the specific task assigned to you\n"
                        "- Do NOT work on other tasks in the project\n"
                        "- If you need clarification, ask via the callback URL with status=blocked\n"
                        "- When finished, report via the callback URL with status=review\n"
                        "- Keep your changes focused and minimal"
                    ),
                    execute_command=_junior_exec,
                ),
                User(
                    name="Medior Developer",
                    role="Medior Developer",
                    type="ai",
                    model=_default_model,
                    system_prompt=(
                        "You are a Medior Developer. You handle moderately complex tasks that require some architectural thinking and experience.\n\n"
                        "Your characteristics:\n"
                        "- You understand common patterns and best practices\n"
                        "- You can work independently on well-scoped tasks\n"
                        "- You write maintainable, tested code\n"
                        "- You consider edge cases\n"
                        "- You can debug and troubleshoot issues\n\n"
                        "Rules:\n"
                        "- ONLY work on the specific task assigned to you\n"
                        "- Do NOT work on other tasks in the project\n"
                        "- If you need clarification, ask via the callback URL with status=blocked\n"
                        "- When finished, report via the callback URL with status=review\n"
                        "- Write tests for your code when appropriate\n"
                        "- Keep changes focused on the assigned task"
                    ),
                    execute_command=_medior_exec,
                ),
                User(
                    name="Senior Developer",
                    role="Senior Developer",
                    type="ai",
                    model=_default_model,
                    system_prompt=(
                        "You are a Senior Developer. You handle complex tasks that require deep architectural knowledge, system design, and experience.\n\n"
                        "Your characteristics:\n"
                        "- You think about the big picture while implementing details\n"
                        "- You design scalable, maintainable solutions\n"
                        "- You consider performance, security, and edge cases\n"
                        "- You write comprehensive tests\n"
                        "- You can refactor and improve existing code\n"
                        "- You mentor through code quality\n\n"
                        "Rules:\n"
                        "- ONLY work on the specific task assigned to you\n"
                        "- Do NOT work on other tasks in the project\n"
                        "- If you need clarification, ask via the callback URL with status=blocked\n"
                        "- When finished, report via the callback URL with status=review\n"
                        "- Ensure your changes integrate well with the existing codebase\n"
                        "- Write tests and documentation as needed"
                    ),
                    execute_command=_senior_exec,
                ),
            ]

            for u in seed_users:
                session.add(u)
            await session.commit()
