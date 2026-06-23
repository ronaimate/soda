<div align="center">

# 🥤 Soda

### Type an idea. Get a finished project.

**Soda is an AI-native build studio.** Describe what you want in plain language, hit go, and watch a team of AI agents plan it, build it, review it, and ship it to GitHub — all visualized on a live Kanban board you never have to touch.

Vibe coding, but it actually finishes the work.

<br/>

![Soda in action — from an idea to a fully shipped project](screenshots/demo.gif)

</div>

---

## ✨ Why Soda?

You have an idea. You don't want to scaffold a repo, break it into tickets, write the boilerplate, open PRs, review diffs, and merge to main.

**Soda does all of that for you.** It turns a one-sentence idea into a real, committed, reviewed codebase on GitHub — and shows you every step happening on a board so you always know what's going on.

- 💬 **Idea in, project out** — Write what you want like you'd describe it to a teammate. No tickets, no setup.
- 🧠 **An AI team, not a single prompt** — An **Architect** plans and reviews; a **Coder** implements. They hand off work to each other automatically.
- 📋 **A board that builds itself** — Tasks flow Backlog → Running → Checking → Done on their own. You watch, you don't manage.
- 🚀 **Real code, real GitHub** — Every task becomes commits and merges on a real repository. Git stays completely hidden unless you want it.
- 🪄 **Full auto or step-by-step** — Let it run end to end, or take it one task at a time.
- 🐛 **Keep iterating** — Drop in a new idea or a bug in plain text and Soda spins up exactly the tasks needed.

---

## 🎬 How it works

### 1. Start with an idea

No forms, no Jira. Just say what you want to build. The Architect reads it, asks clarifying questions when it needs to, and turns it into a real project plan.

![Capture an idea in plain language](screenshots/ideas.png)

### 2. Watch the board build your project

Soda breaks the idea into dependency-aware tasks and runs them for you. The **Coder** writes the code, the **Architect** verifies each step in the **Checking** column, and finished work lands in **Done** — automatically, in order.

> The board is a *progress view*, not a chore. In full-auto mode you don't drag anything — you just watch it fill up.

![The pipeline runs itself across the board](screenshots/board-overview.png)

### 3. Ship to GitHub — without touching git

Under the hood, every task is committed to its own branch, reviewed, and merged into `main` on a real GitHub repository. You get a clean commit history and a working codebase. The plumbing stays invisible.

![Real commits and a real repo, generated end to end](screenshots/github-result.png)

### 4. Done — but never stuck

When the plan is complete, Soda gives you a celebration screen **and** keeps the full board around so you can review everything that was built. Want more? Add another **Idea** or report a **Bug** in plain text and the agents get back to work.

![All done — with the full board still available to review](screenshots/all-done.png)

---

## 🧑‍🚀 Meet your AI team

Soda ships with two agents out of the box, each with its own role, model, and responsibilities. They're fully configurable.

| Agent | Role |
|-------|------|
| 🏛️ **Architect** | Plans the project, breaks ideas into tasks, and reviews every change before it merges. |
| 👷 **Coder** | Picks up tasks and writes the actual code, sized by complexity (XS → XL). |

![Your AI team — Architect and Coder](screenshots/users.png)

Tune each agent's provider, model, and the task sizes it handles:

![Configure any model and the work each agent takes on](screenshots/edit-user.png)

---

## 🔌 Bring your own models

Soda is provider-agnostic. Plug in the AI backend you already use, mix and match for planning vs. coding, and swap models anytime.

- **OpenCode**
- **OpenRouter**
- **Minimax**

Set everything up once in a guided wizard, then tweak keys and models whenever you like.

![One place for providers, models, and GitHub](screenshots/settings-providers.png)

---

## 🚀 Quick Start

### Prerequisites

- Docker / Podman
- PostgreSQL 16 (bundled in the compose file below)
- A GitHub token with `repo` scope
- An API key for at least one AI provider (OpenCode, OpenRouter, or Minimax)

### Run it

```bash
git clone https://github.com/ronaimate-agent/soda.git
cd soda
docker compose up -d
```

Then open **http://localhost:8000** and follow the setup wizard:

1. Enable your AI provider(s) and paste your API key.
2. Pick a planning model (Architect) and a coding model (Coder).
3. Add your GitHub token — your username is detected automatically.
4. Head to **Ideas**, describe your first project, and hit generate.

That's it. Watch the board do the rest.

### Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string, e.g. `postgresql+asyncpg://soda:soda@127.0.0.1:5432/soda` |

Everything else — API keys, models, GitHub token — is managed from the in-app **Settings** screen.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI (Python 3.12+) |
| Database | PostgreSQL 16 · SQLAlchemy 2.0 · asyncpg |
| Frontend | Jinja2 + vanilla JS, modern dark UI |
| Agents | Pluggable: OpenCode · OpenRouter · Minimax |
| Delivery | Real GitHub repos via branches, reviews & auto-merge |
| Packaging | Single Docker image |

---

## 💡 The idea behind Soda

Most AI coding tools give you a chat box and a wall of diffs. Soda gives you a **workflow**: an idea becomes a plan, a plan becomes tasks, tasks become reviewed code, and reviewed code becomes a shipped project — visualized the whole way through.

It's the vibe-coding feeling, with a team that actually finishes the job.

---

## 📄 License

MIT
