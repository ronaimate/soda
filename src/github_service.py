"""
GitHub API service layer for Soda application.
Centralizes all GitHub API interactions and repository operations.
"""
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import git
import httpx

logger = logging.getLogger("soda.github")


class GitHubService:
    """Service for GitHub API operations."""
    
    GITHUB_API_URL = "https://api.github.com"
    GITHUB_RAW_URL = "https://github.com"
    
    def __init__(self, username: str, token: str):
        self.username = username
        self.token = token
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
    
    def get_authenticated_repo_url(self, repo_url: str) -> str:
        """Convert public repo URL to authenticated URL with credentials."""
        if not repo_url:
            return repo_url
        
        auth_url = repo_url.replace(
            "https://github.com/",
            f"https://{self.username}:{self.token}@github.com/"
        )
        auth_url = auth_url.replace(
            "http://github.com/",
            f"https://{self.username}:{self.token}@github.com/"
        )
        return auth_url
    
    def get_repo_url(self, repo_name: str) -> str:
        """Get public GitHub repository URL."""
        return f"{self.GITHUB_RAW_URL}/{self.username}/{repo_name}"
    
    def get_authenticated_clone_url(self, repo_name: str) -> str:
        """Get authenticated clone URL with credentials."""
        return f"https://{self.username}:{self.token}@github.com/{self.username}/{repo_name}.git"
    
    async def check_repo_exists(self, repo_name: str) -> bool:
        """Check if a repository exists."""
        url = f"{self.GITHUB_API_URL}/repos/{self.username}/{repo_name}"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=self.headers, timeout=30)
                return response.status_code == 200
            except Exception as e:
                logger.error(f"Failed to check repo existence: {e}")
                return False
    
    async def create_repository(
        self,
        repo_name: str,
        description: str = "",
        private: bool = False,
        auto_init: bool = False
    ) -> Dict[str, Any]:
        """
        Create a new GitHub repository.
        
        Args:
            repo_name: Name of the repository
            description: Repository description
            private: Whether the repo should be private
            auto_init: Whether to initialize with README
            
        Returns:
            Dict with 'success', 'data', and 'error' keys
        """
        url = f"{self.GITHUB_API_URL}/user/repos"
        data = {
            "name": repo_name,
            "description": description,
            "private": private,
            "auto_init": auto_init
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=data, headers=self.headers, timeout=30)
                
                if response.status_code in [200, 201]:
                    logger.info(f"Repository created: {self.username}/{repo_name}")
                    return {
                        "success": True,
                        "data": response.json(),
                        "error": None
                    }
                else:
                    error_msg = f"GitHub API error {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    return {
                        "success": False,
                        "data": None,
                        "error": error_msg
                    }
            except Exception as e:
                error_msg = f"Failed to create repository: {e}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "data": None,
                    "error": error_msg
                }
    
    async def create_pull_request(
        self,
        repo_name: str,
        title: str,
        head: str,
        base: str,
        body: str = ""
    ) -> Dict[str, Any]:
        """
        Create a pull request.

        
        Args:
            repo_name: Repository name
            title: PR title
            head: Source branch
            base: Target branch
            body: PR description
            
        Returns:
            Dict with 'success', 'pr_url', and 'error' keys
        """
        url = f"{self.GITHUB_API_URL}/repos/{self.username}/{repo_name}/pulls"
        data = {
            "title": title,
            "head": head,
            "base": base,
            "body": body
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=data, headers=self.headers, timeout=30)
                
                if response.status_code in [200, 201]:
                    pr_data = response.json()
                    pr_url = pr_data.get("html_url", "")
                    logger.info(f"Pull request created: {pr_url}")
                    return {
                        "success": True,
                        "pr_url": pr_url,
                        "pr_data": pr_data,
                        "error": None
                    }
                else:
                    error_msg = f"Failed to create PR: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return {
                        "success": False,
                        "pr_url": None,
                        "pr_data": None,
                        "error": error_msg
                    }
            except Exception as e:
                error_msg = f"PR creation error: {e}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "pr_url": None,
                    "pr_data": None,
                    "error": error_msg
                }
    
    async def commit_gitignore(self, repo_name: str, branch: str = "main") -> bool:
        """
        Commit a default .gitignore file to the repository.
        
        Args:
            repo_name: Repository name
            branch: Branch to commit to
            
        Returns:
            True if successful, False otherwise
        """
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Database
*.sqlite3
*.db

# Temporary files
*.tmp
*.temp
.cache/
"""
        
        # Check if .gitignore already exists
        check_url = f"{self.GITHUB_API_URL}/repos/{self.username}/{repo_name}/contents/.gitignore"
        
        async with httpx.AsyncClient() as client:
            try:
                check_response = await client.get(check_url, headers=self.headers, timeout=30)
                
                if check_response.status_code == 200:
                    logger.info(f".gitignore already exists in {repo_name}")
                    return True
                
                # Create .gitignore
                content_encoded = base64.b64encode(gitignore_content.encode()).decode()
                put_url = f"{self.GITHUB_API_URL}/repos/{self.username}/{repo_name}/contents/.gitignore"
                put_data = {
                    "message": "Add default .gitignore",
                    "content": content_encoded,
                    "branch": branch
                }
                
                put_response = await client.put(put_url, json=put_data, headers=self.headers, timeout=30)
                
                if put_response.status_code in [200, 201]:
                    logger.info(f".gitignore committed to {repo_name}")
                    return True
                else:
                    logger.error(f"Failed to commit .gitignore: {put_response.status_code}")
                    return False
            except Exception as e:
                logger.error(f"Error committing .gitignore: {e}")
                return False

    async def merge_branch(self, repo_name: str, head: str, base: str = "main") -> Dict[str, Any]:
        """
        Merge a branch into the base branch using GitHub's merge API.
        Returns: {"status": "merged" | "conflict" | "error", "message": str, "data": ...}
        """
        url = f"{self.GITHUB_API_URL}/repos/{self.username}/{repo_name}/merges"
        payload = {
            "base": base,
            "head": head,
            "commit_message": f"Merge branch '{head}' into {base}",
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(url, json=payload, headers=self.headers)
                if response.status_code in (200, 201):
                    return {"status": "merged", "data": response.json()}
                elif response.status_code == 409:
                    try:
                        err = response.json()
                    except Exception:
                        err = {"message": response.text}
                    return {
                        "status": "conflict",
                        "message": err.get("message", "Merge conflict"),
                    }
                elif response.status_code == 404:
                    return {"status": "error", "message": f"Branch not found: {head}"}
                elif response.status_code == 422:
                    try:
                        err = response.json()
                    except Exception:
                        err = {"message": response.text}
                    msg = err.get("message", "Merge failed")
                    if "no commits" in msg.lower() or "already" in msg.lower():
                        return {"status": "merged", "message": msg, "data": err}
                    return {"status": "conflict", "message": msg}
                else:
                    try:
                        err = response.json()
                    except Exception:
                        err = {"message": response.text}
                    return {
                        "status": "error",
                        "message": err.get("message", f"HTTP {response.status_code}"),
                    }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def close(self) -> None:
        """No-op for backward compatibility."""
        pass


class GitOperations:
    """Service for local git operations."""
    
    @staticmethod
    def clone_repository(
        repo_url: str,
        target_dir: Path,
        branch: str = "main"
    ) -> bool:
        """
        Clone a repository to target directory.
        
        Args:
            repo_url: Repository URL (can be authenticated)
            target_dir: Target directory path
            branch: Branch to checkout
            
        Returns:
            True if successful, False otherwise
        """
        try:
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            
            repo = git.Repo.clone_from(repo_url, target_dir)
            
            if branch != "main":
                repo.git.checkout(branch)
            
            logger.info(f"Repository cloned to {target_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to clone repository: {e}")
            return False
    
    @staticmethod
    def create_and_push_branch(
        repo_path: Path,
        branch_name: str,
        commit_message: str,
        remote_name: str = "origin"
    ) -> bool:
        """
        Create a new branch, commit changes, and push to remote.
        
        Args:
            repo_path: Path to git repository
            branch_name: Name of the new branch
            commit_message: Commit message
            remote_name: Remote name (default: origin)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            repo = git.Repo(repo_path)
            
            # Create and checkout new branch
            repo.git.checkout("-b", branch_name)
            
            # Add all changes
            repo.git.add(A=True)
            
            # Commit if there are changes
            if repo.is_dirty() or repo.untracked_files:
                repo.index.commit(commit_message)
                
                # Push to remote
                repo.git.push(remote_name, branch_name)
                
                logger.info(f"Branch '{branch_name}' created and pushed")
                return True
            else:
                logger.info(f"No changes to commit in {repo_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to create and push branch: {e}")
            return False
    
    @staticmethod
    def copy_directory(src: Path, dest: Path, exclude_patterns: Optional[list[str]] = None):
        """
        Copy directory contents with exclusion patterns.
        
        Args:
            src: Source directory
            dest: Destination directory
            exclude_patterns: List of patterns to exclude
        """
        import shutil
        
        exclude_patterns = exclude_patterns or []
        
        if not dest.exists():
            dest.mkdir(parents=True, exist_ok=True)
        
        for item in src.iterdir():
            # Check if item should be excluded
            if any(pattern in str(item) for pattern in exclude_patterns):
                continue
            
            dest_item = dest / item.name
            
            if item.is_dir():
                if dest_item.exists():
                    shutil.rmtree(dest_item)
                shutil.copytree(item, dest_item)
            else:
                shutil.copy2(item, dest_item)

    async def merge_branch(self, head: str, base: str = "main") -> Dict[str, Any]:
        """
        Merge a branch into the base branch using GitHub's merge API.
        Returns: {"status": "merged" | "conflict" | "error", "message": str, "data": ...}
        """
        try:
            url = f"{self.api_base}/repos/{self.username}/{self.repo}/merges"
            payload = {
                "base": base,
                "head": head,
                "commit_message": f"Merge branch '{head}' into {base}",
            }
            resp = await self.client.post(url, json=payload)
            if resp.status_code in (200, 201):
                return {"status": "merged", "data": resp.json()}
            elif resp.status_code == 409:
                # Conflict or branch protection
                try:
                    err = resp.json()
                except Exception:
                    err = {"message": resp.text}
                return {
                    "status": "conflict",
                    "message": err.get("message", "Merge conflict"),
                }
            elif resp.status_code == 404:
                return {"status": "error", "message": f"Branch not found: {head}"}
            elif resp.status_code == 422:
                # Could be: head == base, no commits between, or branch protection
                try:
                    err = resp.json()
                except Exception:
                    err = {"message": resp.text}
                msg = err.get("message", "Merge failed")
                # "already up to date" or "no commits" - treat as success
                if "no commits" in msg.lower() or "already" in msg.lower():
                    return {"status": "merged", "message": msg, "data": err}
                return {"status": "conflict", "message": msg}
            else:
                try:
                    err = resp.json()
                except Exception:
                    err = {"message": resp.text}
                return {
                    "status": "error",
                    "message": err.get("message", f"HTTP {resp.status_code}"),
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
