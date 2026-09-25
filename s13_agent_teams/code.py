#!/usr/bin/env python3
"""
s13: Agent Teams - persistent teammates with shared tasks and mailboxes.

运行：  python s13_agent_teams/code.py
依赖：  pip install anthropic python-dotenv，并准备带 ANTHROPIC_API_KEY 的 .env

    +------+  spawn(task_id)  +----------+  result  +------+
    | Lead | ---------------> |   WORK   | -------> | IDLE |
    +--+---+                  +----+-----+          +--+---+
       ^                           |                   |
       | team events               | tools             | wait
       |                           v                   v
    +--+-----------+          +----------+        +----------+
    | MessageBus   |          | Task cwd | <----- | Mailbox  |
    +--------------+          +----------+  claim +----------+

    .tasks/       共享的任务记录和依赖关系
    .mailboxes/   消息、结果和协议响应
    .worktrees/   可选的任务绑定工作目录
"""

import fcntl
import json
import os
import random
import re
import secrets
import select
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from pathlib import Path

try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# -- 任务系统 --

TASKS_DIR = WORKDIR / ".tasks"                       # 共享任务 JSON 文件的存放目录
TASKS_ROOT = TASKS_DIR.resolve()                     # 任务目录的绝对路径，用来做越界校验
TASK_ID_PATTERN = re.compile(r"^task_[0-9a-f]{8}$")  # 合法任务 ID 的格式：task_ 加 8 位十六进制
task_lock = threading.RLock()                        # 可重入锁，保护任务数据的并发读写
TASK_LOCK_PATH = TASKS_DIR / ".lock"                 # 跨进程文件锁的锁文件路径
_task_store_state = threading.local()                # 线程本地状态，记录加锁深度和锁文件句柄

# owner -> {"task_id": str, "cwd": Path}。一个队友同一时间只能领一个任务，
# 所有文件系统工具都通过这张登记表找到自己的工作目录。
teammate_assignments: dict[str, dict[str, object]] = {}  # 队友当前认领的任务 ID 和工作目录
assignment_versions: dict[str, int] = {}                 # 认领版本号，任务变更时加一，让旧审批失效


# 用 contextmanager 装饰器，把下面的函数包装成可以用 with 的上下文管理器
@contextmanager
def task_store_lock():
    """让任务的修改在不同线程和进程之间串行执行，防止并发写坏数据。"""
    with task_lock:
        # 线程本地变量记录加锁深度，0 表示本线程还没持有文件锁
        depth = getattr(_task_store_state, "depth", 0)
        if depth == 0:
            # 本线程第一次进入：先确保任务目录存在，再打开锁文件
            TASKS_DIR.mkdir(parents=True, exist_ok=True)
            handle = TASK_LOCK_PATH.open("a+", encoding="utf-8")
            # flock 文件锁负责不同进程之间的互斥，线程锁只能管本进程
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            _task_store_state.handle = handle
        # 同一线程嵌套调用时只累加深度，不重复加文件锁
        _task_store_state.depth = depth + 1
        try:
            yield
        finally:
            _task_store_state.depth -= 1
            # 深度归零说明最外层退出，这时才真正释放文件锁并关闭句柄
            if _task_store_state.depth == 0:
                handle = _task_store_state.handle
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
                del _task_store_state.handle


def advance_assignment_version(owner: str):
    """让旧的计划审批失效，但不清除“必须先交计划”的要求。

    每次认领变动时把版本号加一，作废该队友之前提交、还没批完的计划请求；
    如果它被要求交计划，状态重置为 required，等它重新提交。

    Args:
        owner: 队友名字，即认领记录和门禁表里的键。
    """
    with task_lock:
        assignment_versions[owner] = assignment_versions.get(owner, 0) + 1
        gates = globals().get("plan_gates")
        request_ids = globals().get("plan_request_ids")
        team = globals().get("team_lock")
        if team is not None:
            team.acquire()
        try:
            if (isinstance(gates, dict) and owner in gates
                    and gates[owner] != "not_required"):
                gates[owner] = "required"
            if isinstance(request_ids, dict):
                request_ids.pop(owner, None)
        finally:
            if team is not None:
                team.release()


@dataclass
class Task:
    """一条共享任务的记录，每个任务对应 .tasks/ 目录下的一个 JSON 文件。"""

    id: str                            # 任务唯一 ID，形如 task_ 加 8 位十六进制
    subject: str                       # 任务标题，一句话说明要做什么
    description: str                   # 任务详情，给认领者的补充说明
    status: str                        # 取值：pending | in_progress | completed
    owner: str | None                  # 当前认领者名字，None 表示还没人认领
    blockedBy: list[str]               # 依赖的任务 ID 列表，全部完成才能开工
    worktree: str | None = None        # 绑定的 worktree 名字，None 表示直接用主工作目录


def _task_path(task_id: str) -> Path:
    """把任务 ID 换算成 .tasks/ 目录下的 JSON 文件路径。

    Args:
        task_id: 任务 ID，形如 task_ 加 8 位十六进制。

    Returns:
        Path: 任务文件的真实绝对路径。

    Raises:
        ValueError: ID 格式不合法，或算出的路径越出任务目录时抛出。
    """
    if not isinstance(task_id, str) or not TASK_ID_PATTERN.fullmatch(task_id):
        raise ValueError(f"Invalid task ID: {task_id!r}")
    path = (TASKS_DIR / f"{task_id}.json").resolve()
    if (not TASKS_ROOT.is_relative_to(WORKDIR.resolve())
            or not path.is_relative_to(TASKS_ROOT)):
        raise ValueError(f"Invalid task ID: {task_id!r}")
    return path


def create_task(subject: str, description: str = "") -> Task:
    """新建一个任务文件，返回生成的任务对象。

    Args:
        subject: 任务标题，一句话说明要做什么，不能是空白。
        description: 任务详情，给认领者的补充说明，默认为空。

    Returns:
        Task: 新建的任务对象，状态为 pending、无人认领、无依赖。

    Raises:
        ValueError: 标题是空白字符串时抛出。
        RuntimeError: 连续 100 次都撞上重复 ID、分配不出唯一 ID 时抛出。
    """
    subject = subject.strip()
    if not subject:
        raise ValueError("Task subject cannot be empty")
    with task_store_lock():
        for _ in range(100):
            task = Task(
                id=f"task_{secrets.token_hex(4)}",
                subject=subject,
                description=description,
                status="pending",
                owner=None,
                blockedBy=[],
            )
            try:
                with _task_path(task.id).open("x", encoding="utf-8") as handle:
                    json.dump(asdict(task), handle, indent=2)
                return task
            except FileExistsError:
                continue
    raise RuntimeError("Could not allocate a unique task ID")


def _task_depends_on(task_id: str, target_id: str) -> bool:
    """判断 task_id 是否直接或间接依赖 target_id。

    Args:
        task_id: 起点任务 ID。
        target_id: 要找的依赖目标 ID。

    Returns:
        bool: 沿着 blockedBy 一路往下找能碰到 target_id 就返回 True，
            否则返回 False。
    """
    pending = [task_id]
    visited = set()
    while pending:
        current = pending.pop()
        if current == target_id:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(load_task(current).blockedBy)
    return False


def update_task(task_id: str, addBlockedBy: list[str]) -> Task:
    """等 create_task 返回真实任务 ID 后，再用它添加依赖关系。

    Args:
        task_id: 要加依赖的任务 ID，必须还是 pending 且没人认领。
        addBlockedBy: 依赖的任务 ID 列表，写入前会自动去重。

    Returns:
        Task: 更新好 blockedBy 之后的任务对象。

    Raises:
        ValueError: addBlockedBy 不是列表、任务依赖自己、依赖不存在、
            或加了会形成依赖环时抛出。
    """
    if not isinstance(addBlockedBy, list):
        raise ValueError("addBlockedBy must be a list of task IDs")

    with task_store_lock():
        task = load_task(task_id)
        if task.status != "pending" or task.owner is not None:
            raise ValueError(
                f"Task {task_id} dependencies can only be updated while "
                "pending and unowned"
            )

        dependencies = list(dict.fromkeys(addBlockedBy))
        for dependency in dependencies:
            if dependency == task_id:
                raise ValueError("Task cannot depend on itself")
            if not _task_path(dependency).is_file():
                raise ValueError(f"Dependency not found: {dependency}")
            if dependency not in task.blockedBy and _task_depends_on(
                dependency, task_id
            ):
                raise ValueError(
                    f"Dependency cycle detected: {task_id} -> {dependency}"
                )

        task.blockedBy.extend(
            dependency for dependency in dependencies
            if dependency not in task.blockedBy
        )
        save_task(task)
        return task


def save_task(task: Task):
    """把任务对象写回对应的 JSON 文件。

    先写临时文件再原子替换，避免写一半被其他进程读到坏数据。

    Args:
        task: 要保存的任务对象。
    """
    with task_store_lock():
        path = _task_path(task.id)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(asdict(task), indent=2), encoding="utf-8"
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def load_task(task_id: str) -> Task:
    """从 JSON 文件读出一个任务对象。

    Args:
        task_id: 任务 ID。

    Returns:
        Task: 从文件内容还原的任务对象。

    Raises:
        FileNotFoundError: 任务文件不存在时抛出。
        ValueError: 文件里的 ID 与 task_id 对不上、或状态值非法时抛出。
    """
    with task_lock:
        data = json.loads(_task_path(task_id).read_text(encoding="utf-8"))
        task = Task(**data)
        if task.id != task_id:
            raise ValueError(f"Task file ID does not match {task_id}")
        if task.status not in {"pending", "in_progress", "completed"}:
            raise ValueError(f"Invalid task status: {task.status}")
        return task


def list_tasks() -> list[Task]:
    """列出 .tasks/ 目录下的全部任务。

    Returns:
        list[Task]: 按文件名（即任务 ID）排序的任务列表；
            任务目录还不存在时返回空列表。

    Raises:
        ValueError: 任务目录路径越出工作区时抛出。
    """
    with task_lock:
        if not TASKS_DIR.exists():
            return []
        if not TASKS_ROOT.is_relative_to(WORKDIR.resolve()):
            raise ValueError("Tasks directory escapes workspace")
        return [load_task(path.stem)
                for path in sorted(TASKS_DIR.glob("task_*.json"))]


def get_task(task_id: str) -> str:
    """以 JSON 字符串返回任务的完整信息。

    Args:
        task_id: 任务 ID。

    Returns:
        str: 缩进格式化的任务 JSON 文本；异常由 load_task 原样抛出。
    """
    task = load_task(task_id)
    return json.dumps(asdict(task), indent=2)


def can_start(task_id: str) -> bool:
    """检查 blockedBy 里的依赖是否全部完成。
    依赖文件缺失时同样视为被阻塞。

    Args:
        task_id: 任务 ID。

    Returns:
        bool: 所有依赖都存在且已完成时返回 True，否则返回 False。
    """
    task = load_task(task_id)
    for dep_id in task.blockedBy:
        try:
            dep_path = _task_path(dep_id)
        except ValueError:
            return False
        if not dep_path.exists():
            return False
        if load_task(dep_id).status != "completed":
            return False
    return True


def _owner_in_progress(owner: str) -> Task | None:
    """找出该认领者手里正在进行中的任务。

    Args:
        owner: 认领者名字。

    Returns:
        Task | None: 它名下 in_progress 的任务；一个都没有时返回 None。
    """
    return next((task for task in list_tasks()
                 if task.status == "in_progress" and task.owner == owner), None)


def _incomplete_dependencies(task: Task) -> list[str]:
    """找出该任务还没完成的依赖。

    Args:
        task: 要检查的任务对象。

    Returns:
        list[str]: 未完成依赖的 ID 列表；依赖 ID 非法或文件缺失也计入。
    """
    incomplete = []
    for dep_id in task.blockedBy:
        try:
            dep_path = _task_path(dep_id)
        except ValueError:
            incomplete.append(dep_id)
            continue
        if not dep_path.exists() or load_task(dep_id).status != "completed":
            incomplete.append(dep_id)
    return incomplete


def claim_task(task_id: str, owner: str = "agent") -> str:
    """原子地认领一个任务，并把认领者绑定到对应的工作目录。

    Args:
        task_id: 要认领的任务 ID。
        owner: 认领者名字，默认为 “agent”。

    Returns:
        str: 成功时返回认领结果文本；失败时返回原因说明而不抛异常，
            例如任务不处于 pending、已被别人认领、认领者手里还有任务、
            依赖没完成、worktree 不可用等情况。
    """
    with task_store_lock():
        task = load_task(task_id)
        if task.status != "pending":
            return f"Task {task_id} is {task.status}, cannot claim"
        if task.owner:
            return f"Task {task_id} is already owned by {task.owner}"
        assignment = teammate_assignments.get(owner)
        if assignment:
            return (f"Owner {owner} must finish the current work turn for "
                    f"{assignment['task_id']} before claiming another task")
        current = _owner_in_progress(owner)
        if current:
            return (f"Owner {owner} must complete {current.id} before "
                    "claiming another task")
        if not can_start(task_id):
            return f"Blocked by: {_incomplete_dependencies(task)}"
        cwd, error = task_worktree_cwd(task)
        if error:
            return f"Cannot claim {task_id}: {error}"
        task.owner = owner
        task.status = "in_progress"
        save_task(task)
        teammate_assignments[owner] = {"task_id": task.id, "cwd": cwd}
        advance_assignment_version(owner)
    print(f"  [claim] {task.subject} -> in_progress (owner: {owner})")
    return f"Claimed {task.id} ({task.subject})"


def complete_task(task_id: str, owner: str = "agent") -> str:
    """只有调用者本人是任务归属者时，才允许完成任务。

    Args:
        task_id: 要完成的任务 ID。
        owner: 调用者名字，默认为 “agent”。

    Returns:
        str: 成功时返回完成结果文本；若这次完成解锁了其他任务，
            结果里会附带列出被解锁的任务标题。失败时返回原因说明而不抛异常，
            例如 plan 门禁未通过、调用者不是归属者等情况。
    """
    with task_store_lock():
        task = load_task(task_id)
        if task.status != "in_progress":
            return f"Task {task_id} is {task.status}, cannot complete"
        if task.owner != owner:
            return (f"Task {task_id} is owned by {task.owner}, "
                    f"not {owner}; cannot complete")
        gate = globals().get("plan_gates", {}).get(owner, "not_required")
        if gate in {"required", "pending", "rejected"}:
            return f"Task {task_id} cannot complete while plan status is {gate}"
        assignment = teammate_assignments.get(owner)
        if not assignment or assignment.get("task_id") != task.id:
            cwd, error = task_worktree_cwd(task)
            if error:
                return f"Task {task_id} cannot complete: {error}"
            teammate_assignments[owner] = {"task_id": task.id, "cwd": cwd}
        task.status = "completed"
        save_task(task)
        unblocked = [t.subject for t in list_tasks()
                     if t.status == "pending" and t.blockedBy and can_start(t.id)]
    print(f"  [complete] {task.subject}")
    msg = f"Completed {task.id} ({task.subject})"
    if unblocked:
        msg += f"\nUnblocked: {', '.join(unblocked)}"
        print(f"  [unblocked] {', '.join(unblocked)}")
    return msg


# -- 任务绑定的 Worktree --

WORKTREES_DIR = WORKDIR / ".worktrees"          # 任务专属 worktree 的存放目录
WORKTREES_ROOT = WORKTREES_DIR.resolve()         # worktree 目录的绝对路径，用来做越界校验
VALID_WORKTREE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")     # 合法名字格式：字母或数字开头，总长 1~64，不含路径分隔符


def validate_worktree_name(name: str) -> str | None:
    """校验 worktree 名字是否合法。

    Args:
        name: 待校验的 worktree 名字。

    Returns:
        str | None: 不合法时返回错误说明文本；合法时返回 None。
    """
    if not isinstance(name, str) or not VALID_WORKTREE_NAME.fullmatch(name):
        return ("worktree name must be 1-64 letters, digits, dots, "
                "underscores, or dashes, and start with a letter or digit")
    if name in {".", ".."} or ".." in name:
        return "worktree name cannot contain '..'"
    return None


def _worktree_path(name: str) -> Path:
    """把 worktree 名字换算成 .worktrees/ 下的真实路径。

    Args:
        name: worktree 名字。

    Returns:
        Path: worktree 目录的真实绝对路径。

    Raises:
        ValueError: 算出的路径越出 .worktrees/ 目录、或正好等于目录本身时抛出。
    """
    path = (WORKTREES_DIR / name).resolve()
    if (not WORKTREES_ROOT.is_relative_to(WORKDIR.resolve())
            or not path.is_relative_to(WORKTREES_ROOT)
            or path == WORKTREES_ROOT):
        raise ValueError(f"Worktree path escapes directory: {name!r}")
    return path


def _worktree_branch(name: str) -> str:
    """按约定生成 worktree 对应的 Git 分支名。

    Args:
        name: worktree 名字。

    Returns:
        str: “wt/” 前缀加名字组成的分支名，例如 wt/login。
    """
    return f"wt/{name}"


def _run_git(args: list[str], cwd: Path | None = None) -> tuple[bool, str]:
    """不经 shell 解析直接运行 Git，保留原始机器输出。

    Args:
        args: 传给 git 的参数列表。
        cwd: 执行命令的工作目录，默认用项目根目录。

    Returns:
        tuple[bool, str]: 第一项表示命令是否成功（退出码为 0），
            第二项是 stdout 加 stderr 的合并输出；命令没跑起来时返回错误说明。
    """
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd or WORKDIR,
            capture_output=True, text=True, errors="replace", timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output or "(no output)"


def run_git(args: list[str], cwd: Path | None = None) -> tuple[bool, str]:
    """运行 Git，只对返回给模型的文本做长度截断。

    Args:
        args: 传给 git 的参数列表。
        cwd: 执行命令的工作目录，默认用项目根目录。

    Returns:
        tuple[bool, str]: 同 _run_git，但输出最多保留 5000 个字符。
    """
    ok, output = _run_git(args, cwd)
    return ok, output[:5000]


def _registered_worktrees() -> tuple[dict[Path, dict[str, str]], str | None]:
    """读取 Git 登记的全部 worktree 清单。

    Returns:
        tuple[dict[Path, dict[str, str]], str | None]: 成功时返回
            “路径 -> 登记信息”的字典和 None；读取失败时返回空字典和错误说明。
    """
    ok, output = _run_git(["worktree", "list", "--porcelain"])
    if not ok:
        return {}, f"cannot read Git worktree registry: {output}"
    entries: dict[Path, dict[str, str]] = {}
    current: dict[str, str] = {}
    for line in output.splitlines() + [""]:
        if not line:
            raw_path = current.get("worktree")
            if raw_path:
                entries[Path(raw_path).resolve()] = current
            current = {}
            continue
        key, _, value = line.partition(" ")
        current[key] = value
    return entries, None


def _registered_worktree(name: str) -> tuple[Path | None, str | None]:
    """查某个 worktree 在 Git 里的登记情况，核对路径和分支都符合预期。

    Args:
        name: worktree 名字。

    Returns:
        tuple[Path | None, str | None]: 校验通过时返回（路径, None）；
            名字非法、未在 Git 登记、目录缺失或分支不符时返回（None, 错误说明）。
    """
    try:
        path = _worktree_path(name)
    except ValueError as exc:
        return None, str(exc)
    entries, error = _registered_worktrees()
    if error:
        return None, error
    if path not in entries:
        return None, f"worktree '{name}' is not registered with Git"
    if not path.is_dir():
        return None, f"worktree '{name}' is missing at {path}"
    expected_branch = f"refs/heads/{_worktree_branch(name)}"
    if entries[path].get("branch") != expected_branch:
        return None, (f"worktree '{name}' is not registered on expected "
                      f"branch '{_worktree_branch(name)}'")
    return path, None

 
    return (path or WORKDIR), error

def task_worktree_cwd(task: Task) -> tuple[Path, str | None]:
    """解析任务该用的工作目录，worktree 绑定坏了就报错挡住。

    Args:
        task: 要解析目录的任务对象。

    Returns:
        tuple[Path, str | None]: 任务没绑定 worktree 时直接返回
            （主工作目录, None）；绑定完好时返回（worktree 路径, None）；
            绑定损坏时返回（主工作目录, 错误说明），调用方看到
            error 就该停止，不能拿回退目录继续干活。
    """
    if not task.worktree:
        return WORKDIR, None
    path, error = _registered_worktree(task.worktree)
    return (path or WORKDIR), error
 
def assignment_cwd(owner: str) -> Path:
    """查出认领者当前该用的工作目录，并核对登记信息仍然有效。

    Args:
        owner: 认领者名字。

    Returns:
        Path: 认领者当前任务对应的工作目录；没有认领记录时返回主工作目录。

    Raises:
        ValueError: 任务已不属于该认领者、worktree 绑定损坏、
            或登记的工作目录和实际不一致时抛出。
    """
    with task_lock:
        assignment = teammate_assignments.get(owner)
        task = _owner_in_progress(owner)
        if task and (not assignment or assignment.get("task_id") != task.id):
            cwd, error = task_worktree_cwd(task)
            if error:
                raise ValueError(error)
            assignment = {"task_id": task.id, "cwd": cwd}
            teammate_assignments[owner] = assignment
        elif not assignment:
            return WORKDIR
        task = load_task(str(assignment["task_id"]))
        if task.status not in {"in_progress", "completed"} or task.owner != owner:
            raise ValueError(f"Assignment for {owner} is no longer active")
        cwd, error = task_worktree_cwd(task)
        if error:
            raise ValueError(error)
        if cwd.resolve() != Path(assignment["cwd"]).resolve():
            raise ValueError(f"Assignment cwd changed for task {task.id}")
        return cwd


def release_completed_assignment(owner: str) -> bool:
    """只在模型回合边界释放已完成任务的目录占用。

    Args:
        owner: 认领者名字。

    Returns:
        bool: 成功释放返回 True；没有认领记录、或任务不是该认领者
            已完成的状态时返回 False。
    """
    with task_lock:
        assignment = teammate_assignments.get(owner)
        if not assignment:
            return False
        task = load_task(str(assignment["task_id"]))
        if task.status != "completed" or task.owner != owner:
            return False
        teammate_assignments.pop(owner, None)
        advance_assignment_version(owner)
        if owner in globals().get("plan_gates", {}):
            globals()["plan_gates"][owner] = "not_required"
        return True


def release_teammate_assignment(owner: str):
    """线程退出时，把队友没做完的任务放回任务板。

    无论收尾过程是否出错，都会清掉认领记录、递增版本号并重置 plan 门禁。

    Args:
        owner: 队友名字。
    """
    with task_lock:
        try:
            task = _owner_in_progress(owner)
            if task:
                task.status = "pending"
                task.owner = None
                save_task(task)
        finally:
            teammate_assignments.pop(owner, None)
            advance_assignment_version(owner)
            if owner in globals().get("plan_gates", {}):
                globals()["plan_gates"][owner] = "not_required"


def create_worktree(name: str, task_id: str) -> str:
    """在全部输入校验通过后，创建并绑定专用 worktree。

    Args:
        name: worktree 名字，需先通过 validate_worktree_name 校验。
        task_id: 要绑定的任务 ID，任务必须是 pending 且无人认领。

    Returns:
        str: 成功时返回创建结果文本；失败时返回错误说明。git 命令半途
            失败时会列出遗留产物并提示人工清理，不会删除任何 Git 数据。
    """
    error = validate_worktree_name(name)
    if error:
        return f"Error: {error}"
    try:
        path = _worktree_path(name)
        task_path = _task_path(task_id)
    except ValueError as exc:
        return f"Error: {exc}"
    branch = _worktree_branch(name)

    with task_lock:
        if not task_path.exists():
            return f"Error: Task {task_id} not found"
        task = load_task(task_id)
        if task.status != "pending" or task.owner is not None:
            return f"Error: Task {task_id} must be pending and unowned"
        if task.worktree:
            return f"Error: Task {task_id} already uses worktree '{task.worktree}'"
        if any(t.worktree == name for t in list_tasks() if t.id != task_id):
            return f"Error: Worktree '{name}' is already bound to another task"
        if path.exists():
            return f"Error: Worktree path already exists: {path}"

        ok, root = run_git(["rev-parse", "--show-toplevel"])
        if not ok or Path(root).resolve() != WORKDIR.resolve():
            return "Error: Working directory must be the root of a Git repository"
        ok, branch_check = run_git(["check-ref-format", "--branch", branch])
        if not ok:
            return f"Error: Invalid worktree branch '{branch}': {branch_check}"
        exists, _ = run_git(["show-ref", "--verify", "--quiet",
                             f"refs/heads/{branch}"])
        if exists:
            return f"Error: Branch '{branch}' already exists"
        entries, registry_error = _registered_worktrees()
        if registry_error:
            return f"Error: {registry_error}"
        if path in entries:
            return f"Error: Worktree path is already registered: {path}"

        WORKTREES_DIR.mkdir(parents=True, exist_ok=True)
        ok, result = run_git(["worktree", "add", "-b", branch,
                              str(path), "HEAD"])
        if not ok:
            entries, registry_error = _registered_worktrees()
            branch_exists, _ = run_git(
                ["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"]
            )
            artifacts = []
            if path.exists():
                artifacts.append(f"checkout path '{path}'")
            if registry_error is None and path in entries:
                artifacts.append("registered Git worktree")
            if branch_exists:
                artifacts.append(f"branch '{branch}'")
            if artifacts:
                return (
                    "Partial operation: git worktree add reported an error "
                    f"after leaving {', '.join(artifacts)}. Task {task_id} "
                    "remains unbound and no Git data was deleted. Run "
                    f"`git worktree list`, inspect '{path}' and '{branch}', "
                    "then keep or remove those artifacts manually after "
                    f"preserving any work. Git error: {result}"
                )
            return f"Git error: {result}"

        try:
            task.worktree = name
            save_task(task)
        except Exception as exc:
            return (f"Partial success: Worktree '{name}' was created at "
                    f"{path} on branch '{branch}', but task binding failed: "
                    f"{exc}. Git data was retained for manual recovery.")

    print(f"  \033[33m[worktree] created: {name} at {path}\033[0m")
    return f"Worktree '{name}' created at {path} for task {task_id}"


def remove_worktree(name: str, discard_changes: bool = False) -> str:
    """移除已注册的检出目录，但始终保留对应分支。

    Args:
        name: worktree 名字。
        discard_changes: 是否丢弃未提交的改动，默认为 False；为 False 时
            目录里若有未提交改动会拒绝移除。

    Returns:
        str: 成功时返回移除结果文本（注明分支已保留）；失败时返回错误说明。
            解绑任务失败时返回部分成功提示，需要人工处理。
    """
    error = validate_worktree_name(name)
    if error:
        return f"Error: {error}"

    with task_lock:
        path, error = _registered_worktree(name)
        if error:
            return f"Error: {error}"
        bound = [task for task in list_tasks() if task.worktree == name]
        if not bound:
            return f"Error: Worktree '{name}' is not bound to a task"
        active = [task for task in bound if task.status != "completed"]
        if active:
            return (f"Error: Worktree '{name}' is bound to active task "
                    f"{active[0].id}; complete it before removal")
        leased = [owner for owner, assignment in teammate_assignments.items()
                  if Path(assignment["cwd"]).resolve() == path.resolve()]
        if leased:
            return (f"Error: Worktree '{name}' is still in use by "
                    f"{', '.join(sorted(leased))}; wait for the turn to end")
        ok, status = run_git(
            ["status", "--porcelain", "--ignored"], cwd=path
        )
        if not ok:
            return f"Error: Cannot verify worktree '{name}' status: {status}"
        if status != "(no output)" and not discard_changes:
            changed = len([line for line in status.splitlines() if line.strip()])
            return (f"Error: Worktree '{name}' has {changed} uncommitted "
                    "change(s); preserve or discard them manually")

        args = ["worktree", "remove"]
        if discard_changes:
            args.append("--force")
        args.append(str(path))
        ok, result = run_git(args)
        if not ok:
            return f"Git error: {result}"

        try:
            for task in bound:
                task.worktree = None
                save_task(task)
        except Exception as exc:
            return (f"Partial success: Worktree '{name}' was removed and "
                    f"branch '{_worktree_branch(name)}' retained, but task "
                    f"unbinding failed: {exc}. Manual recovery is required.")

    print(f"  [worktree] removed: {name}; branch retained")
    return f"Worktree '{name}' removed; branch '{_worktree_branch(name)}' retained"


# -- 系统提示词 --

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": "Available tools: bash, read_file, write_file, edit_file, glob, "
             "create_task, update_task, list_tasks, get_task, claim_task, "
             "complete_task, "
             "spawn_teammate, list_teammates, send_message, request_shutdown, "
             "request_plan, review_plan, create_worktree.",
    "tasks": (
        "Create all task nodes first. Only after create_task returns "
        "runtime-generated IDs, use update_task with those exact IDs to add "
        "dependencies. Only the Lead changes task dependencies."
    ),
    # 团队协作规则：
    #  当并行干活确实有帮助时，先提出一个分工明确的小团队方案，等用户确认。
    #  用户确认之前，不许调用 spawn_teammate。
    #  确认之后，把相互独立的工作拆开，每个并行改动建一个 Task 来分派。
    #  分派已就绪的工作时，把 task_id 传给 spawn_teammate；只有当独立工作目录确实能避免编辑冲突时，才为任务创建绑定的 worktree。
    #  队友必须先完成手头的 Task，才能认领下一个。
    #  worktree 只是改变工具的默认工作目录，它不是沙箱（不做安全隔离）。
    #  删除 worktree 的事归宿主（Lead）或用户管。
    #  生成队友之后就结束当前回合，不要原地轮询它的状态；运行时会把团队事件送过来并唤醒 Lead。
    #  对这些事件做出反应；协调完成后把队友关掉。
    "teams": (
        "When parallel work would help, first propose a small team with clear "
        "responsibilities and wait for the user's confirmation. Do not call "
        "spawn_teammate before the user confirms. After confirmation, delegate "
        "independent work by creating a Task for each parallel change. Pass "
        "task_id to spawn_teammate when assigning ready work, then "
        "create a task-bound worktree only when a separate working directory "
        "would prevent conflicting edits. A teammate must complete its current "
        "Task before claiming another. A worktree changes tool default cwd "
        "only; it is not a sandbox. Worktree removal stays with the host or "
        "user. After spawning a teammate, end the current turn instead of "
        "polling its status; the runtime will deliver team events and wake the "
        "Lead. React to those events, and shut teammates down when "
        "coordination is complete."
    ),
    "workspace": f"Working directory: {WORKDIR}",
}

SYSTEM = "\n\n".join(PROMPT_SECTIONS.values())


# -- Base Tools --
# -- 基础工具 --

def safe_path(p: str, cwd: Path | None = None) -> Path:
    base = (cwd or WORKDIR).resolve()
    path = (base / p).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd or WORKDIR,
            capture_output=True,
            text=True, errors="replace",
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        output = output[:50000] if output else "(no output)"
        if result.returncode:
            return f"Error: command exited with status {result.returncode}\n{output}"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except OSError as exc:
        return f"Error: {type(exc).__name__}: {exc}"


def run_read(path: str, limit: int | None = None,
             cwd: Path | None = None) -> str:
    try:
        lines = safe_path(path, cwd).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str, cwd: Path | None = None) -> str:
    try:
        fp = safe_path(path, cwd)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str,
             cwd: Path | None = None) -> str:
    try:
        target = safe_path(path, cwd)
        content = target.read_text(encoding="utf-8")
        count = content.count(old_text)
        if count != 1:
            return f"Error: Expected 1 occurrence, found {count}"
        target.write_text(content.replace(old_text, new_text), encoding="utf-8")
        return f"Edited {path}"
    except Exception as exc:
        return f"Error: {exc}"


def run_glob(pattern: str, cwd: Path | None = None) -> str:
    try:
        base = (cwd or WORKDIR).resolve()
        matches = [
            str(path.relative_to(base))
            for path in sorted(base.glob(pattern))
            if path.resolve().is_relative_to(base)
        ]
        shown = matches[:200]
        if len(matches) > 200:
            shown.append("... (more matches omitted; narrow the pattern)")
        return "\n".join(shown) or "No files found"
    except Exception as exc:
        return f"Error: {exc}"


def _agent_cwd() -> tuple[Path | None, str | None]:
    """查出 Lead 自己（“agent”）当前该用的工作目录。

    Returns:
        tuple[Path | None, str | None]: 成功时返回（工作目录, None）；
            认领记录异常或 worktree 绑定损坏时返回（None, 错误说明），
            不往外抛异常。
    """
    try:
        return assignment_cwd("agent"), None
    except (FileNotFoundError, ValueError) as exc:
        return None, f"Error: Invalid task assignment: {exc}"


def run_agent_bash(command: str) -> str:
    cwd, error = _agent_cwd()
    return error or run_bash(command, cwd)


def run_agent_read(path: str, limit: int | None = None) -> str:
    cwd, error = _agent_cwd()
    return error or run_read(path, limit, cwd)


def run_agent_write(path: str, content: str) -> str:
    cwd, error = _agent_cwd()
    return error or run_write(path, content, cwd)


def run_agent_edit(path: str, old_text: str, new_text: str) -> str:
    cwd, error = _agent_cwd()
    return error or run_edit(path, old_text, new_text, cwd)


def run_agent_glob(pattern: str) -> str:
    cwd, error = _agent_cwd()
    return error or run_glob(pattern, cwd)


# -- Task Tools --
# -- 任务工具 --

def run_create_task(subject: str, description: str = "") -> str:
    """创建一个任务，并把结果包装成给模型看的文本。

    Args:
        subject: 任务标题。
        description: 任务详情，默认为空。

    Returns:
        str: 成功时返回“Created 任务ID: 标题”格式的文本。
    """
    task = create_task(subject, description)
    print(f"  \033[34m[create] {task.subject}\033[0m")
    return f"Created {task.id}: {task.subject}"


def run_update_task(task_id: str, addBlockedBy: list[str]) -> str:
    """给任务追加依赖，出错时不抛异常而是返回错误文本。

    Args:
        task_id: 要更新的任务 ID。
        addBlockedBy: 要追加的依赖任务 ID 列表。

    Returns:
        str: 成功时返回更新结果和当前 blockedBy 列表；依赖 ID 非法
            或任务不存在时返回错误说明。
    """
    try:
        task = update_task(task_id, addBlockedBy)
    except ValueError as exc:
        return f"Error: {exc}"
    except FileNotFoundError:
        return f"Error: Task {task_id} not found"
    dependencies = ", ".join(task.blockedBy) or "(none)"
    print(f"  \033[34m[update] {task.subject} blockedBy: {dependencies}\033[0m")
    return f"Updated {task.id} blockedBy: {dependencies}"


def run_list_tasks() -> str:
    """把任务板整理成一行一个任务的可读文本。

    Returns:
        str: 每行含状态图标、ID、标题、归属者、依赖和 worktree 信息；
            任务板为空时返回提示文本。
    """
    tasks = list_tasks()
    if not tasks:
        return "No tasks. Use create_task to add some."
    lines = []
    for t in tasks:
        icon = {"pending": "[ ]", "in_progress": "[~]",
                "completed": "[x]"}.get(t.status, "[?]")
        deps = f" (blockedBy: {', '.join(t.blockedBy)})" if t.blockedBy else ""
        owner = f" [{t.owner}]" if t.owner else ""
        worktree = f" (worktree: {t.worktree})" if t.worktree else ""
        lines.append(f"  {icon} {t.id}: {t.subject} "
                     f"[{t.status}]{owner}{deps}{worktree}")
    return "\n".join(lines)


def run_get_task(task_id: str) -> str:
    """查单个任务的详情。

    Args:
        task_id: 要查询的任务 ID。

    Returns:
        str: 成功时返回任务详情文本；ID 非法或任务不存在时返回错误说明。
    """
    try:
        return get_task(task_id)
    except ValueError as exc:
        return f"Error: {exc}"
    except FileNotFoundError:
        return f"Error: Task {task_id} not found"


def run_claim_task(task_id: str) -> str:
    """让 Lead 以“agent”的身份认领一个任务。

    Args:
        task_id: 要认领的任务 ID。

    Returns:
        str: 成功时返回认领结果文本；任务不处于 pending、已被认领
            或依赖没完成时返回错误说明。
    """
    try:
        return claim_task(task_id, owner="agent")
    except ValueError as exc:
        return f"Error: {exc}"
    except FileNotFoundError:
        return f"Error: Task {task_id} not found"


def run_complete_task(task_id: str) -> str:
    """让 Lead 以“agent”的身份完成一个任务。

    Args:
        task_id: 要完成的任务 ID。

    Returns:
        str: 成功时返回完成结果文本，解锁了其他任务会一并列出；
            plan 门禁没过或调用者不是归属者时返回错误说明。
    """
    try:
        return complete_task(task_id, owner="agent")
    except ValueError as exc:
        return f"Error: {exc}"
    except FileNotFoundError:
        return f"Error: Task {task_id} not found"


# -- MessageBus and Team Protocols --
# -- 消息总线与团队协议 --


MAILBOX_DIR = WORKDIR / ".mailboxes"       # 队友邮箱文件的存放目录
MAILBOX_ROOT = MAILBOX_DIR.resolve()       # 邮箱目录的绝对路径，用来做越界校验
VALID_AGENT_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")       # 合法名字格式：字母、数字、下划线、连字符，总长 1~64
RESERVED_TEAMMATE_NAMES = {"lead", "agent"}       # 保留名字：Lead 和默认主 agent 不许拿来当队友名


def is_valid_agent_name(name: str) -> bool:
    """判断名字是否合法：字母、数字、下划线、连字符，1~64 位。

    Args:
        name: 待检查的名字。

    Returns:
        bool: 合法返回 True，否则返回 False。
    """
    return bool(VALID_AGENT_NAME.fullmatch(name))


class MessageBus:
    """线程安全的文件邮箱，读取即清空。"""

    def __init__(self):
        """初始化锁和条件变量，供线程间等待和唤醒用。"""
        self._lock = threading.RLock()       # 可重入锁，保护所有邮箱的读写
        self._changed = threading.Condition(self._lock)       # 条件变量，挂起等新消息的线程

    def _path(self, agent: str) -> Path:
        """把收件人名字换算成邮箱文件路径，顺便做安全和越界校验。

        Args:
            agent: 收件人名字。

        Returns:
            Path: 该智能体邮箱文件的绝对路径。

        Raises:
            ValueError: 名字不合法或算出的路径越出邮箱目录时抛出。
        """
        if not is_valid_agent_name(agent):
            raise ValueError(f"Invalid mailbox recipient: {agent!r}")
        path = (MAILBOX_DIR / f"{agent}.jsonl").resolve()
        if not path.is_relative_to(MAILBOX_ROOT):
            raise ValueError(f"Mailbox path escapes directory: {agent!r}")
        return path

    def _read_unlocked(self, agent: str) -> list[dict]:
        """读走邮箱里的全部消息并清空文件（调用前要先拿住锁）。

        Args:
            agent: 收件人名字。

        Returns:
            list[dict]: 读到的消息列表；邮箱不存在时返回空列表。
        """
        inbox = self._path(agent)
        if not inbox.exists():
            return []
        msgs = [json.loads(line) for line in inbox.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        # 读走就删文件，实现“读取即清空”
        inbox.unlink()
        return msgs

    def send(self, from_agent: str, to_agent: str, content: str,
             msg_type: str = "message", metadata: dict | None = None):
        """往收件人的邮箱追加一条消息，并唤醒所有等消息的线程。

        Args:
            from_agent: 发件人名字。
            to_agent: 收件人名字。
            content: 消息正文。
            msg_type: 消息类型，默认为 "message"。
            metadata: 附加信息字典，默认为空。

        Raises:
            ValueError: 收件人名字不合法或路径越界时抛出。
        """
        msg = {"from": from_agent, "to": to_agent,
               "content": content, "type": msg_type,
               "ts": time.time(), "metadata": metadata or {}}
        with self._changed:
            MAILBOX_DIR.mkdir(parents=True, exist_ok=True)
            with self._path(to_agent).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(msg, ensure_ascii=True) + "\n")       # 一行一条 JSON，追加写入
            self._changed.notify_all()       # 叫醒所有正在等消息的线程
        print(f"  [bus] {from_agent} -> {to_agent}: "
              f"({msg_type}) {content[:50]}")

    def read_inbox(self, agent: str) -> list[dict]:
        """读走并清空该智能体邮箱里的全部消息。

        Args:
            agent: 收件人名字。

        Returns:
            list[dict]: 读到的消息列表；邮箱为空或不存在时返回空列表。
        """
        with self._lock:
            return self._read_unlocked(agent)

    def peek(self, agent: str) -> bool:
        """只看一眼邮箱里有没有消息，不取走。

        Args:
            agent: 收件人名字。

        Returns:
            bool: 邮箱文件存在且非空返回 True，否则返回 False。
        """
        with self._lock:
            inbox = self._path(agent)
            return inbox.exists() and inbox.stat().st_size > 0

    def wait_for_messages(self, agent: str,
                          timeout: float | None = None) -> list[dict]:
        """阻塞等待，直到该智能体有新消息或超时为止。

        Args:
            agent: 收件人名字。
            timeout: 最长等待秒数；None 表示一直等到有消息为止。

        Returns:
            list[dict]: 来了消息就返回消息列表（读走即清空）；
                超时还没消息则返回空列表。
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._changed:
            while not self.peek(agent):
                remaining = (None if deadline is None
                             else deadline - time.monotonic())
                if remaining is not None and remaining <= 0:
                    return []
                self._changed.wait(remaining)
            return self._read_unlocked(agent)


BUS = MessageBus()       # 全局唯一的消息总线，全队共用一个

# 取值：working | waiting_approval | idle | stopping
active_teammates: dict[str, str] = {}       # 在线队友 -> 当前状态
plan_gates: dict[str, str] = {}       # 队友 -> plan 门禁状态（not_required/required/pending/rejected）
plan_request_ids: dict[str, str] = {}       # 队友 -> 最近一次 plan 请求的 ID
team_lock = threading.RLock()       # 保护上面三个字典的可重入锁


@dataclass
class ProtocolState:
    """一次协议请求（计划审批 / 关机握手）的完整状态记录。

    从请求登记到 Lead 做出决定，响应消息的合法性都以这条记录为准来校验，
    防止伪造或过期的响应被应用。

    Attributes:
        request_id: 全局唯一的请求 ID，形如 req_ 加六位数字。
        type: 请求类型，取值 shutdown 或 plan_approval。
        sender: 请求发起方，计划审批时为队友、关机时为 lead。
        target: 请求接收方，计划审批时为 lead、关机时为队友。
        status: 请求状态，取值 pending / approved / rejected。
        payload: 请求附带的内容，例如计划正文。
        work_version: 登记时队友的认领版本号，任务变更后旧审批据此失效。
        task_id: 登记时队友认领的任务 ID，未认领时为 None。
        created_at: 请求登记时的时间戳。
    """
    request_id: str
    type: str
    sender: str
    target: str
    status: str
    payload: str
    work_version: int | None = None
    task_id: str | None = None
    created_at: float = field(default_factory=time.time)

 # 请求 ID -> 协议请求状态，等待 Lead 决定的请求都在这里
pending_requests: dict[str, ProtocolState] = {}      


def new_request_id() -> str:
    """随机生成一个当前未被占用的协议请求 ID。

    Returns:
        形如 req_ 加六位数字的字符串，保证不与 pending_requests
        中已登记的 ID 重复。
    """
    while True:
        request_id = f"req_{random.randint(0, 999999):06d}"
        if request_id not in pending_requests:
            return request_id


def match_response(response_type: str, request_id: str, approve: bool,
                   from_agent: str, to_agent: str) -> bool:
    """把一条协议响应和一条待处理请求进行匹配。

    依次校验请求存在、响应类型正确、响应方与请求双方对得上、
    请求尚未处理，全部通过才把状态落成 approved 或 rejected。

    Args:
        response_type: 响应消息类型，应为 shutdown_response 或
            plan_approval_response。
        request_id: 响应对应的请求 ID。
        approve: True 表示批准，False 表示拒绝。
        from_agent: 响应发送方，必须等于原请求的 target。
        to_agent: 响应接收方，必须等于原请求的 sender。

    Returns:
        匹配成功并已更新请求状态时返回 True；任何一步校验失败返回 False。
    """
    with team_lock:
        state = pending_requests.get(request_id)
        if not state:
            print(f"  [protocol] unknown request_id: {request_id}")
            return False
        expected = {
            "shutdown": "shutdown_response",
            "plan_approval": "plan_approval_response",
        }[state.type]
        if response_type != expected:
            print(f"  [protocol] expected {expected}, got {response_type}")
            return False
        if from_agent != state.target or to_agent != state.sender:
            print(f"  [protocol] {request_id} responder mismatch")
            return False
        if state.status != "pending":
            print(f"  [protocol] {request_id} already {state.status}")
            return False
        state.status = "approved" if approve else "rejected"
    print(f"  [protocol] {request_id} -> {state.status}")
    return True


def consume_lead_inbox() -> list[dict]:
    """先消费 Lead 收件箱并更新协议状态，再把事件交给模型。

    遍历收件箱中带 request_id 的 *_response 消息，逐条调用 match_response
    落实协议状态；之后无论匹配结果如何，都把原始消息原样交给调用方。

    Returns:
        从 Lead 收件箱读到的全部消息，按到达顺序排列，可能为空列表。
    """
    msgs = BUS.read_inbox("lead")
    for msg in msgs:
        metadata = msg.get("metadata", {})
        request_id = metadata.get("request_id", "")
        if request_id and msg.get("type", "").endswith("_response"):
            match_response(msg["type"], request_id,
                           metadata.get("approve", False),
                           msg.get("from", ""), msg.get("to", ""))
    return msgs


def format_team_events(msgs: list[dict]) -> str:
    """把团队事件消息列表压缩成一段可注入模型上下文的文本。

    Args:
        msgs: MessageBus 返回的消息列表，每条含 type/from/content
            以及可选的 metadata。

    Returns:
        以 "[Team events]" 开头的多行字符串，每行一条事件，
        带 request_id 的事件会附上该 ID；列表为空时只保留标题行。
    """
    lines = []
    for msg in msgs:
        metadata = msg.get("metadata", {})
        request_id = metadata.get("request_id")
        suffix = f" request_id={request_id}" if request_id else ""
        lines.append(
            f"[{msg['type']}{suffix}] {msg['from']}: {msg['content']}"
        )
    return "[Team events]\n" + "\n".join(lines)


def _last_assistant_text(content) -> str:
    """从助手消息的内容块里提取文本，供事件回显等场景使用。

    Args:
        content: 消息内容，元素为 SDK 的内容块对象或普通 dict。

    Returns:
        首个文本块的文字（去掉首尾空白）；没有文本块时返回空字符串。
    """
    for block in content:
        if getattr(block, "type", None) == "text":
            return block.text.strip()
        if isinstance(block, dict) and block.get("type") == "text":
            return str(block.get("text", "")).strip()
    return ""


def current_work_identity(owner: str) -> tuple[int, str | None]:
    """读取队友当前的认领版本号和任务 ID，作为审批时效校验的快照。

    Args:
        owner: 队友名字，即认领记录表中的键。

    Returns:
        (work_version, task_id) 二元组：版本号取自 assignment_versions，
        没有记录时为 0；task_id 取自当前认领记录，未认领时为 None。
    """
    with task_lock:
        assignment = teammate_assignments.get(owner)
        task_id = str(assignment["task_id"]) if assignment else None
        return assignment_versions.get(owner, 0), task_id


def _teammate_submit_plan(from_name: str, plan: str) -> str:
    """队友向 Lead 提交计划，登记协议请求并把计划发到 Lead 收件箱。

    提交前快照当前的认领版本号和任务 ID；若已有计划在等待审批则拒绝重复提交。

    Args:
        from_name: 提交计划的队友名字。
        plan: 计划正文，会原样发送给 Lead。

    Returns:
        提交结果说明：已有计划待审时返回等待提示；成功时返回带请求 ID 的提示。
    """
    with task_lock:
        assignment = teammate_assignments.get(from_name)
        task_id = str(assignment["task_id"]) if assignment else None
        work_version = assignment_versions.get(from_name, 0)
        with team_lock:
            if plan_gates.get(from_name) == "pending":
                return "A plan is already waiting for review."
            request_id = new_request_id()
            pending_requests[request_id] = ProtocolState(
                request_id=request_id,
                type="plan_approval",
                sender=from_name,
                target="lead",
                status="pending",
                payload=plan,
                work_version=work_version,
                task_id=task_id,
            )
            plan_gates[from_name] = "pending"
            plan_request_ids[from_name] = request_id
            active_teammates[from_name] = "waiting_approval"
    BUS.send(from_name, "lead", plan, "plan_approval_request",
             {"request_id": request_id})
    return f"Plan submitted ({request_id}). Wait for Lead's decision."


def _run_teammate_tool(name: str, block, handlers: dict) -> str:
    """队友执行单个工具调用，先过计划门禁和权限检查再真正运行。

    Args:
        name: 队友名字，用于查询 plan 门禁状态。
        block: 工具调用块，含 name（工具名）和 input（参数字典）。
        handlers: 工具名 -> 处理函数的映射表。

    Returns:
        工具的输出文本；计划未获批、权限被拒、工具不存在或执行抛异常时，
        返回对应的错误说明字符串。
    """
    gate = plan_gates.get(name, "not_required")
    if block.name in {"bash", "write_file", "edit_file"}:
        if gate != "approved":
            if gate != "not_required":
                return (f"Blocked: plan status is {gate}. Submit or revise the "
                        "plan and wait for approval before changing the workspace.")
        blocked = check_permission(block, prompt_user=False)
        if blocked:
            return blocked
    handler = handlers.get(block.name)
    if not handler:
        return f"Unknown tool: {block.name}"
    trigger_hooks("PreToolUse", block, skip_permission=True)
    try:
        output = str(handler(**block.input))
    except Exception as exc:
        output = f"Error: {type(exc).__name__}: {exc}"
    trigger_hooks("PostToolUse", block, output)
    return output


def apply_plan_response(name: str, msg: dict) -> tuple[bool, str]:
    """只应用 Lead 对该队友当前计划请求的响应。

    校验消息来源、请求 ID、认领版本号与任务 ID 都和当前待处理请求一致，
    全部匹配才把门禁状态落地并把队友恢复为 working。

    Args:
        name: 收到响应的队友名字。
        msg: 从收件箱读到的 plan_approval_response 消息。

    Returns:
        (accepted, notice) 二元组：accepted 表示响应有效并已应用；
        notice 是交给模型的提示文本，响应被忽略时为忽略原因。
    """
    metadata = msg.get("metadata", {})
    request_id = metadata.get("request_id", "")
    work_version, task_id = current_work_identity(name)
    with team_lock:
        state = pending_requests.get(request_id)
        expected_id = plan_request_ids.get(name)
        valid = (
            msg.get("from") == "lead"
            and msg.get("to") == name
            and request_id == expected_id
            and state is not None
            and state.type == "plan_approval"
            and state.sender == name
            and state.target == "lead"
            and state.work_version == work_version
            and state.task_id == task_id
            and state.status in {"approved", "rejected"}
            and metadata.get("approve", False)
            == (state.status == "approved")
        )
        if not valid:
            return False, "[Ignored plan response: request mismatch]"
        plan_gates[name] = state.status
        active_teammates[name] = "working"
        plan_request_ids.pop(name, None)
        outcome = state.status
    return True, f"[Plan {outcome}] {msg['content']}"


def apply_shutdown_request(name: str, msg: dict) -> tuple[bool, str]:
    """只接受 Lead 发给该队友且尚未处理的关机请求。

    Args:
        name: 收到请求的队友名字。
        msg: 从收件箱读到的 shutdown 请求消息，metadata 中需带 request_id。

    Returns:
        (accepted, request_id) 二元组：请求有效并已置为 stopping 时
        accepted 为 True，request_id 为该请求 ID；被忽略时 accepted
        为 False，request_id 位置是忽略原因文本。
    """
    request_id = msg.get("metadata", {}).get("request_id", "")
    with team_lock:
        state = pending_requests.get(request_id)
        valid = (
            msg.get("from") == "lead"
            and msg.get("to") == name
            and state is not None
            and state.type == "shutdown"
            and state.sender == "lead"
            and state.target == name
            and state.status == "pending"
            and active_teammates.get(name) != "stopping"
        )
        if not valid:
            return False, "[Ignored shutdown request: request mismatch]"
        active_teammates[name] = "stopping"
    return True, request_id


def _teammate_send_message(from_name: str, to: str, content: str) -> str:
    """队友向其他成员发送消息，目标不存在时拒绝发送。

    Args:
        from_name: 发送方队友名字。
        to: 接收方名字，可以是 lead 或任一在线队友。
        content: 消息正文。

    Returns:
        发送成功时的确认文本；目标不是 lead 也不在线时返回错误说明。
    """
    with team_lock:
        if to != "lead" and to not in active_teammates:
            return f"Agent '{to}' is not active"
    BUS.send(from_name, to, content)
    return f"Sent to {to}"

# IDLE task
# -- 空闲任务发现 --

# 空闲队友扫描任务队列的时间间隔（秒），避免忙等
IDLE_SCAN_INTERVAL = 2.0


def scan_unclaimed_tasks() -> list[Task]:
    """返回可以开工、且可选 worktree 绑定仍然可用的任务。

    逐项过滤：任务必须是 pending 且无人认领，blockedBy
    的依赖全部完成，可选绑定的 worktree 能正常定位。

    Returns:
        满足全部就绪条件的任务列表，可能为空列表。
    """
    with task_lock:
        ready = []
        for task in list_tasks():
            if (task.status != "pending" or task.owner is not None
                    or not can_start(task.id)):
                continue
            _, error = task_worktree_cwd(task)
            if not error:
                ready.append(task)
        return ready


def claim_next_task(name: str) -> Task | None:
    """认领第一个仍然可领的任务；手里已有任务时不再认领。

    先确认自己名下没有进行中的任务，再按就绪顺序逐个尝试认领，
    竞态中被别人抢先的任务自动跳过。

    Args:
        name: 队友名字，作为认领任务的 owner 登记。

    Returns:
        成功认领的任务对象；已有任务在身或没有可领任务时返回 None。
    """
    with task_lock:
        if teammate_assignments.get(name) or _owner_in_progress(name):
            return None
    for task in scan_unclaimed_tasks():
        result = claim_task(task.id, owner=name)
        if result.startswith("Claimed "):
            return load_task(task.id)
    return None


# -- Teammate Runtime --
# -- 队友运行时 --


class TeammateRuntime:
    """一个持久化队友：消息独立，分 WORK 和 IDLE 两种阶段。"""

    def __init__(self, name: str, role: str, prompt: str,
                 task_id: str | None, require_plan: bool):
        """初始化队友的系统提示词、首条消息与工具表。

        Args:
            name: 队友名字，同时用作消息收发和认领登记的键。
            role: 队友角色描述，会拼进系统提示词。
            prompt: spawn 时下达的初始指令，作为首条 user 消息。
            task_id: 开局即认领的任务 ID；为 None 时先进入 IDLE 自主找活。
            require_plan: True 表示改文件或跑 bash 前必须先提交计划等批准。
        """
        self.name = name
        # 系统提示词：向模型交代角色、协作规则与工作纪律（发给模型的提示词，保留英文）
        self.system = (
            f"You are '{name}', a {role}. Use tools to complete the assigned "
            "Task, then call complete_task and report a concise result. "
            "If the first user message contains [Assigned task], that Task is "
            "already claimed; do not call claim_task for it again. "
            "When asked for a plan, call submit_plan and wait for approval "
            "before bash or file changes. File and shell tools use the Task's "
            "working directory; that directory is not a sandbox. The runtime "
            "delivers your final text to Lead. Use send_message only for "
            "intermediate coordination, and address the coordinator as 'lead'."
        )
        # 首条 user 消息保存初始指令，开局任务详情也追加在这条消息上
        self.messages = [{"role": "user", "content": prompt}]
        # 开局直接派了任务：把任务详情和工作目录补进首条消息，无需再 claim
        if task_id:
            task = load_task(task_id)
            cwd = assignment_cwd(name)
            self.messages[0]["content"] += (
                f"\n\n[Assigned task {task.id}] {task.subject}\n"
                f"{task.description}\nWork directory: {cwd}"
            )
        # 要求先出计划：把计划门禁规则写进首条消息，模型动手前会先 submit_plan
        if require_plan:
            self.messages[0]["content"] += (
                "\n\n[Plan required] Submit a plan and wait for Lead approval "
                "before changing files or using bash."
            )
        # 工具名 -> 处理函数映射；收发消息与提交计划绑定本队友身份
        self.handlers = {
            "bash": self.bash,
            "read_file": self.read,
            "write_file": self.write,
            "edit_file": self.edit,
            "glob": self.glob,
            "send_message": lambda to, content: _teammate_send_message(
                name, to, content),
            "submit_plan": lambda plan: _teammate_submit_plan(name, plan),
            "list_tasks": run_list_tasks,
            "claim_task": self.claim,
            "complete_task": self.complete,
        }

    def current_cwd(self) -> tuple[Path | None, str | None]:
        """解析当前认领任务的工作目录，供所有文件系统工具复用。

        Returns:
            (cwd, error) 二元组：成功时 error 为 None；未认领任务或
            认领记录非法时 cwd 为 None，error 为交给模型的错误提示。
        """
        # 没有认领记录时文件系统工具无从定位工作目录，直接拒绝
        if self.name not in teammate_assignments:
            return None, "Error: Claim a Task before using workspace tools."
        try:
            return assignment_cwd(self.name), None
        except (FileNotFoundError, ValueError) as exc:
            # 目录消失或与登记不一致，都视为非法认领
            return None, f"Error: Invalid task assignment: {exc}"

    def bash(self, command: str) -> str:
        cwd, error = self.current_cwd()
        return error or run_bash(command, cwd=cwd)

    def read(self, path: str, limit: int | None = None) -> str:
        cwd, error = self.current_cwd()
        return error or run_read(path, limit=limit, cwd=cwd)

    def write(self, path: str, content: str) -> str:
        cwd, error = self.current_cwd()
        return error or run_write(path, content, cwd=cwd)

    def edit(self, path: str, old_text: str, new_text: str) -> str:
        cwd, error = self.current_cwd()
        return error or run_edit(path, old_text, new_text, cwd=cwd)

    def glob(self, pattern: str) -> str:
        cwd, error = self.current_cwd()
        return error or run_glob(pattern, cwd=cwd)

    def claim(self, task_id: str) -> str:
        try:
            return claim_task(task_id, owner=self.name)
        except ValueError as exc:
            return f"Error: {exc}"
        except FileNotFoundError:
            return f"Error: Task {task_id} not found"

    def complete(self, task_id: str) -> str:
        try:
            return complete_task(task_id, owner=self.name)
        except ValueError as exc:
            return f"Error: {exc}"
        except FileNotFoundError:
            return f"Error: Task {task_id} not found"

    def handle_inbox(self, inbox: list[dict]) -> bool:
        """把工作消息追加进对话；收到有效关机请求时返回 True。

        关机请求交给 apply_shutdown_request 校验，通过后回执确认并直接返回；
        计划响应和计划要求转成提示文本，普通消息加上来源前缀，
        最后把所有待处理文本合并成一条 user 消息挂到对话末尾。

        Args:
            inbox: 从消息总线读到的本队友消息列表。

        Returns:
            收到有效关机请求并已确认时返回 True；否则返回 False。
        """
        work_messages = []
        for msg in inbox:
            msg_type = msg.get("type", "message")
            if msg_type == "shutdown_request":
                accepted, notice = apply_shutdown_request(self.name, msg)
                if not accepted:
                    work_messages.append(notice)
                    continue
                BUS.send(self.name, "lead", "Shutdown acknowledged.",
                         "shutdown_response",
                         {"request_id": notice, "approve": True})
                return True
            if msg_type == "plan_approval_response":
                _, notice = apply_plan_response(self.name, msg)
                work_messages.append(notice)
                continue
            if msg_type == "plan_request":
                work_messages.append(f"[Plan required] {msg['content']}")
                continue
            work_messages.append(
                f"[Message from {msg['from']}] {msg['content']}"
            )
        if work_messages:
            self.messages.append({"role": "user",
                                  "content": "\n".join(work_messages)})
        return False

    def work(self) -> str:
        """执行一轮模型调用。返回 continue、idle 或 stop。

        先消费收件箱，再调用一次模型；有工具调用就逐个执行，
        把结果作为 user 消息追加后返回 continue；没有工具调用则把最终
        文本汇报给 Lead，并按计划门禁状态转入 waiting_approval 或释放
        任务转入 idle。

        Returns:
            "continue" 表示还有后续轮次；"idle" 表示本轮工作已收尾；
            "stop" 表示收到关机请求或模型调用失败，线程应退出。
        """
        if self.handle_inbox(BUS.read_inbox(self.name)):
            return "stop"
        with team_lock:
            active_teammates[self.name] = "working"
        try:
            response = client.messages.create(
                model=MODEL,
                system=self.system,
                messages=self.messages,
                tools=TEAMMATE_TOOLS,
                max_tokens=8000,
            )
        except Exception as exc:
            BUS.send(self.name, "lead",
                     f"{type(exc).__name__}: {exc}", "error")
            return "stop"

        self.messages.append({"role": "assistant",
                              "content": response.content})
        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]
        if tool_calls:
            results = []
            for block in tool_calls:
                output = _run_teammate_tool(
                    self.name, block, self.handlers
                )
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": output})
            self.messages.append({"role": "user", "content": results})
            return "continue"

        summary = _last_assistant_text(response.content)
        gate = plan_gates.get(self.name, "not_required")
        if gate != "pending" and summary:
            BUS.send(self.name, "lead", summary, "result")
        if gate == "pending":
            with team_lock:
                active_teammates[self.name] = "waiting_approval"
        else:
            release_completed_assignment(self.name)
            with team_lock:
                active_teammates[self.name] = "idle"
            BUS.send(self.name, "lead", "Waiting for more work.",
                     "idle_notification")
        return "idle"

    def wait_for_work(self) -> bool:
        """等待新消息，或原子地认领下一个就绪任务。

        返回值表示是否有新工作：有新工作消息或成功自动认领任务时
        返回 True，继续 WORK 阶段；收到关机请求时返回 False。

        Returns:
            True 表示应回到 WORK 继续干活；False 表示线程应收尾退出。
        """
        while True:
            inbox = BUS.wait_for_messages(self.name, IDLE_SCAN_INTERVAL)
            if inbox:
                before = len(self.messages)
                if self.handle_inbox(inbox):
                    return False
                if len(self.messages) > before:
                    return True
                continue

            task = claim_next_task(self.name)
            if not task:
                continue
            cwd = assignment_cwd(self.name)
            self.messages.append({
                "role": "user",
                "content": (
                    f"[Auto-claimed task {task.id}] {task.subject}\n"
                    f"{task.description}\nWork directory: {cwd}"
                ),
            })
            print(f"  [idle] {self.name} claimed {task.id}: {task.subject}")
            return True

    def run(self):
        """队友线程的主循环：在 WORK 与 IDLE 之间流转直到退出。

        循环执行 work()；返回 idle 时调用 wait_for_work() 等活，
        等不到（关机）就退出。任何异常都上报 Lead；收尾时统一释放
        任务认领，并从各登记表里注销自己。
        """
        try:
            state = "continue"
            while state != "stop":
                if state == "idle" and not self.wait_for_work():
                    break
                state = self.work()
        except Exception as exc:
            try:
                BUS.send(self.name, "lead",
                         f"{type(exc).__name__}: {exc}", "error")
            except Exception:
                pass
        finally:
            try:
                release_teammate_assignment(self.name)
            except Exception as exc:
                try:
                    BUS.send(
                        self.name, "lead",
                        f"Assignment cleanup failed: {type(exc).__name__}: {exc}",
                        "error",
                    )
                except Exception:
                    pass
            with team_lock:
                active_teammates.pop(self.name, None)
                plan_gates.pop(self.name, None)
                plan_request_ids.pop(self.name, None)
                teammate_threads.pop(self.name, None)
            print(f"  [teammate] {self.name} finished")

 # 队友名字 -> 线程对象，便于运行时定位与排查
teammate_threads: dict[str, threading.Thread] = {}      


def spawn_teammate_thread(name: str, role: str, prompt: str,
                          task_id: str | None = None,
                          require_plan: bool = False) -> str:
    """先认领初始任务，再启动一个持久化队友线程。

    校验名字合法性与唯一性，登记初始状态后尝试认领任务；认领失败
    会回滚登记并返回错误，成功才创建 TeammateRuntime 并启动守护线程。

    Args:
        name: 队友名字，需通过合法性校验且与在线队友不重名（忽略大小写）。
        role: 队友角色描述，会拼进系统提示词。
        prompt: 开局指令，作为队友的首条 user 消息。
        task_id: 开局要认领的任务 ID；为 None 时不带初始任务。
        require_plan: True 表示该队友改文件或跑 bash 前必须先提交计划。

    Returns:
        给 Lead 的结果说明：名字非法、重名或认领失败时为错误提示；
        成功时为确认文本，并提示 Lead 结束本轮等待事件送达。
    """
    if not is_valid_agent_name(name):
        return ("Invalid teammate name: use 1-64 letters, digits, "
                "underscores, or dashes")
    if name.lower() in RESERVED_TEAMMATE_NAMES:
        return f"Invalid teammate name: '{name}' is reserved by the runtime"
    with team_lock:
        if any(existing.casefold() == name.casefold()
               for existing in active_teammates):
            return f"Teammate '{name}' already exists"
        active_teammates[name] = "working"
        plan_gates[name] = "required" if require_plan else "not_required"
        assignment_versions[name] = 0

    if task_id:
        try:
            claimed = claim_task(task_id, owner=name)
        except (FileNotFoundError, ValueError) as exc:
            claimed = f"Error: {exc}"
        if not claimed.startswith("Claimed "):
            with team_lock:
                active_teammates.pop(name, None)
                plan_gates.pop(name, None)
                assignment_versions.pop(name, None)
            return f"Cannot spawn teammate '{name}': {claimed}"

    runtime = TeammateRuntime(name, role, prompt, task_id, require_plan)
    thread = threading.Thread(target=runtime.run, daemon=True)
    with team_lock:
        teammate_threads[name] = thread
    thread.start()
    print(f"  [teammate] {name} spawned as {role}")
    assigned = f" for {task_id}" if task_id else " without an initial Task"
    return (
        f"Teammate '{name}' spawned as {role}{assigned}. "
        "End this turn; the runtime will deliver its events."
    )


# -- Lead Team Tools --
# -- Lead 的团队工具 --

def run_spawn_teammate(name: str, role: str, prompt: str,
                       task_id: str | None = None,
                       require_plan: bool = False) -> str:
    """Lead 工具：spawn 一个持久化队友，参数原样转发给 spawn_teammate_thread。"""
    return spawn_teammate_thread(name, role, prompt, task_id, require_plan)


def run_list_teammates() -> str:
    """Lead 工具：列出全部在线队友及其当前状态。"""
    with team_lock:
        if not active_teammates:
            return "No active teammates."
        return "\n".join(
            f"{name}: {status}"
            for name, status in sorted(active_teammates.items())
        )


def run_send_message(to: str, content: str) -> str:
    """Lead 工具：以 lead 身份向指定队友发送消息。"""
    if to not in active_teammates:
        return f"Teammate '{to}' is not active"
    BUS.send("lead", to, content)
    return f"Sent to {to}"


def run_request_shutdown(teammate: str) -> str:
    """Lead 工具：登记关机协议请求并发给队友，等它确认后退出。"""
    if teammate not in active_teammates:
        return f"Teammate '{teammate}' is not active"
    with team_lock:
        request_id = new_request_id()
        pending_requests[request_id] = ProtocolState(
            request_id=request_id,
            type="shutdown",
            sender="lead",
            target=teammate,
            status="pending",
            payload="",
        )
    BUS.send("lead", teammate, "Finish the current step and shut down.",
             "shutdown_request", {"request_id": request_id})
    return f"Shutdown requested from {teammate} ({request_id})"


def run_request_plan(teammate: str, task: str) -> str:
    """Lead 工具：要求队友先提交计划，把门禁置为 required 并下发任务说明。"""
    if teammate not in active_teammates:
        return f"Teammate '{teammate}' is not active"
    with team_lock:
        plan_gates[teammate] = "required"
    BUS.send("lead", teammate, task, "plan_request")
    return f"Plan requested from {teammate}"


def run_review_plan(request_id: str, approve: bool,
                    feedback: str = "") -> str:
    """Lead 工具：审批队友的待审计划，把决定与反馈发回给队友。"""
    state = pending_requests.get(request_id)
    if not state:
        return f"Request {request_id} not found"
    work_version, task_id = current_work_identity(state.sender)
    with team_lock:
        state = pending_requests.get(request_id)
        if not state:
            return f"Request {request_id} not found"
        if state.type != "plan_approval":
            return f"Request {request_id} is not a plan"
        if state.status != "pending":
            return f"Request {request_id} already {state.status}"
        if (state.work_version != work_version or state.task_id != task_id):
            return f"Request {request_id} belongs to an earlier assignment"
        if plan_request_ids.get(state.sender) != request_id:
            return f"Request {request_id} is not the current plan"
        state.status = "approved" if approve else "rejected"
    content = feedback or ("Plan approved." if approve
                           else "Revise the plan and submit it again.")
    BUS.send("lead", state.sender, content, "plan_approval_response",
             {"request_id": request_id, "approve": approve})
    return f"Plan {state.status} ({request_id})"


def run_create_worktree(name: str, task_id: str) -> str:
    """Lead 工具：为任务创建隔离 worktree，参数原样转发给 create_worktree。"""
    return create_worktree(name, task_id)


# -- 工具定义 --

BASE_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object",
                      "properties": {"command": {"type": "string"}},
                      "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "limit": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text once.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "old_text": {"type": "string"},
                                     "new_text": {"type": "string"}},
                      "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files by glob pattern; ** matches recursively.",
     "input_schema": {"type": "object",
                      "properties": {"pattern": {"type": "string"}},
                      "required": ["pattern"]}},
]

TASK_TOOLS = [
    {"name": "create_task",
     "description": "Create a task and return its runtime-generated ID.",
     "input_schema": {"type": "object",
                      "properties": {
                          "subject": {"type": "string"},
                          "description": {"type": "string"}},
                      "required": ["subject"],
                      "additionalProperties": False}},
    {"name": "update_task",
     "description": "Add dependencies using IDs returned by create_task.",
     "input_schema": {"type": "object",
                      "properties": {
                          "task_id": {"type": "string",
                                      "pattern": "^task_[0-9a-f]{8}$"},
                          "addBlockedBy": {
                              "type": "array",
                              "items": {"type": "string",
                                        "pattern": "^task_[0-9a-f]{8}$"},
                              "minItems": 1}},
                      "required": ["task_id", "addBlockedBy"],
                      "additionalProperties": False}},
    {"name": "list_tasks", "description": "List shared tasks.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_task", "description": "Get one task by ID.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "claim_task", "description": "Claim a ready task.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "complete_task", "description": "Complete an owned task.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
]

TEAMMATE_TOOLS = [
    *BASE_TOOLS,
    {"name": "send_message",
     "description": "Send an intermediate message to 'lead' or an active teammate.",
     "input_schema": {"type": "object",
                      "properties": {"to": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["to", "content"]}},
    {"name": "submit_plan",
     "description": "Submit a work plan for Lead approval.",
     "input_schema": {"type": "object",
                      "properties": {"plan": {"type": "string"}},
                      "required": ["plan"]}},
    next(tool for tool in TASK_TOOLS if tool["name"] == "list_tasks"),
    next(tool for tool in TASK_TOOLS if tool["name"] == "claim_task"),
    next(tool for tool in TASK_TOOLS if tool["name"] == "complete_task"),
]

TEAM_TOOLS = [
    {"name": "spawn_teammate",
     "description": "Spawn a persistent teammate.",
     "input_schema": {"type": "object",
                      "properties": {
                          "name": {"type": "string",
                                   "pattern": "^[A-Za-z0-9_-]{1,64}$"},
                          "role": {"type": "string"},
                          "prompt": {"type": "string"},
                          "task_id": {"type": "string",
                                      "pattern": "^task_[0-9a-f]{8}$"},
                          "require_plan": {"type": "boolean"}},
                      "required": ["name", "role", "prompt"]}},
    {"name": "list_teammates", "description": "List active teammates.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "send_message", "description": "Message a teammate.",
     "input_schema": {"type": "object",
                      "properties": {"to": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["to", "content"]}},
    {"name": "request_shutdown",
     "description": "Ask a teammate to shut down.",
     "input_schema": {"type": "object",
                      "properties": {"teammate": {"type": "string"}},
                      "required": ["teammate"]}},
    {"name": "request_plan",
     "description": "Require a teammate plan before workspace changes.",
     "input_schema": {"type": "object",
                      "properties": {"teammate": {"type": "string"},
                                     "task": {"type": "string"}},
                      "required": ["teammate", "task"]}},
    {"name": "review_plan", "description": "Approve or reject a plan.",
     "input_schema": {"type": "object",
                      "properties": {
                          "request_id": {"type": "string"},
                          "approve": {"type": "boolean"},
                          "feedback": {"type": "string"}},
                      "required": ["request_id", "approve"]}},
    {"name": "create_worktree",
     "description": "Create and bind a task worktree.",
     "input_schema": {
         "type": "object",
         "properties": {
             "name": {"type": "string",
                      "pattern": "^(?!.*\\.\\.)[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
                      "maxLength": 64},
             "task_id": {"type": "string"}},
         "required": ["name", "task_id"],
         "additionalProperties": False}},
]

TOOLS = [*BASE_TOOLS, *TASK_TOOLS, *TEAM_TOOLS]

TOOL_HANDLERS = {
    "bash": run_agent_bash,
    "read_file": run_agent_read,
    "write_file": run_agent_write,
    "edit_file": run_agent_edit,
    "glob": run_agent_glob,
    "create_task": run_create_task,
    "update_task": run_update_task,
    "list_tasks": run_list_tasks,
    "get_task": run_get_task,
    "claim_task": run_claim_task,
    "complete_task": run_complete_task,
    "spawn_teammate": run_spawn_teammate,
    "list_teammates": run_list_teammates,
    "send_message": run_send_message,
    "request_shutdown": run_request_shutdown,
    "request_plan": run_request_plan,
    "review_plan": run_review_plan,
    "create_worktree": run_create_worktree,
}


# -- 钩子与权限检查 --

HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]


def contains_destructive_command(command: str) -> bool:
    return bool(DESTRUCTIVE_COMMAND_WORD.search(command))


def register_hook(event: str, callback):
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args, skip_permission: bool = False):
    for callback in HOOKS[event]:
        if skip_permission and callback is permission_hook:
            continue
        result = callback(*args)
        if result is not None:
            return result
    return None


def check_permission(block, prompt_user: bool = True) -> str | None:
    if block.name == "bash":
        command = block.input.get("command", "")
        for pattern in DENY_LIST:
            if pattern in command:
                return f"Permission denied by deny list: {pattern}"
        if contains_destructive_command(command) or any(
            keyword in command for keyword in DESTRUCTIVE
        ):
            if not prompt_user:
                return "Permission required: ask Lead to run this command."
            print(f"\n[permission] {block.name}({block.input})")
            if input("Allow? [y/N] ").strip().lower() not in {"y", "yes"}:
                return "Permission denied by user"

    if block.name in {"read_file", "write_file", "edit_file"}:
        raw_path = block.input.get("path", "")
        if not (WORKDIR / raw_path).resolve().is_relative_to(WORKDIR.resolve()):
            if not prompt_user:
                return "Permission required: path is outside the workspace."
            print(f"\n[permission] {block.name}({block.input})")
            if input("Allow? [y/N] ").strip().lower() not in {"y", "yes"}:
                return "Permission denied by user"
    return None


def permission_hook(block):
    return check_permission(block, prompt_user=True)


def log_hook(block):
    preview = str(list(block.input.values())[:2])[:60]
    print(f"[hook] {block.name}({preview})")
    return None


def large_output_hook(block, output):
    if len(str(output)) > 100000:
        print(f"[hook] Large output from {block.name}: {len(str(output))} chars")
    return None


def context_hook(query: str):
    print(f"[hook] UserPromptSubmit: working in {WORKDIR}")
    return None


def summary_hook(messages: list):
    tool_count = sum(
        1
        for message in messages
        for block in (
            message.get("content")
            if isinstance(message.get("content"), list)
            else []
        )
        if isinstance(block, dict) and block.get("type") == "tool_result"
    )
    print(f"[hook] Stop: session used {tool_count} tool calls")
    return None


register_hook("UserPromptSubmit", context_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


def execute_tool(block) -> str:
    blocked = trigger_hooks("PreToolUse", block)
    if blocked:
        return str(blocked)
    handler = TOOL_HANDLERS.get(block.name)
    if not handler:
        return f"Unknown tool: {block.name}"
    try:
        output = str(handler(**block.input))
    except Exception as exc:
        output = f"Error: {type(exc).__name__}: {exc}"
    trigger_hooks("PostToolUse", block, output)
    return output


# -- Agent Loop --

def agent_loop(messages: list):
    while True:
        try:
            response = client.messages.create(
                model=MODEL,
                system=SYSTEM,
                messages=messages,
                tools=TOOLS,
                max_tokens=8000,
            )
        except Exception as exc:
            messages.append({
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": f"[Error] {type(exc).__name__}: {exc}",
                }],
            })
            release_completed_assignment("agent")
            trigger_hooks("Stop", messages)
            return

        messages.append({"role": "assistant", "content": response.content})
        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]
        if not tool_calls:
            release_completed_assignment("agent")
            trigger_hooks("Stop", messages)
            return

        results = []
        for block in tool_calls:
            print(f"> {block.name}")
            output = execute_tool(block)
            print(output[:300])
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })
        messages.append({"role": "user", "content": results})


def print_last_assistant_message(history: list):
    if not history:
        return
    for block in history[-1].get("content", []):
        if getattr(block, "type", None) == "text":
            print(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            print(block.get("text", ""))


def wait_for_cli_event() -> tuple[str, str | None]:
    prompt_visible = False
    while True:
        if BUS.peek("lead"):
            if prompt_visible:
                print()
            return "wake", None
        if not prompt_visible:
            print("s13 >> ", end="", flush=True)
            prompt_visible = True
        readable, _, _ = select.select([sys.stdin], [], [], 0.25)
        if readable:
            line = sys.stdin.readline()
            if line == "":
                return "quit", None
            return "user", line.rstrip("\n")


if __name__ == "__main__":
    print("s13: agent teams")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    history = []
    had_teammates = False

    while True:
        kind, payload = wait_for_cli_event()
        if kind == "quit":
            break
        if kind == "user":
            if payload is None or payload.strip().lower() in {"q", "exit", ""}:
                break
            trigger_hooks("UserPromptSubmit", payload)
            history.append({"role": "user", "content": payload})
        else:
            inbox = consume_lead_inbox()
            if not inbox:
                continue
            history.append({
                "role": "user",
                "content": format_team_events(inbox),
            })
            print(f"[wake: {len(inbox)} team event(s) -> new turn]")

        agent_loop(history)
        print_last_assistant_message(history)

        if active_teammates:
            had_teammates = True
        elif had_teammates and not BUS.peek("lead"):
            print("[all teammates shut down]")
            had_teammates = False
        print()
