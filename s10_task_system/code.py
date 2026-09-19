#!/usr/bin/env python3
"""
s10_task_system.py - 任务系统

    .tasks/
      task_a1b2c3d4.json  {status: completed, blockedBy: []}
      task_e5f6a7b8.json  {status: pending, blockedBy: [task_a1b2c3d4]}
      task_11223344.json  {status: pending, blockedBy: [task_e5f6a7b8]}

    依赖关系图：

    +-----------+      +-----------+      +-----------+
    | schema    | ---> | API       | ---> | tests     |
    | completed |      | pending   |      | pending   |
    +-----------+      +-----------+      +-----------+

    can_start(API) 返回 true，因为 schema 已完成。

    任务生命周期：

    pending --claim_task--> in_progress --complete_task--> completed
"""

import glob
import json
import os
import re
import secrets
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import readline

    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
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

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Use task tools to track dependencies and progress. Create all task nodes "
    "first. After create_task returns runtime-generated IDs, use update_task "
    "with those exact IDs to add dependencies."
)


# -- s10 新增：持久化任务记录 --

# 任务存储目录：工作区的 .tasks/ 下
TASKS_DIR = WORKDIR / ".tasks"
# 任务 ID 格式白名单：task_ 前缀 + 8 位小写十六进制，从源头防止路径注入
TASK_ID_PATTERN = re.compile(r"^task_[0-9a-f]{8}$")


@dataclass
class Task:
    """单条任务的内存表示，字段与 .tasks/ 下的 JSON 文件一一对应。

    Attributes:
        id:          全局唯一任务 ID，格式为 task_ + 8 位十六进制。
        subject:     任务标题（一句话摘要）。
        description: 任务的详细说明。
        status:      生命周期状态，取值 pending / in_progress / completed。
        owner:       认领该任务的 agent 名称，未被认领时为 None。
        blockedBy:   依赖的任务 ID 列表，全部完成后本任务才可认领。
    """
    id: str
    subject: str
    description: str
    status: str
    owner: str | None
    blockedBy: list[str]


class TaskStore:
    """任务存储层：负责任务 JSON 文件的创建、读取、校验与依赖管理。"""

    def __init__(self, directory: Path):
        """初始化存储实例。"""

        # 任务 JSON 文件所在的目录路径。
        self.directory = directory

    def _root(self, create: bool = False) -> Path:
        """解析并校验任务存储根目录，确保它不会逃逸出工作区。

        Args:
            create: 为 True 时若目录不存在则递归创建。

        Returns:
            解析后的存储根目录绝对路径。

        Raises:
            ValueError: 存储目录位于工作区之外时抛出。
        """
        # 需要时先创建目录（含父目录），已存在则跳过
        if create:
            self.directory.mkdir(parents=True, exist_ok=True)
        # resolve() 消除 .. 与符号链接等相对成分，得到真实绝对路径
        root = self.directory.resolve()
        # 安全边界：目录必须位于工作区内，防止配置指向外部路径
        if not root.is_relative_to(WORKDIR.resolve()):
            raise ValueError("Task store escapes the workspace")
        return root

    def _path(self, task_id: str, create_root: bool = False) -> Path:
        """把任务 ID 映射为存储文件路径，并做严格格式与越界校验。

        Args:
            task_id: 待映射的任务 ID。
            create_root: 为 True 时允许在映射前创建存储根目录。

        Returns:
            该任务对应的 JSON 文件绝对路径。

        Raises:
            ValueError: 任务 ID 不符合 task_ + 8 位十六进制格式，
                或拼出的路径逃逸出存储根目录时抛出。
        """
        # 先用白名单正则限制 ID 字符集，杜绝 ../ 等路径注入
        if not isinstance(task_id, str) or not TASK_ID_PATTERN.fullmatch(task_id):
            raise ValueError(f"Invalid task ID: {task_id!r}")
        root = self._root(create=create_root)
        # 二次防御：resolve 后确认文件路径仍落在根目录之内
        path = (root / f"{task_id}.json").resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Invalid task ID: {task_id!r}")
        return path

    def exists(self, task_id: str) -> bool:
        """判断指定任务 ID 的存储文件是否存在。

        Args:
            task_id: 待检查的任务 ID。

        Returns:
            文件存在且确实是普通文件时返回 True。
        """
        return self._path(task_id).is_file()

    def create(self, subject: str, description: str = "") -> Task:
        """创建一条 pending 状态的新任务并持久化到磁盘。

        Args:
            subject: 任务标题，去除首尾空白后不能为空。
            description: 任务的详细说明，可为空。

        Returns:
            创建成功的任务对象（含运行时生成的 ID）。

        Raises:
            ValueError: 标题去除空白后为空字符串时抛出。
            RuntimeError: 连续 100 次随机 ID 均与已有文件冲突时抛出。
        """
        subject = subject.strip()
        if not subject:
            raise ValueError("Task subject cannot be empty")

        # 提前创建存储目录，后续写文件不再担心目录缺失
        self._root(create=True)
        # 最多重试 100 次，应对随机 ID 碰撞的极小概率情况
        for _ in range(100):
            task = Task(
                # secrets 生成 4 字节随机数，保证 ID 不可预测
                id=f"task_{secrets.token_hex(4)}",
                subject=subject,
                description=description,
                status="pending",
                owner=None,
                blockedBy=[],
            )
            try:
                # "x" 模式为独占创建：文件已存在时抛错而非覆盖
                with self._path(task.id, create_root=True).open(
                    "x", encoding="utf-8"
                ) as handle:
                    json.dump(asdict(task), handle, indent=2)
                return task

            except FileExistsError:
                # ID 碰撞则换一个重新尝试
                continue
        raise RuntimeError("Could not allocate a unique task ID")

    def _depends_on(self, task_id: str, target_id: str) -> bool:
        """返回 task_id 是否传递性地依赖 target_id。

        沿 blockedBy 边做深度优先遍历，用于在追加依赖前检测环。

        Args:
            task_id: 起始任务 ID。
            target_id: 要判断是否为其（间接）依赖的任务 ID。

        Returns:
            存在依赖路径时返回 True，否则返回 False。
        """
        # 用栈实现 DFS，pending 保存待访问的任务 ID
        pending = [task_id]
        visited = set()
        while pending:
            current = pending.pop()
            # 命中目标说明存在 task_id -> target_id 的依赖路径
            if current == target_id:
                return True
            # 已访问的节点跳过，避免环导致死循环
            if current in visited:
                continue
            visited.add(current)
            # 把当前任务的全部依赖压栈继续搜索
            pending.extend(self.load(current).blockedBy)
        return False

    def update_dependencies(self, task_id: str,
                            add_blocked_by: list[str]) -> Task:
        """给 pending 且未被认领的任务追加依赖（blockedBy）。

        Args:
            task_id: 目标任务 ID。
            add_blocked_by: 要追加的依赖任务 ID 列表。

        Returns:
            更新后的任务对象。

        Raises:
            ValueError: add_blocked_by 不是列表、目标任务已被认领或
                不处于 pending 状态、依赖不存在、出现自依赖或
                依赖环时抛出。
        """
        # 类型防御：依赖列表必须是 list
        if not isinstance(add_blocked_by, list):
            raise ValueError("addBlockedBy must be a list of task IDs")

        task = self.load(task_id)
        # 依赖关系只在任务尚未开工（pending 且无 owner）时允许调整
        if task.status != "pending" or task.owner is not None:
            raise ValueError(
                f"Task {task_id} dependencies can only be updated while "
                "pending and unowned"
            )

        # dict.fromkeys 去重且保持原有顺序
        dependencies = list(dict.fromkeys(add_blocked_by))
        for dependency in dependencies:
            # 禁止自己依赖自己
            if dependency == task_id:
                raise ValueError("Task cannot depend on itself")
            # 依赖的任务必须真实存在
            if not self.exists(dependency):
                raise ValueError(f"Dependency not found: {dependency}")
            # 若依赖方（直接或间接）依赖本任务，追加会形成环，拒绝
            # not in task.blockedBy的细节：如果依赖已经是直接依赖，说明它上次加入时已通过全部校验（不可能成环）
            if dependency not in task.blockedBy and self._depends_on(
                dependency, task_id
            ):
                raise ValueError(
                    f"Dependency cycle detected: {task_id} -> {dependency}"
                )

        # 只追加尚未存在的依赖，避免重复条目
        task.blockedBy.extend(
            dependency for dependency in dependencies
            if dependency not in task.blockedBy
        )
        self.save(task)
        return task

    def save(self, task: Task) -> None:
        """把任务对象序列化后整体覆写回对应的 JSON 文件。

        Args:
            task: 要持久化的任务对象。
        """
        # asdict 将 dataclass 转成可 JSON 序列化的字典
        self._path(task.id, create_root=True).write_text(
            json.dumps(asdict(task), indent=2),
            encoding="utf-8",
        )

    def load(self, task_id: str) -> Task:
        """从磁盘读取任务文件并做完整性与状态校验。

        Args:
            task_id: 要读取的任务 ID。

        Returns:
            校验通过的任务对象。

        Raises:
            ValueError: 文件内的 id 与请求的 ID 不一致，或 status
                不是三种合法生命周期状态之一时抛出。
        """
        data = json.loads(self._path(task_id).read_text(encoding="utf-8"))
        task = Task(**data)
        # 防止文件被改名后错位：文件名与文件内 id 必须一致
        if task.id != task_id:
            raise ValueError(f"Task file ID does not match {task_id}")
        # 状态字段只允许三个合法值，防止被手工改成垃圾数据
        if task.status not in ("pending", "in_progress", "completed"):
            raise ValueError(f"Invalid task status: {task.status}")
        return task

    def list(self) -> list[Task]:
        """列出存储目录中的全部任务，按文件名排序保证顺序稳定。

        Returns:
            全部任务对象组成的列表，目录不存在时返回空列表。
        """
        # 目录尚未创建说明还没有任何任务
        if not self.directory.exists():
            return []
        root = self._root()
        # path.stem 去掉 .json 后缀即任务 ID
        return [self.load(path.stem)
                for path in sorted(root.glob("task_*.json"))]


# 模块级共享的任务存储实例，供下方各工具函数直接使用
TASKS = TaskStore(TASKS_DIR)


def create_task(subject: str, description: str = "") -> Task:
    """创建新任务（TaskStore.create 的模块级入口）。

    Args:
        subject: 任务标题。
        description: 任务详细说明，可为空。

    Returns:
        创建成功的任务对象。
    """
    return TASKS.create(subject, description)


def update_task(task_id: str, addBlockedBy: list[str]) -> Task:
    """为任务追加依赖（TaskStore.update_dependencies 的模块级入口）。

    Args:
        task_id: 目标任务 ID。
        addBlockedBy: 要追加的依赖任务 ID 列表。

    Returns:
        更新后的任务对象。
    """
    return TASKS.update_dependencies(task_id, addBlockedBy)


def load_task(task_id: str) -> Task:
    """按 ID 读取任务（TaskStore.load 的模块级入口）。

    Args:
        task_id: 任务 ID。

    Returns:
        校验通过的任务对象。
    """
    return TASKS.load(task_id)


def list_tasks() -> list[Task]:
    """列出全部任务（TaskStore.list 的模块级入口）。

    Returns:
        全部任务对象组成的列表。
    """
    return TASKS.list()


def get_task(task_id: str) -> str:
    """以格式化 JSON 字符串返回单个任务的完整信息。

    Args:
        task_id: 任务 ID。

    Returns:
        缩进美化的任务 JSON 文本，便于喂给模型查看。
    """
    return json.dumps(asdict(load_task(task_id)), indent=2)


def incomplete_dependencies(task: Task) -> list[str]:
    """统计任务尚未完成的依赖 ID 列表，用于判断能否开工。

    Args:
        task: 要检查的任务对象。

    Returns:
        状态不是 completed 的依赖 ID 列表（含读不到或损坏的依赖），
        为空表示所有依赖均已完成。
    """
    incomplete = []
    for dependency in task.blockedBy:
        try:
            # 依赖存在但未完成 -> 记为未完成
            if load_task(dependency).status != "completed":
                incomplete.append(dependency)
        except (FileNotFoundError, ValueError):
            # 依赖文件丢失或内容损坏 -> 视为未完成，任务保持阻塞
            incomplete.append(dependency)
    return incomplete

def can_start(task_id: str) -> bool:
    """判断任务是否已无未完成依赖、可以认领开工。

    Args:
        task_id: 任务 ID。

    Returns:
        全部依赖均完成时返回 True。
    """
    return not incomplete_dependencies(load_task(task_id))


def claim_task(task_id: str, owner: str = "agent") -> str:
    """认领任务：校验状态与依赖后将其置为 in_progress。

    Args:
        task_id: 要认领的任务 ID。
        owner: 认领者名称，默认为 agent。

    Returns:
        成功时返回确认文本；失败时返回原因文本（不抛异常，
        便于把结果直接喂回模型）。
    """
    task = load_task(task_id)
    # 只允许认领 pending 状态的任务
    if task.status != "pending":
        return f"Task {task_id} is {task.status}, cannot claim"
    # 存在未完成依赖时拒绝认领，并告知阻塞来源
    dependencies = incomplete_dependencies(task)
    if dependencies:
        return f"Blocked by: {dependencies}"
    # 同时记录认领者并推进生命周期状态
    task.owner = owner
    task.status = "in_progress"
    TASKS.save(task)
    print(f"  [claim] {task.subject} -> in_progress (owner: {owner})")
    return f"Claimed {task.id} ({task.subject})"


def complete_task(task_id: str, owner: str = "agent") -> str:
    """完成任务：只有当前 owner 才能把 in_progress 置为 completed。

    完成后对比前后快照，找出因本次完成而新解锁的下游任务，
    一并告知模型可以继续认领。

    Args:
        task_id: 要完成的任务 ID。
        owner: 调用方声称的认领者名称，默认为 agent。

    Returns:
        成功时返回完成信息（可能附带新解锁任务列表）；
        失败时返回原因文本（不抛异常）。
    """
    task = load_task(task_id)
    # 生命周期约束：只有 in_progress 的任务能被完成
    if task.status != "in_progress":
        return f"Task {task_id} is {task.status}, cannot complete"
    # 所有权约束：认领者本人才能完成，防止他人误操作
    if task.owner != owner:
        return f"Task {task_id} is owned by {task.owner}, not {owner}"
    # 完成前快照：ready_before里是 已处于可开工状态且在pending的带依赖任务
    ready_before = {
        candidate.id
        for candidate in list_tasks()
        if candidate.status == "pending"
        and candidate.blockedBy
        and can_start(candidate.id)
    }
    task.status = "completed"
    TASKS.save(task)
    # 完成后重扫：新变为可开工且不在快照中的即为本次解锁的
    # unblocked本质上是一条系统发给模型的调度通知：“你刚完成的事，解锁了这几个后续任务，可以去做它们了。”
    unblocked = [candidate.subject for candidate in list_tasks()
                 if candidate.status == "pending"
                 and candidate.blockedBy
                 and candidate.id not in ready_before
                 and can_start(candidate.id)]
    print(f"  [complete] {task.subject}")
    message = f"Completed {task.id} ({task.subject})"
    if unblocked:
        message += f"\nUnblocked: {', '.join(unblocked)}"
        print(f"  [unblocked] {', '.join(unblocked)}")
    return message


# -- 来自 s04：工具实现 --

def run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True, errors="replace",
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = (WORKDIR / path).resolve().read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as error:
        return f"Error: {error}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as error:
        return f"Error: {error}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as error:
        return f"Error: {error}"


def run_glob(pattern: str) -> str:
    try:
        matches = sorted({
            match
            for match in glob.glob(pattern, root_dir=WORKDIR, recursive=True)
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR)
        })
        shown = matches[:200]
        if len(matches) > 200:
            shown.append("... (more matches omitted; narrow the pattern)")
        return "\n".join(shown) if shown else "(no matches)"
    except Exception as error:
        return f"Error: {error}"


def run_create_task(subject: str, description: str = "") -> str:
    
    task = create_task(subject, description)
    # 终端 print 供人阅读；返回值才是模型看到的 tool_result
    print(f"  [create] {task.subject}")
    return f"Created {task.id}: {task.subject}"


def run_update_task(task_id: str, addBlockedBy: list[str]) -> str:
    """update_task 工具处理器：追加依赖并回显最新依赖列表。

    Args:
        task_id: 目标任务 ID。
        addBlockedBy: 要追加的依赖任务 ID 列表（工具协议用驼峰命名，
            由 update_task 翻译给存储层的 snake_case）。

    Returns:
        喂回模型的确认文本，含追加后的完整 blockedBy。
    """
    task = update_task(task_id, addBlockedBy)
    # join 空列表得到空串（falsy），or 兜底显示 (none)
    dependencies = ", ".join(task.blockedBy) or "(none)"
    print(f"  [update] {task.subject} blockedBy: {dependencies}")
    return f"Updated {task.id} blockedBy: {dependencies}"


def run_list_tasks() -> str:
    """list_tasks 工具处理器：把任务列表渲染成清单式多行文本。

    Returns:
        每行一个任务的文本（状态标记 + ID + 标题 + 状态 + 认领者
        + 依赖）；无任务时返回引导提示。
    """
    tasks = list_tasks()
    if not tasks:
        return "No tasks. Use create_task to add some."
    lines = []
    for task in tasks:
        # 状态映射为复选框符号；未知状态兜底显示 [?]
        marker = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "completed": "[x]",
        }.get(task.status, "[?]")
        # 有依赖时显示阻塞列表，无依赖则留空
        dependencies = (
            f" (blockedBy: {', '.join(task.blockedBy)})"
            if task.blockedBy else ""
        )
        # 已被认领时追加 [owner]，未认领则留空
        owner = f" [{task.owner}]" if task.owner else ""
        lines.append(
            f"{marker} {task.id}: {task.subject} "
            f"[{task.status}]{owner}{dependencies}"
        )
    return "\n".join(lines)


def run_get_task(task_id: str) -> str:
    return get_task(task_id)


def run_claim_task(task_id: str) -> str:
    return claim_task(task_id, owner="agent")


def run_complete_task(task_id: str) -> str:
    return complete_task(task_id, owner="agent")


TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern; ** matches recursively.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
    {"name": "create_task", "description": "Create a task and return its runtime-generated ID.",
     "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"], "additionalProperties": False}},
    {"name": "update_task", "description": "Add dependencies using IDs returned by create_task.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string", "pattern": "^task_[0-9a-f]{8}$"}, "addBlockedBy": {"type": "array", "items": {"type": "string", "pattern": "^task_[0-9a-f]{8}$"}, "minItems": 1}}, "required": ["task_id", "addBlockedBy"], "additionalProperties": False}},
    {"name": "list_tasks", "description": "List tasks with status, owner, and dependencies.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_task", "description": "Get a task by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}},
    {"name": "claim_task", "description": "Claim a pending task whose dependencies are complete.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}},
    {"name": "complete_task", "description": "Complete the task claimed by this agent.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}},
]

TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
    "create_task": run_create_task,
    "update_task": run_update_task,
    "list_tasks": run_list_tasks,
    "get_task": run_get_task,
    "claim_task": run_claim_task,
    "complete_task": run_complete_task,
}


# -- 来自 s04：钩子与权限检查 --

HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}


def register_hook(event: str, callback):
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]


def contains_destructive_command(command: str) -> bool:
    return bool(DESTRUCTIVE_COMMAND_WORD.search(command))


def permission_hook(block):
    if block.name == "bash":
        command = block.input.get("command", "")
        for pattern in DENY_LIST:
            if pattern in command:
                print(f"\n\033[31m[blocked] '{pattern}'\033[0m")
                return "Permission denied by deny list"
        if contains_destructive_command(command) or any(
            keyword in command for keyword in DESTRUCTIVE
        ):
            print("\n\033[33m[permission] Potentially destructive command\033[0m")
            print(f"   Tool: {block.name}({block.input})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"

    if block.name in ("read_file", "write_file", "edit_file"):
        path = block.input.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print("\n\033[33m[permission] Access outside workspace\033[0m")
            print(f"   Tool: {block.name}({block.input})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    return None


def log_hook(block):
    preview = str(list(block.input.values())[:2])[:60]
    print(f"\033[90m[HOOK] {block.name}({preview})\033[0m")
    return None


def large_output_hook(block, output):
    if len(str(output)) > 100000:
        print(
            f"\033[33m[HOOK] Large output from {block.name}: "
            f"{len(str(output))} chars\033[0m"
        )
    return None


def context_hook(query: str):
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
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
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
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
    try:
        output = handler(**block.input) if handler else f"Unknown: {block.name}"
    except Exception as error:
        output = f"Error: {error}"

    trigger_hooks("PostToolUse", block, output)
    return str(output)


# -- Agent 循环 --

def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]
        if not tool_calls:
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return

        results = []
        for block in tool_calls:
            output = execute_tool(block)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("s10: Task System - dependencies and task state")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            # \001/\002 告诉 Readline 这些 ANSI 转义符的显示宽度为零。
            query = input("\001\033[36m\002s10 >> \001\033[0m\002")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        trigger_hooks("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        agent_loop(history)
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(block.text)
        print()
