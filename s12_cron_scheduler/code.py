#!/usr/bin/env python3
"""
s12_cron_scheduler.py - 定时任务调度器

    +--------------------------+   09:00   +-----------------------+
    | 0 9 * * *               | -------->  | [Scheduled] 运行测试  |
    | prompt: "运行测试"       |            +-----------+-----------+
    +--------------------------+                       |
          scheduled_jobs                    cron_queue | agent 空闲
                                                        v
                                                +-------------+
                                                | Agent Loop  |
                                                +-------------+
                                                
"""

import glob
import json
import os
import re
import secrets
import subprocess
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
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
DURABLE_PATH = WORKDIR / ".scheduled_tasks.json"
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. "
    "Use schedule_cron for work that should start at a future local time."
)


# -- 来自 s04:工具实现 --

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
        if result.returncode != 0:
            return f"Error: command exited with status {result.returncode}\n{output}"
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        lines = file_path.read_text(encoding="utf-8").splitlines()
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


TOOLS = [
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
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "old_text": {"type": "string"},
                                     "new_text": {"type": "string"}},
                      "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern; ** matches recursively.",
     "input_schema": {"type": "object",
                      "properties": {"pattern": {"type": "string"}},
                      "required": ["pattern"]}},
]

TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}


# -- 来自 s04:钩子与权限检查 --

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


def request_permission(block, reason: str) -> str | None:
    if threading.current_thread() is not threading.main_thread():
        return "Permission denied: scheduled turns cannot request interactive approval"

    print(f"\n\033[33m[permission] {reason}\033[0m")
    print(f"   Tool: {block.name}({block.input})")
    choice = input("   Allow? [y/N] ").strip().lower()
    if choice not in ("y", "yes"):
        return "Permission denied by user"
    return None


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
            return request_permission(block, "Potentially destructive command")

    if block.name in ("read_file", "write_file", "edit_file"):
        path = block.input.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            return request_permission(block, "Access outside workspace")
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


def context_inject_hook(query: str):
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


register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


# -- s12 新增:定时任务(cron) --

@dataclass
class CronJob:
    """一条定时任务的完整记录，调度、投递、持久化都围绕它进行。

    调度线程按 cron 表达式比对当前时间，命中后把 pending_delivery 置为
    True 并放入 cron_queue；
    Agent Loop 接收后，周期任务清掉标记等下次触发，一次性任务则直接删除。
    durable 为 True 的任务会随状态变化写回磁盘。
    """

    id: str                         # 任务唯一 ID，"cron_" 前缀 + 随机十六进制
    cron: str                       # 五段式 cron 表达式(分 时 日 月 星期)，决定触发时机
    prompt: str                     # 触发后以 "[Scheduled] ..." 消息交给 Agent 执行的任务
    recurring: bool                 # True 为周期任务，False 为一次性任务，触发一次即删除
    durable: bool                   # True 持久化到磁盘，程序重启后可恢复
    pending_delivery: bool = False  # 已到期但还没被 Agent 接收时为 True，防止重复投递
    last_fired: str | None = None   # 上次触发的 "YYYY-MM-DD HH:MM" 标记，防止同一分钟重复入队


# 任务登记表：所有已注册的定时任务都在这里，是 id 到 CronJob 的映射；
# 调度、取消、投递确认都要先查这张表
scheduled_jobs: dict[str, CronJob] = {}

# 到期待投递队列：已触发但还没被 Agent 执行的任务在此排队，
# Agent 空闲时由 queue_processor_loop 取走消费
cron_queue: list[CronJob] = []

# 可重入锁：保护上面两个结构；用 RLock 是因为持锁期间还会调用
# save_durable_jobs() 等函数，它们内部要再拿同一把锁，普通 Lock 会死锁
cron_lock = threading.RLock()


def _cron_field_matches(field: str, value: int) -> bool:
    """判断单个 cron 字段是否命中某个数值。

    Args:
        field: 单个字段，支持 *、*/步长、逗号列表、范围和精确值。
        value: 用于比对的数值。

    Returns:
        bool: 命中返回 True。
    """
    if field == "*":
        return True
    if field.startswith("*/"):
        return value % int(field[2:]) == 0
    if "," in field:
        return any(_cron_field_matches(part.strip(), value)
                   for part in field.split(","))
    if "-" in field:
        start, end = field.split("-", 1)
        return int(start) <= value <= int(end)
    return value == int(field)


def cron_matches(cron_expr: str, moment: datetime) -> bool:
    """判断某个时刻是否命中一条五段式 cron 表达式。

    Args:
        cron_expr: "分 时 日 月 星期" 五段式表达式。
        moment: 待比对的时刻。

    Returns:
        bool: 命中返回 True；段数不是 5 时返回 False。
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False

    minute, hour, day, month, weekday = fields
    # Python 的星期一是 0，cron 的星期天是 0，换算对齐
    cron_weekday = (moment.weekday() + 1) % 7
    if not (
        _cron_field_matches(minute, moment.minute)
        and _cron_field_matches(hour, moment.hour)
        and _cron_field_matches(month, moment.month)
    ):
        return False

    day_matches = _cron_field_matches(day, moment.day)
    weekday_matches = _cron_field_matches(weekday, cron_weekday)
    if day == "*" and weekday == "*":
        return True
    if day == "*":
        return weekday_matches
    if weekday == "*":
        return day_matches
    # 日和星期都指定时按标准 cron 语义取“或”：命中其一即触发
    return day_matches or weekday_matches


def _validate_cron_field(field: str, minimum: int, maximum: int) -> str | None:
    """校验单个 cron 字段的写法与取值范围。

    Args:
        field: 单个字段，支持 *、*/步长、逗号列表、范围和精确值。
        minimum: 该字段允许的最小值。
        maximum: 该字段允许的最大值。

    Returns:
        str | None: 合法返回 None；非法返回错误说明字符串。
    """
    if field == "*":
        return None
    if field.startswith("*/"):
        step = field[2:]
        if not step.isdigit() or int(step) <= 0:
            return f"Invalid step: {field}"
        return None
    if "," in field:
        for part in field.split(","):
            error = _validate_cron_field(part.strip(), minimum, maximum)
            if error:
                return error
        return None
    if "-" in field:
        start, end = field.split("-", 1)
        if not start.isdigit() or not end.isdigit():
            return f"Invalid range: {field}"
        start_value, end_value = int(start), int(end)
        if start_value > end_value:
            return f"Range start is greater than end: {field}"
        if start_value < minimum or end_value > maximum:
            return f"Range {field} is outside [{minimum}-{maximum}]"
        return None
    if not field.isdigit():
        return f"Invalid field: {field}"
    value = int(field)
    if value < minimum or value > maximum:
        return f"Value {value} is outside [{minimum}-{maximum}]"
    return None


def validate_cron(cron_expr: str) -> str | None:
    """校验整条 cron 表达式的段数与各段取值。

    Args:
        cron_expr: 五段式 cron 表达式。

    Returns:
        str | None: 合法返回 None；非法返回带字段名的错误说明。
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"Expected 5 fields, got {len(fields)}"

    # 五段各自的名称与取值范围
    field_rules = [
        ("minute", 0, 59),
        ("hour", 0, 23),
        ("day-of-month", 1, 31),
        ("month", 1, 12),
        ("day-of-week", 0, 6),
    ]
    for field, (name, minimum, maximum) in zip(fields, field_rules):
        error = _validate_cron_field(field, minimum, maximum)
        if error:
            return f"{name}: {error}"
    return None


def save_durable_jobs():
    """把全部 durable 任务整体写入磁盘存档。

    先写临时文件再原子替换，文件内容要么完整是旧的、要么完整是新的，
    中途崩溃也不会留下写一半的脏数据。

    Returns:
        None
    """
    with cron_lock:
        # 只导出 durable 任务，session 任务重启后本来就不需要
        payload = [
            asdict(job)
            for job in scheduled_jobs.values()
            if job.durable
        ]
        # 临时文件名带进程号和线程号，避免并发写入互相踩踏
        temporary = DURABLE_PATH.with_name(
            f"{DURABLE_PATH.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temporary, DURABLE_PATH)
        finally:
            # 替换成功后临时文件已不存在，missing_ok 兜底
            temporary.unlink(missing_ok=True)


def load_durable_jobs():
    """启动时从磁盘恢复 durable 任务。

    文件不存在视为首次运行；文件损坏只提示不崩溃；
    单条数据非法时跳过该条，不影响其余任务。

    Returns:
        None
    """
    if not DURABLE_PATH.exists():
        return
    try:
        payload = json.loads(DURABLE_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("expected a JSON list")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        # 文件损坏：提示后放弃恢复，程序照常启动
        print(f"  [cron] could not load {DURABLE_PATH.name}: {error}")
        return

    loaded = 0
    with cron_lock:
        for item in payload:
            try:
                # 逐条重建并校验：cron 合法、id 前缀正确、prompt 非空
                job = CronJob(**item)
                error = validate_cron(job.cron)
                if error:
                    raise ValueError(error)
                if not job.id.startswith("cron_"):
                    raise ValueError("invalid job ID")
                if not job.prompt.strip():
                    raise ValueError("prompt cannot be empty")
            except (TypeError, ValueError) as error:
                # 坏数据只跳过这一条，不拖垮其余任务
                print(f"  [cron] skipped invalid saved job: {error}")
                continue
            scheduled_jobs[job.id] = job
            if job.pending_delivery:
                # 上次到期还没投出去的，重新排队保证投递不丢
                cron_queue.append(job)
            loaded += 1
    if loaded:
        print(f"  [cron] loaded {loaded} durable job(s)")


def new_cron_id() -> str:
    """生成一个不与现有任务冲突的新任务 ID。

    Returns:
        str: 形如 "cron_" 加 8 位随机十六进制的 ID。

    Raises:
        RuntimeError: 连续 100 次生成的 ID 都撞车时抛出，概率极低。
    """
    for _ in range(100):
        job_id = f"cron_{secrets.token_hex(4)}"
        if job_id not in scheduled_jobs:
            return job_id
    raise RuntimeError("Could not allocate a cron job ID")


def schedule_job(cron: str, prompt: str, recurring: bool = True,
                 durable: bool = True) -> CronJob | str:
    """校验并创建一条定时任务，登记后按需落盘。

    Args:
        cron: 五段式 cron 表达式。
        prompt: 触发后交给 Agent 执行的任务内容。
        recurring: 是否周期任务，默认 True。
        durable: 是否持久化到磁盘，默认 True。

    Returns:
        CronJob | str: 成功返回新建的任务对象；
            校验失败返回错误说明字符串。
    """
    error = validate_cron(cron)
    if error:
        return error
    if not prompt.strip():
        return "Prompt cannot be empty"

    with cron_lock:
        job = CronJob(
            id=new_cron_id(),
            cron=cron,
            prompt=prompt,
            recurring=recurring,
            durable=durable,
        )
        scheduled_jobs[job.id] = job
        try:
            if durable:
                save_durable_jobs()
        except Exception:
            # 落盘失败就回滚登记，保证内存与磁盘一致
            scheduled_jobs.pop(job.id, None)
            raise
    print(f"  [cron] scheduled {job.id}: {cron} -> {prompt[:60]}")
    return job


def cancel_job(job_id: str) -> str:
    """按 ID 取消一条定时任务。

    Args:
        job_id: 要取消的任务 ID。

    Returns:
        str: 成功返回确认文本；ID 不存在返回提示文本。
    """
    with cron_lock:
        job = scheduled_jobs.get(job_id)
        if job is None:
            return f"Job {job_id} not found"

        previous_queue = list(cron_queue)   # 队列快照，落盘失败时还原
        scheduled_jobs.pop(job_id)
        cron_queue[:] = [queued for queued in cron_queue if queued.id != job_id]
        try:
            if job.durable:
                save_durable_jobs()
        except Exception:
            # 落盘失败：恢复登记表和队列，当作没取消过
            scheduled_jobs[job_id] = job
            cron_queue[:] = previous_queue
            raise
    print(f"  [cron] cancelled {job_id}")
    return f"Cancelled {job_id}"


def _enqueue_due_job(job: CronJob, minute_marker: str | None = None):
    """把一条到期任务标记后放入投递队列。

    先落盘再入队：保证进了队列的任务一定已持久化，
    程序中途崩溃重启后也不会丢投递。

    Args:
        job: 已命中 cron 表达式的到期任务。
        minute_marker: 本次触发的 "YYYY-MM-DD HH:MM" 标记。

    Raises:
        Exception: 落盘失败时原样抛出，任务不会进入队列。
    """
    # 记下旧值，落盘失败时回滚
    old_pending = job.pending_delivery
    old_last_fired = job.last_fired
    job.pending_delivery = True
    if minute_marker is not None:
        job.last_fired = minute_marker
    try:
        if job.durable:
            save_durable_jobs()
    except Exception:
        # 回滚标记，下次到点还能重试
        job.pending_delivery = old_pending
        job.last_fired = old_last_fired
        raise
    # 落盘成功才入队
    cron_queue.append(job)


def poll_due_jobs(moment: datetime):
    """扫描全部任务，把当前时刻命中的任务放入投递队列。

    Args:
        moment: 当前时刻，由调度线程定时传入。
    """
    minute_marker = moment.strftime("%Y-%m-%d %H:%M")
    with cron_lock:
        # 拷贝一份再遍历，防止循环期间登记表被其他线程改动
        for job in list(scheduled_jobs.values()):
            try:
                # 还没被接收的、本分钟已触发过的，跳过防止重复入队
                if job.pending_delivery or job.last_fired == minute_marker:
                    continue
                if cron_matches(job.cron, moment):
                    _enqueue_due_job(job, minute_marker)
                    print(f"  [cron] due {job.id}: {job.prompt[:60]}")
            except Exception as error:
                # 单条任务出错只记录，不拖垮整轮调度
                print(f"  [cron] could not enqueue {job.id}: {error}")


def consume_cron_queue() -> list[CronJob]:
    """一次性取走队列里的全部任务并清空。

    Returns:
        list[CronJob]: 取走的任务列表，可能为空。
    """
    with cron_lock:
        jobs = list(cron_queue)
        # 取走即清空，避免同一条任务被重复消费
        cron_queue.clear()
    return jobs


def acknowledge_cron_jobs(jobs: list[CronJob]):
    """确认一批任务已被 Agent 接收。

    周期任务清掉投递标记等待下次触发，一次性任务直接删除；
    落盘失败时整体回滚，效果等同没确认过。

    Args:
        jobs: 本轮从队列消费掉的任务列表。
    """
    changed: list[tuple[CronJob, bool]] = []   # (任务, 旧标记)，回滚用
    removed: list[CronJob] = []
    with cron_lock:
        for delivered in jobs:
            current = scheduled_jobs.get(delivered.id)
            if current is None:
                # 任务已被取消，不用确认
                continue 
            changed.append((current, current.pending_delivery))
            if current.recurring:
                # 周期任务：清掉标记，等下次到点再触发
                current.pending_delivery = False
            else:
                # 一次性任务：确认收到后直接删除
                removed.append(current)
                scheduled_jobs.pop(current.id)

        try:
            if any(job.durable for job, _ in changed):
                save_durable_jobs()
        except Exception:
            # 落盘失败：恢复被删任务和投递标记，不在队列的补回队列
            for job in removed:
                scheduled_jobs[job.id] = job
            for job, pending in changed:
                job.pending_delivery = pending
            queued_ids = {job.id for job in cron_queue}
            for job, _ in changed:
                if job.id not in queued_ids:
                    cron_queue.append(job)
            raise


def restore_cron_jobs(jobs: list[CronJob]):
    """把已投递但本轮 Agent 循环失败的任务退回队列，等待重新投递。

    Args:
        jobs: 要退回的任务列表。
    """
    with cron_lock:
        queued_ids = {job.id for job in cron_queue}
        for delivered in jobs:
            current = scheduled_jobs.get(delivered.id)
            if current is None:
                # 已被取消的不恢复
                continue
            current.pending_delivery = True
            if current.id not in queued_ids:
                # 已在队列里的不重复入队
                cron_queue.append(current)
                queued_ids.add(current.id)


def has_cron_queue() -> bool:
    """判断投递队列里有没有待执行的任务。

    Returns:
        bool: 队列非空返回 True。
    """
    with cron_lock:
        return bool(cron_queue)


def run_schedule_cron(cron: str, prompt: str, recurring: bool = True,
                      durable: bool = True) -> str:
    """schedule_cron 工具入口：把调度结果转成给模型看的文本。

    Args:
        cron: 五段式 cron 表达式。
        prompt: 触发后交给 Agent 执行的任务内容。
        recurring: 是否周期任务，默认 True。
        durable: 是否持久化到磁盘，默认 True。

    Returns:
        str: 成功返回确认文本；失败返回 Error 前缀的错误说明。
    """
    result = schedule_job(cron, prompt, recurring, durable)
    # 返回字符串说明是校验错误
    if isinstance(result, str):
        return f"Error: {result}"
    return f"Scheduled {result.id}: {cron} -> {prompt}"


def run_list_crons() -> str:
    """list_crons 工具入口：列出全部定时任务。

    Returns:
        str: 每行一条任务并标注周期性与存储方式；没有任务时返回提示。
    """
    with cron_lock:
        jobs = list(scheduled_jobs.values())   # 锁内只取快照，拼接放锁外
    if not jobs:
        return "No cron jobs."

    lines = []
    for job in jobs:
        frequency = "recurring" if job.recurring else "one-shot"
        storage = "durable" if job.durable else "session"
        lines.append(
            f"{job.id}: {job.cron} -> {job.prompt[:60]} "
            f"[{frequency}, {storage}]"
        )
    return "\n".join(lines)


def run_cancel_cron(job_id: str) -> str:
    """cancel_cron 工具入口：直接转发给 cancel_job。

    Args:
        job_id: 要取消的任务 ID。

    Returns:
        str: 取消结果文本。
    """
    return cancel_job(job_id)


TOOLS.extend([
    {"name": "schedule_cron",
     "description": "Schedule a prompt with a 5-field cron expression.",
     "input_schema": {"type": "object",
                      "properties": {
                          "cron": {"type": "string"},
                          "prompt": {"type": "string"},
                          "recurring": {"type": "boolean"},
                          "durable": {"type": "boolean"}},
                      "required": ["cron", "prompt"]}},
    {"name": "list_crons", "description": "List scheduled cron jobs.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "cancel_cron", "description": "Cancel a cron job by ID.",
     "input_schema": {"type": "object",
                      "properties": {"job_id": {"type": "string"}},
                      "required": ["job_id"]}},
])

TOOL_HANDLERS.update({
    "schedule_cron": run_schedule_cron,
    "list_crons": run_list_crons,
    "cancel_cron": run_cancel_cron,
})


def execute_tool(block) -> str:
    blocked = trigger_hooks("PreToolUse", block)
    if blocked is not None:
        return str(blocked)

    handler = TOOL_HANDLERS.get(block.name)
    try:
        output = handler(**block.input) if handler else f"Unknown: {block.name}"
    except Exception as error:
        output = f"Error: {error}"
    trigger_hooks("PostToolUse", block, output)
    return str(output)


# -- 调度器与 agent 循环 --

# 全局停止信号：stop_runtime_threads 置位后，两个后台线程依次退出
RUNTIME_STOP = threading.Event()

# 已启动的后台线程登记表，停止时逐个 join 回收
runtime_threads: list[threading.Thread] = []

# 是否已启动过后台线程，防止 start_runtime_threads 重复启动
runtime_started = False

# 启动/停止过程自身的互斥锁，防止并发地启动或停止运行时
runtime_lock = threading.Lock()

# Agent 循环的互斥锁：保证同一时刻只有一轮 Agent 循环在跑，
# 主线程交互和队列处理线程都要先拿到它
agent_lock = threading.Lock()

# 整个会话共享的对话历史，主线程与队列处理线程共用（受 agent_lock 保护）
session_history: list = []


def cron_scheduler_loop(stop_event: threading.Event = RUNTIME_STOP):
    """调度线程的主循环：每秒醒来一次，扫描到期任务。

    Args:
        stop_event: 停止信号，默认用全局 RUNTIME_STOP；被置位后循环退出。
    """
    # Event.wait(1.0) 一举两得：休眠 1 秒，期间被置位就返回 True 退出
    while not stop_event.wait(1.0):
        poll_due_jobs(datetime.now())


def agent_loop(messages: list, context: dict | None = None):
    # 取走队列里全部到期的定时任务,取走即清空
    fired = consume_cron_queue()   
    # 记录注入前的消息条数,失败时按这个位置回滚
    scheduled_start = len(messages)   
    for job in fired:
        # 把每个到期任务包装成一条用户消息塞进对话,让模型当作新任务去处理
        messages.append({"role": "user", "content": f"[Scheduled] {job.prompt}"})
        print(f"  [cron] delivered {job.id}: {job.prompt[:60]}")

    # 已投递但还没确认接收的任务,等模型回应后再确认
    waiting_for_ack = list(fired)   
    while True:
        try:
            response = client.messages.create(
                model=MODEL,
                system=SYSTEM,
                messages=messages,
                tools=TOOLS,
                max_tokens=8000,
            )
        except Exception as error:
            if waiting_for_ack:
                # 请求失败就撤回刚注入的定时任务消息
                del messages[scheduled_start:]
                # 再把任务退回队列,等下一轮重新投递
                restore_cron_jobs(waiting_for_ack)
            print(f"  [error] {type(error).__name__}: {error}")
            return context

        messages.append({"role": "assistant", "content": response.content})
        if waiting_for_ack:
            try:
                # 模型已收到任务,现在正式确认:周期任务清标记等下次触发,一次性任务直接删除
                acknowledge_cron_jobs(waiting_for_ack)
            except Exception as error:
                print(f"  [cron] acknowledgement failed: {error}")
            waiting_for_ack = []   # 确认完毕,清空待确认列表

        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]
        if not tool_calls:
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return context

        results = []
        for block in tool_calls:
            output = execute_tool(block)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })
        messages.append({"role": "user", "content": results})


def print_latest_assistant_text(messages: list):
    """打印最新一条 assistant 消息的文本。

    Args:
        messages: 对话消息历史列表，从后往前找第一个 assistant 消息。

    Returns:
        None
    """
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            print(content)
        else:
            # content 是块列表：SDK 块对象和 dict 两种形态都兼容
            for block in content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    print(block.get("text", ""))
        # 只打印最新一条，处理完立即结束
        return


def run_agent_turn_locked(user_query: str | None = None):
    """执行一轮 Agent 循环并打印最新回复，调用方须已持有 agent_lock。

    Args:
        user_query: 用户输入；为 None 表示由定时任务触发，不带新输入。
    """
    if user_query is not None:
        # 先过 UserPromptSubmit 钩子，再把输入追加进会话历史
        trigger_hooks("UserPromptSubmit", user_query)
        session_history.append({"role": "user", "content": user_query})
    agent_loop(session_history)
    print_latest_assistant_text(session_history)
    # 末尾补一个空行，分隔两轮输出
    print()


def queue_processor_loop(stop_event: threading.Event = RUNTIME_STOP):
    """队列处理线程的主循环：Agent 空闲时消费到期的定时任务。

    Args:
        stop_event: 停止信号，默认用全局 RUNTIME_STOP；被置位后循环退出。
    """
    # 每 0.2 秒醒一次，兼做停止检测
    while not stop_event.wait(0.2):
        # 队列为空、或 Agent 正忙（主线程持锁），都跳过这一轮
        if not has_cron_queue() or not agent_lock.acquire(blocking=False):
            continue
        try:
            # 抢到锁后再查一次：等待期间队列可能已被主线程消费完
            if has_cron_queue():
                run_agent_turn_locked()
        finally:
            agent_lock.release()


def start_runtime_threads():
    """启动调度线程和队列处理线程（幂等，重复调用只生效一次）。

    Returns:
        None
    """
    global runtime_started
    with runtime_lock:
        if runtime_started:
            return
        # 启动前先从磁盘恢复 durable 任务
        load_durable_jobs()
        # 清掉停止信号，支持停止后再次启动
        RUNTIME_STOP.clear()
        # 两个守护线程：调度循环 + 队列处理循环，主程序退出时自动结束
        runtime_threads.extend([
            threading.Thread(
                target=cron_scheduler_loop,
                name="cron-scheduler",
                daemon=True,
            ),
            threading.Thread(
                target=queue_processor_loop,
                name="cron-queue-processor",
                daemon=True,
            ),
        ])
        for thread in runtime_threads:
            thread.start()
        runtime_started = True


def stop_runtime_threads():
    """停止两个后台线程（幂等，未启动时直接返回）。

    Returns:
        None
    """
    global runtime_started
    with runtime_lock:
        if not runtime_started:
            return
        # 置位停止信号，两个循环会在下个轮询周期退出
        RUNTIME_STOP.set()
        for thread in runtime_threads:
            # 最多等 1 秒；daemon 线程超时后会随主进程一起结束
            thread.join(timeout=1)
        runtime_threads.clear()
        runtime_started = False


if __name__ == "__main__":
    print("s12: Cron Scheduler - run prompts on a local schedule")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    # 启动两个后台线程:调度线程到点触发任务,队列处理线程负责投递
    start_runtime_threads()
    # 用 try/finally 兜底:无论怎么退出,最后都把后台线程停干净
    try:
        # 主交互循环:读用户输入,一轮一轮地跑对话
        while True:
            try:
                # \001/\002 告诉 Readline:这些 ANSI 转义符不占显示宽度
                query = input("\001\033[36m\002s12 >> \001\033[0m\002")
            except (EOFError, KeyboardInterrupt):
                # Ctrl+D 或 Ctrl+C 直接退出
                break
            if query.strip().lower() in ("q", "exit", ""):
                # 输入 q、exit 或空行就算退出
                break
            with agent_lock:
                # 拿到锁才能跑本轮对话,避免和队列处理线程同时跑 agent_loop
                run_agent_turn_locked(query)
    finally:
        # 无论正常退出还是中途报错,都停掉后台线程
        stop_runtime_threads()
