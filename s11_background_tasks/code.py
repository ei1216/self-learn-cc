#!/usr/bin/env python3
"""
s11_background_tasks.py - 后台任务

    主线程                                     后台线程
    +------------------------------+         +----------------------+
    | bash(run_in_background=True) | ------> | 运行命令             |
    | 返回 bg_id                   |         | 结果入队             |
    | 继续执行 agent 循环          | <------ +----------------------+
    | 下一轮:收集结果              |
    +------------------------------+
"""

import atexit
import glob
import os
import re
import signal
import subprocess
import threading
import time
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
    f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. "
    "Set run_in_background to true only for independent Bash commands."
)


# -- 来自 s04:工具实现 --

# 全局 Shell 进程登记表：记录本模块启动且仍在运行的命令进程，供退出时统一清理
_shell_processes: set[subprocess.Popen] = set()
# 可重入锁：保护 _shell_processes 的并发读写；用 RLock 是因为信号处理可能在
# 主线程持锁期间触发，同线程重复加锁时普通 Lock 会死锁
_shell_process_lock = threading.RLock()


def _stop_process_group(process: subprocess.Popen):
    """停止仍留在命令原始进程组中的进程。

    先发送 SIGTERM 请求优雅退出，稍候仍未结束则升级为 SIGKILL 强制终止；
    进程组已不存在（命令已自行退出）时直接返回，不视为错误。

    Args:
        process: 由 _run_bash_process 创建的 Popen 进程对象；
            因启动时使用 start_new_session=True，其 pid 即命令的进程组 ID。

    Returns:
        None
    """
    # 信号逐级升级：先 SIGTERM 温和终止，再 SIGKILL 强杀
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            # 对整个进程组发信号，连带清理命令派生的全部子进程
            os.killpg(process.pid, sig)
        except (ProcessLookupError, OSError):
            # 进程组已消失，说明命令及其子进程均已退出，无需再清理
            return
        # 短暂等待信号生效，再决定是否升级信号强度
        time.sleep(0.05)


def _stop_all_shell_processes():
    """停止当前所有仍在运行的 Shell 命令进程。

    先在锁内对登记集合复制快照，再到锁外逐个清理，
    避免持锁执行耗时的信号发送与等待。供 atexit 与信号处理共用。

    Returns:
        None
    """
    # 锁内只复制快照：防止清理期间其他线程增删集合导致遍历异常
    with _shell_process_lock:
        processes = list(_shell_processes)
    # 锁外逐个停止进程，尽量缩短锁的持有时间
    for process in processes:
        _stop_process_group(process)


def _handle_termination_signal(signum, _frame):
    """处理 SIGTERM 终止信号：先清理全部 Shell 命令进程，再按惯例退出码退出。

    Args:
        signum: 操作系统传入的信号编号（本模块仅注册 SIGTERM）。
        _frame: 信号到达时的当前栈帧，本函数不使用。

    Raises:
        SystemExit: 以 128 + 信号编号的退出码退出，
            遵循 Unix 中"进程因信号终止"的惯例（SIGTERM 即 143）。
    """
    # 程序被终止前先回收命令进程组，防止孤儿进程残留
    _stop_all_shell_processes()
    # 128 + signum 是 Unix 惯例中表示"因信号终止"的退出码
    raise SystemExit(128 + signum)


# 解释器正常退出（含 sys.exit、主循环结束）时兜底清理 Shell 进程
atexit.register(_stop_all_shell_processes)
# 注册 SIGTERM 处理：进程被 kill 时也能先清理 Shell 进程再退出
signal.signal(signal.SIGTERM, _handle_termination_signal)


def _run_bash_process(command: str) -> tuple[str, int | None]:
    """同步执行一条 Shell 命令，返回合并输出与退出码。

    命令在独立进程组（新会话）中运行，便于整体清理其派生的全部子进程；
    执行期间进程登记进全局集合，供退出钩子与信号处理统一终止。

    Args:
        command: 待执行的 Shell 命令字符串。

    Returns:
        二元组 (output, exit_code)：
            output: stdout 与 stderr 直接拼接、去首尾空白后的文本，
                超长时截断到 50000 字符，无输出时为 "(no output)"，
                超时或启动失败时为 "Error: " 开头的错误说明。
            exit_code: 命令退出码；超时或启动失败拿不到退出码时为 None。
    """
    # 先置 None：finally 中据此判断是否真的创建过进程
    process = None
    try:
        # 创建子进程：shell=True 交由系统 Shell 解释执行；
        # start_new_session=True 让命令进入新会话/新进程组，
        # 这样 killpg(process.pid) 才能连带清理其全部子进程
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=WORKDIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # 文本模式读取输出；非法字节用替换符代替，避免解码报错
            text=True, errors="replace",
            start_new_session=True,
        )
        # 在锁内登记进程，让退出钩子与信号处理能找到仍在运行的命令
        with _shell_process_lock:
            _shell_processes.add(process)
        # 阻塞等待命令结束并收集输出，超过 120 秒抛 TimeoutExpired
        stdout, stderr = process.communicate(timeout=120)
        # stdout 与 stderr 直接拼接后去除首尾空白
        output = (stdout + stderr).strip()
        # 截断到 50000 字符，防止超长输出撑爆上下文；空输出给占位提示
        return (output[:50000] if output else "(no output)"), process.returncode
    except subprocess.TimeoutExpired:
        # 超时路径：输出已不可靠，进程组由 finally 统一终止
        return "Error: Timeout (120s)", None
    except OSError as error:
        # 进程启动失败（如权限不足、资源耗尽）：返回异常摘要
        return f"Error: {type(error).__name__}: {error}", None
    finally:
        if process is not None:
            # 无论成功、超时还是异常，都确保命令进程组被终止，不留孤儿进程
            _stop_process_group(process)
            try:
                # 短暂等待进程真正退出，回收系统资源
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                # 0.2 秒内未退出也不再等待，进程组已被强制清理
                pass
            # 从登记表中移除，避免集合无限增长
            with _shell_process_lock:
                _shell_processes.discard(process)


def _format_bash_result(output: str, exit_code: int | None) -> str:
    """按退出码格式化命令结果，失败时附加错误前缀。

    Args:
        output: _run_bash_process 返回的输出文本。
        exit_code: 命令退出码；None 表示超时或启动失败。

    Returns:
        退出码为 0 或 None 时原样返回 output（None 时输出本身已带
        Error 前缀）；非零退出码时返回以
        "Error: command exited with status ..." 开头的文本，
        让模型能显式识别命令失败。
    """
    # 退出码 0 为成功；None 表示超时/启动失败，输出已含错误说明，原样返回
    if exit_code in (0, None):
        return output
    # 非零退出码统一加错误前缀，显式告知模型命令执行失败
    return f"Error: command exited with status {exit_code}\n{output}"


def run_bash(command: str, run_in_background: bool = False) -> str:
    """同步执行一条 Bash 命令并返回格式化结果（bash 工具的处理器入口）。

    本函数只负责前台同步执行；后台执行在 execute_tool 中被
    should_run_background 拦截并交给 BackgroundManager 处理，
    后台线程也是直接调用 _run_bash_process，不会经过本函数。

    Args:
        command: 待执行的 Shell 命令字符串。
        run_in_background: 与 TOOLS 中 bash 工具的参数声明保持一致，
          仅用于工具 Schema 定义；凡真正进入本函数的调用一定是
          前台同步执行，因此该参数在此不参与任何逻辑。

    Returns:
        经 _format_bash_result 格式化后的命令输出：
        成功时为命令输出本身，失败时带 "Error: " 前缀。
    """
    # 解包 _run_bash_process 返回的 (output, exit_code) 二元组，
    # 交给 _format_bash_result 按退出码统一格式化
    return _format_bash_result(*_run_bash_process(command))


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
                      "properties": {
                          "command": {"type": "string"},
                          "run_in_background": {"type": "boolean"}},
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


def call_tool(block) -> str:
    handler = TOOL_HANDLERS.get(block.name)
    try:
        output = handler(**block.input) if handler else f"Unknown: {block.name}"
    except Exception as error:
        output = f"Error: {error}"
    return str(output)


# -- s11 新增:后台执行 --

class BackgroundManager:
    """后台任务管理器：负责启动后台 Bash 任务、在后台线程执行命令并收集结果。"""

    def __init__(self):
        # 任务注册表：task_id -> 任务信息（tool_use_id / command / status）
        self.tasks: dict[str, dict] = {}
        # 已完成任务的结果：task_id -> 格式化后的输出文本
        self.results: dict[str, str] = {}
        # 已完成、等待主循环收取的任务 ID 队列（按完成先后排序）
        self._ready: list[str] = []
        # 任务编号自增计数器，用于生成 bg_0001 这样的递增 ID
        self._counter = 0
        # 互斥锁：共享数据同时被主线程与后台线程读写，所有访问都要加锁
        self._lock = threading.Lock()

    def start(self, block) -> str:
        """校验并启动一个后台 Bash 任务，返回任务 ID（bg_XXXX）。

        只接受 bash 工具调用且命令非空；先在锁内登记任务，
        再用守护线程执行，启动失败时回滚登记。
        """
        # 后台执行只针对 Bash 命令，其他工具一律拒绝
        if block.name != "bash":
            raise ValueError("Only Bash commands can run in the background")
        # 命令必须是非空字符串，缺失或纯空白都视为非法
        command = block.input.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("Bash command cannot be empty")

        # 在锁内注册任务：编号自增并格式化为 4 位，初始状态为 running
        with self._lock:
            self._counter += 1
            task_id = f"bg_{self._counter:04d}"
            self.tasks[task_id] = {
                "tool_use_id": block.id,
                "command": command,
                "status": "running",
            }

        # 用守护线程执行命令：不阻塞主循环，主程序退出时线程随之结束
        thread = threading.Thread(
            target=self._run,
            args=(task_id, command),
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            # 线程启动失败时回滚任务登记，保持状态一致，再向上抛出异常
            with self._lock:
                self.tasks.pop(task_id, None)
            raise
        print(f"  [background] started {task_id}: {command[:60]}")
        return task_id

    def _run(self, task_id: str, command: str):
        """后台线程主体：真正执行命令，并把结果与最终状态写回任务表。"""
        try:
            # 复用 s04 的同步执行函数：在独立进程组中运行命令并等待输出
            output, exit_code = _run_bash_process(command)
            # 按退出码格式化输出，并据此判定任务最终状态
            result = _format_bash_result(output, exit_code)
            status = "completed" if exit_code == 0 else "failed"
        except Exception as error:
            # 执行中的任何异常都记为失败，避免后台线程静默消亡、结果丢失
            result = f"Error: {type(error).__name__}: {error}"
            status = "failed"

        # 在锁内写回结果；若任务已被移出注册表（如启动失败被回滚），直接丢弃
        with self._lock:
            task = self.tasks.get(task_id)
            if task is None:
                return
            task["status"] = status
            self.results[task_id] = result
            # 标记为"可收取"，等待主循环在后续轮次通过 collect 取走
            self._ready.append(task_id)

    def collect(self) -> list[str]:
        """收取所有已完成的任务，组装为 <task_notification> 通知文本列表。

        每次调用都原子地取走全部就绪任务（取出即从注册表中删除），
        无完成任务时返回空列表。
        """
        # 在锁内一次性"取走"全部就绪任务：同步 pop 任务与结果，避免重复收取
        with self._lock:
            ready = []
            for task_id in self._ready:
                task = self.tasks.pop(task_id, None)
                result = self.results.pop(task_id, "")
                if task is not None:
                    ready.append((task_id, task, result))
            self._ready.clear()

        # 通知在锁外组装：摘要最多保留 500 字符，防止超长输出撑爆上下文
        notifications = []
        for task_id, task, result in ready:
            notifications.append(
                f"<task_notification>\n"
                f"  <task_id>{task_id}</task_id>\n"
                f"  <status>{task['status']}</status>\n"
                f"  <command>{task['command']}</command>\n"
                f"  <summary>{result[:500]}</summary>\n"
                f"</task_notification>"
            )
            print(f"  [background] collected {task_id}: {task['status']}")
        return notifications


# 全局唯一的后台任务管理器，供本模块所有执行路径共用
BACKGROUND = BackgroundManager()
# 便捷别名：外部（如测试）可直接查看任务表与结果表
background_tasks = BACKGROUND.tasks
background_results = BACKGROUND.results


def should_run_background(tool_name: str, tool_input: dict) -> bool:
    return (
        tool_name == "bash"
        and tool_input.get("run_in_background") is True
    )


def start_background_task(block) -> str:
    return BACKGROUND.start(block)


def collect_background_results() -> list[str]:
    return BACKGROUND.collect()


def inject_background_results(messages: list) -> int:
    notifications = collect_background_results()
    if not notifications:
        return 0

    blocks = [{"type": "text", "text": item} for item in notifications]
    if messages and messages[-1].get("role") == "user":
        content = messages[-1].get("content", "")
        if isinstance(content, list):
            content.extend(blocks)
        else:
            messages[-1]["content"] = [
                {"type": "text", "text": str(content)},
                *blocks,
            ]
    else:
        messages.append({"role": "user", "content": blocks})
    return len(notifications)


def execute_tool(block) -> str:
    blocked = trigger_hooks("PreToolUse", block)
    if blocked is not None:
        return str(blocked)

    if should_run_background(block.name, block.input):
        try:
            task_id = start_background_task(block)
            output = (
                f"[Background task {task_id} started] "
                "The result will be collected on a later turn."
            )
        except Exception as error:
            output = f"Error: {error}"
    else:
        output = call_tool(block)

    trigger_hooks("PostToolUse", block, output)
    return output


# -- Agent 循环 --

def agent_loop(messages: list):
    while True:
        inject_background_results(messages)
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
    print("s11: Background Tasks - explicit background Bash execution")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            # \001/\002 用于告知 Readline:这些 ANSI 转义符的显示宽度为零。
            query = input("\001\033[36m\002s11 >> \001\033[0m\002")
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
