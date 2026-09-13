#!/usr/bin/env python3
"""
s04_hooks.py - 钩子

钩子会在 agent 循环的固定节点上执行回调：

    User prompt
         |
         v
    UserPromptSubmit
         |
         v
    +----------+      +-------+      +------------+      +-------+
    | messages | ---> |  LLM  | ---> | PreToolUse | ---> | Tool  |
    +----------+      +---+---+      | permission |      +---+---+
         ^                | stop     | log        |          |
         |                v          +------------+          v
         |            Stop hook                         PostToolUse
         |                                               |
         +---------------- tool_result ------------------+
"""

import os
import re
import subprocess
from pathlib import Path

try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
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

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


# -- 来自 s02-s03：工具实现 --

def run_bash(command: str) -> str:
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, errors="replace", timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int | None = None) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        lines = file_path.read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

def run_glob(pattern: str) -> str:
    import glob as g
    try:
        matches = sorted({
            match for match in g.glob(
                pattern, root_dir=WORKDIR, recursive=True)
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR)
        })
        shown = matches[:200]
        if len(matches) > 200:
            shown.append("... (more matches omitted; narrow the pattern)")
        return "\n".join(shown) if shown else "(no matches)"
    except Exception as e:
        return f"Error: {e}"

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
]

TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}


# -- s04 新增：钩子系统（s03 的权限逻辑改用钩子实现） --

# 钩子注册表：事件名 -> 回调函数列表（按注册顺序依次触发）
# 四个挂载点：提交提示词前 / 工具执行前 / 工具执行后 / 循环停止时
HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}

def register_hook(event: str, callback):
    """把回调函数注册到指定事件的钩子列表末尾。

    Args:
        event: 事件名（HOOKS 的四个键之一）。
        callback: 事件触发时执行的回调函数。
    """
    HOOKS[event].append(callback)

def trigger_hooks(event: str, *args):
    """依次触发某事件上注册的所有回调。

    Args:
        event: 事件名。
        *args: 透传给每个回调的参数。

    Returns:
        第一个返回非 None 的回调的结果（视为拦截信号）；
        所有回调都放行时返回 None。
    """
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:  # 钩子返回非 None 结果 = 拦截本次工具调用
            return result
            
    return None


# s03 的权限检查逻辑，现在包装成了钩子
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]


def contains_destructive_command(command: str) -> bool:
    """判断命令中是否出现独立的 rm / del 删除命令。

    Args:
        command: bash 命令字符串。

    Returns:
        True 表示疑似破坏性删除命令，需要升级审批。
    """
    return bool(DESTRUCTIVE_COMMAND_WORD.search(command))


def permission_hook(block):
    """PreToolUse 钩子：s03 的 check_permission() 逻辑搬到了这里。

    Args:
        block: 模型的 tool_use 块。

    Returns:
        拒绝原因字符串 = 拦截本次调用；None = 放行。
    """
    # 第一部分：bash 命令检查
    if block.name == "bash":
        command = block.input.get("command", "")
        # 命中硬拒绝清单：红色提示并直接拦截，不询问
        for pattern in DENY_LIST:
            if pattern in command:
                print(f"\n\033[31m[blocked] '{pattern}'\033[0m")
                return "Permission denied by deny list"
        # 疑似破坏性命令：黄色提示，交由用户裁决
        if contains_destructive_command(command) or any(
            kw in command for kw in DESTRUCTIVE
        ):
            print(f"\n\033[33m[permission] Potentially destructive command\033[0m")
            print(f"   Tool: {block.name}({block.input})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    # 第二部分：文件类工具检查
    if block.name in ("read_file", "write_file", "edit_file"):
        path = block.input.get("path", "")
        # 路径解析后逃出工作区：交由用户裁决
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print(f"\n\033[33m[permission] Access outside workspace\033[0m")
            print(f"   Tool: {block.name}({block.input})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    return None

def log_hook(block):
    """PreToolUse 钩子：记录每一次工具调用。

    Args:
        block: 模型的 tool_use 块。

    Returns:
        恒为 None：纯观测钩子，不拦截任何调用。
    """
    # 只取前 2 个参数值、最多 60 字符做灰色预览
    args_preview = str(list(block.input.values())[:2])[:60]
    print(f"\033[90m[HOOK] {block.name}({args_preview})\033[0m")
    return None

def large_output_hook(block, output):
    """PostToolUse 钩子：工具输出过大时给出警告。

    Args:
        block: 模型的 tool_use 块。
        output: 工具执行的返回结果。

    Returns:
        恒为 None：只警告，不影响已产生的工具结果。
    """
    # 超过 10 万字符才提示：输出过大容易撑爆模型上下文
    if len(str(output)) > 100000:
        print(f"\033[33m[HOOK] Large output from {block.name}: {len(str(output))} chars\033[0m")
    return None

# UserPromptSubmit 钩子：在用户输入进入 LLM 之前记录它
def context_inject_hook(query: str):
    """UserPromptSubmit 钩子：提交提示词前触发（可在此注入上下文）。

    Args:
        query: 用户输入的原始文本。

    Returns:
        恒为 None：仅记录，不修改输入。
    """
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None

# Stop 钩子：循环即将退出时打印会话统计摘要
def summary_hook(messages: list):
    """Stop 钩子：循环停止时统计本次会话的工具调用次数。

    Args:
        messages: 完整对话历史。

    Returns:
        恒为 None（正常退出）。若返回非 None，
        agent_loop 会把它当新输入继续循环（强制不停）。
    """
    # 双层遍历历史消息：统计所有 tool_result 块的个数
    tool_count = sum(1 for m in messages
                     for b in (m.get("content") if isinstance(m.get("content"), list) else [])
                     if isinstance(b, dict) and b.get("type") == "tool_result")
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None

# 把 5 个钩子挂到 4 个事件上；同一事件可挂多个，按注册顺序触发
register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


# -- Agent 循环：结构与 s03 相同，但不再有写死的检查 --
# s03 的写法：if not check_permission(block): ...
# s04 的写法：if trigger_hooks("PreToolUse", block): ...

def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]
        # 模型本轮没有请求工具：先触发 Stop 钩子再决定是否退出
        if not tool_calls:
            # Stop 钩子若返回非 None（如一段提醒文本），可强行不让循环结束
            force = trigger_hooks("Stop", messages)
            if force:
                # 把钩子返回的内容伪装成 user 消息塞回历史，模型被迫继续干活
                messages.append({"role": "user", "content": force})
                continue
            # 钩子返回 None：正常退出，这是整个循环唯一的出口
            return

        results = []
        for block in tool_calls:
            # s04 的改动：用钩子取代写死的 check_permission()
            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": str(blocked)})
                continue

            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"

            # s04：后置钩子
            trigger_hooks("PostToolUse", block, output)  

            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("s04: Hooks - extension logic on hooks, loop stays clean")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            # \001/\002 告诉 Readline：这些 ANSI 转义符的显示宽度为零。
            query = input("\001\033[36m\002s04 >> \001\033[0m\002")
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
