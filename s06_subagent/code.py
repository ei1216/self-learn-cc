#!/usr/bin/env python3
"""
s06_subagent.py - Subagents（子Agent）

task 工具会以一份全新的消息列表运行第二个 agent 循环。两个循环
共享同一个工作目录，但只有最终文本会返回给父对话。

    Parent agent                    Subagent
    +------------------+            +------------------+
    | messages=[...]   |            | messages=[prompt]|
    |                  |   task     |                  |
    | tool: task       | ---------> | own agent loop   |
    |                  |            | base tools only  |
    | tool_result      | <--------- | final text       |
    +------------------+            +------------------+

子Agent没有 task 工具，因此无法再次向下委派任务。
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

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Use task for focused exploration or a self-contained subtask."
)
SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the given task, then return a concise final answer."
)


# -- 基础工具 --

def run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, errors="replace", timeout=120,
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
    import glob
    try:
        matches = sorted({
            match for match in glob.glob(
                pattern, root_dir=WORKDIR, recursive=True)
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR)
        })
        shown = matches[:200]
        if len(matches) > 200:
            shown.append("... (more matches omitted; narrow the pattern)")
        return "\n".join(shown) if shown else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


BASE_TOOLS = [
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

BASE_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}


# -- Hook 系统 --

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
    """PreToolUse：拦截被禁止的操作，并就有风险的操作询问用户。"""
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
    """PreToolUse：记录每次工具调用。"""
    args_preview = str(list(block.input.values())[:2])[:60]
    print(f"\033[90m[HOOK] {block.name}({args_preview})\033[0m")
    return None


def large_output_hook(block, output):
    """PostToolUse：对大输出发出警告。"""
    if len(str(output)) > 100000:
        print(f"\033[33m[HOOK] Large output from {block.name}: {len(str(output))} chars\033[0m")
    return None


def context_inject_hook(query: str):
    """UserPromptSubmit：记录工作目录。"""
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None


def summary_hook(messages: list):
    """Stop：打印此消息列表中工具结果的数量。"""
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


def execute_tool(block, handlers: dict) -> str:
    """执行单个工具调用：先过 PreToolUse hook，再查找并调用处理函数，最后过 PostToolUse hook。

    父 agent 和子Agent共用此入口，通过 handlers 参数区分各自的工具集：
    父Agent传入 TOOL_HANDLERS（含 task），子Agent传入 SUB_HANDLERS（仅基础工具）。

    Args:
        block: 模型返回的 tool_use 块，包含工具名和输入参数。
        handlers: 工具名 -> 处理函数 的映射字典。

    Returns:
        工具输出、hook 拦截消息或错误信息（统一转为字符串）。
    """
    # PreToolUse hook：权限检查，返回非 None 表示拦截，直接把拦截消息当作工具结果
    blocked = trigger_hooks("PreToolUse", block)
    if blocked:
        return str(blocked)

    # 按工具名查找处理函数；找不到则返回 Unknown 提示
    handler = handlers.get(block.name)
    try:
        output = handler(**block.input) if handler else f"Unknown: {block.name}"
    except Exception as e:
        # 处理函数抛异常时不中断循环，把错误信息作为工具结果返回给模型
        output = f"Error: {e}"

    # PostToolUse hook：对工具结果做后处理（如大输出警告）
    trigger_hooks("PostToolUse", block, output)
    return str(output)


# -- s06 新增：使用全新消息列表的嵌套 agent 循环 --


def extract_text(content) -> str:
    """从模型回复的 content 块列表中提取所有文本块并拼接。

    子Agent返回给父对话的只有最终文本，中间的工具调用过程全部丢弃。

    Args:
        content: 模型回复的 content（块列表或其他类型）。

    Returns:
        拼接后的文本字符串。
    """
    # content 不是块列表时直接转为字符串返回
    if not isinstance(content, list):
        return str(content)
    # 只保留 type == "text" 的块，取出其中的 text 字段并用换行拼接
    return "\n".join(
        getattr(block, "text", "")
        for block in content
        if getattr(block, "type", None) == "text"
    )

# 子Agent的工具集：只复制基础工具，刻意不含 task 工具，
# 因此子Agent无法再次委派任务，避免无限递归
SUB_TOOLS = list(BASE_TOOLS)
SUB_HANDLERS = dict(BASE_HANDLERS)

def run_subagent(prompt: str) -> str:
    """task 工具的实现：以全新消息列表运行一个独立的子 agent 循环。

    与父循环的区别：
    - messages 从零开始，只有父Agent传入的 prompt，不带父对话历史；
    - 使用 SUB_SYSTEM 系统提示和 SUB_TOOLS 工具集（没有 task）；
    - 最多循环 30 轮，防止子Agent无限运行；
    - 只有最终文本返回给父对话，中间过程不污染父上下文。

    Args:
        prompt: 父Agent交给子Agent的任务描述。

    Returns:
        子Agent的最终文本回答，或超时提示。
    """
    print("\n\033[35m[Subagent started]\033[0m")
    # 全新消息列表：只包含任务描述，与父对话完全隔离
    messages = [{"role": "user", "content": prompt}]

    # 硬性上限 30 轮，防止子Agent陷入死循环
    for _ in range(30):
        # 调用 LLM：使用子Agent专属的系统提示 和 工具集
        response = client.messages.create(
            model=MODEL,
            system=SUB_SYSTEM,
            messages=messages,
            tools=SUB_TOOLS,
            max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        # 提取本轮的工具调用块
        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]
        # 没有工具调用：子Agent认为任务完成，准备返回最终文本
        if not tool_calls:
            # Stop hook 可以强制继续（返回非 None 时）
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            print("\033[35m[Subagent done]\033[0m")
            # 只把最终文本返回给父对话，作为 task 工具的 tool_result
            return extract_text(response.content) or "(no summary)"

        # 执行本轮所有工具调用，结果回填到子Agent自己的消息列表
        results = []
        for block in tool_calls:
            # 注意：传入 SUB_HANDLERS，子Agent只能用基础工具
            output = execute_tool(block, SUB_HANDLERS)
            # 在终端实时显示子Agent的工具调用（截断到 100 字符）
            print(f"  \033[90m[sub] {block.name}: {output[:100]}\033[0m")
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })
        messages.append({"role": "user", "content": results})

    # 跑满 30 轮仍未给出最终答案，放弃并返回提示
    print("\033[35m[Subagent stopped]\033[0m")
    return "Subagent stopped after 30 turns without a final answer."


# task 工具的 schema：只有一个 prompt 参数（父Agent交给子Agent的任务描述）
TASK_TOOL = {
    "name": "task",
    "description": "Run a subagent with fresh conversation context and return its final text.",
    "input_schema": {
        "type": "object",
        "properties": {"prompt": {"type": "string", "minLength": 1}},
        "required": ["prompt"],
    },
}

# 父Agent的工具集 = 基础工具 + task 工具；处理函数同理
# 子Agent的 SUB_TOOLS/SUB_HANDLERS 不含 task，形成单向委派：父 -> 子
TOOLS = [*BASE_TOOLS, TASK_TOOL]
TOOL_HANDLERS = {**BASE_HANDLERS, "task": run_subagent}


# -- 父 agent 循环 --

def agent_loop(messages: list):
    """父 agent 的主循环：请求模型 -> 执行工具 -> 回传结果，直到模型停下。

    与 run_subagent 的子循环共用同一套骨架，靠三组配置区分：
    系统提示（SYSTEM/SUB_SYSTEM）、工具集（TOOLS/SUB_TOOLS）、
    处理函数表（TOOL_HANDLERS/SUB_HANDLERS）。
    
    工具执行抽成了 execute_tool()，循环内不再内联钩子调用。

    Args:
        messages: 对话历史列表，会被就地追加 assistant/tool_result 消息。
    """
    while True:
        # 把完整历史 + 工具清单（含 task）发给模型；它本轮要么回文本，要么回 tool_use 请求
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        # assistant 回复必须原样入史（含 tool_use 块），否则与后面的 tool_result 配不上对
        messages.append({"role": "assistant", "content": response.content})

        # 从回复的内容块中筛出 tool_use 调用
        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]
        # 模型本轮没有请求工具：先问 Stop 钩子，再决定是否真正退出
        if not tool_calls:
            force = trigger_hooks("Stop", messages)
            if force:
                # 钩子返回非 None：当作新的 user 消息塞回历史，强制循环继续
                messages.append({"role": "user", "content": force})
                continue
            # 钩子放行：正常退出，这是父循环唯一的出口
            return

        # 执行本轮所有工具调用（含 task），结果以 user 角色回传
        results = []
        for block in tool_calls:
            # 统一入口：内部完成 PreToolUse 检查 -> 分发调用 -> PostToolUse 后处理；
            # 传 TOOL_HANDLERS，父 agent 因此能调用 task 工具去委派子Agent
            output = execute_tool(block, TOOL_HANDLERS)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })
        # 工具结果伪装成 user 消息追加进历史，下一轮模型就能"看到"执行结果
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("s06: Subagent - fresh messages, final text returns")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            # \001/\002 告诉 Readline ANSI 转义序列的显示宽度为 0。
            query = input("\001\033[36m\002s06 >> \001\033[0m\002")
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
