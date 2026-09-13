#!/usr/bin/env python3
"""
s02_tool_use.py - 工具

s01 的 agent 循环完全不变。本课在它之上新增四个工具
和一张分发表：

    +----------+      +-------+      +--------------------------+
    |   User   | ---> |  LLM  | ---> | Tool Dispatch            |
    |  prompt  |      |       |      | bash       -> run_bash   |
    +----------+      +---+---+      | read_file  -> run_read   |
                          ^          | write_file -> run_write  |
                          |          | edit_file  -> run_edit   |
                          +----------+ glob       -> run_glob   |
                          tool_result+--------------------------+

  + 新增 run_read / run_write / run_edit / run_glob 四个工具函数
  + 用 TOOL_HANDLERS 分发表取代 s01 中写死的 run_bash 调用
  + 用 safe_path 把文件工具限制在工作区内部

核心洞察：循环保持不变；随章节增长的只是工具注册与分发。
"""

import os
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


# -- 来自 s01（未改动） --

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, errors="replace",
                           timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# -- s02 新增：四个工具 --

def safe_path(p: str) -> Path:
    """把模型给的路径解析为工作区内的绝对路径，越界则拒绝。

    Args:
        p: 模型传入的路径（通常是相对路径，如 "src/main.py"）。

    Returns:
        解析后的绝对 Path 对象。

    Raises:
        ValueError: 路径解析后逃出 WORKDIR（如 ".." 上跳、换盘符注入）。
    """
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_read(path: str, limit: int | None = None) -> str:
    """读取文件内容，可选只取前 limit 行。

    Args:
        path: 文件路径（相对工作区，经 safe_path 校验）。
        limit: 最多返回的行数；超出时截断并附加剩余行数提示。

    Returns:
        文件文本；读不到或越界时返回"Error: ..."字符串，不抛异常。
    """
    try:
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """把内容写入文件（整文件覆盖），父目录不存在时自动创建。

    Args:
        path: 目标文件路径（相对工作区，经 safe_path 校验）。
        content: 要写入的完整内容。

    Returns:
        成功时返回写入字节数的确认信息；失败时返回"Error: ..."。
    """
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """把文件中第一处 old_text 精确替换为 new_text。

    Args:
        path: 目标文件路径（相对工作区，经 safe_path 校验）。
        old_text: 要被替换的原文（必须完整精确匹配）。
        new_text: 替换后的新文本。

    Returns:
        成功时返回确认信息；找不到原文或失败时返回"Error: ..."。
    """
    try:
        file_path = safe_path(path)
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str) -> str:
    """按 glob 模式搜索文件，** 可递归匹配子目录。

    Args:
        pattern: glob 模式（如 "**/*.py"，相对工作区解析）。

    Returns:
        排序去重后的匹配路径（每行一个，最多 200 条）；
        无匹配返回"(no matches)"；失败时返回"Error: ..."。
    """
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


# -- s02 新增：工具定义（s01 仅 1 个工具，s02 共 5 个） --

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

# -- s02 新增：分发表（取代 s01 中写死的 run_bash 调用） --

TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}


# -- agent 循环保持 s01 的骨架不变；只有工具分发这一处变了 --
# s01 的写法：output = run_bash(block.input["command"])
# s02 的写法：output = TOOL_HANDLERS[block.name](**block.input)

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
        if not tool_calls:
            return

        results = []
        for block in tool_calls:
            print(f"\033[33m> {block.name}\033[0m")
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            print(str(output)[:200])
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("s02: Tool Use - four tools added to s01")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            # \001/\002 告诉 Readline：这些 ANSI 转义符的显示宽度为零。
            query = input("\001\033[36m\002s02 >> \001\033[0m\002")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(block.text)
        print()
