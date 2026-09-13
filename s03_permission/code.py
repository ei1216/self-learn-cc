#!/usr/bin/env python3
"""
s03_permission.py - 权限系统

在工具执行之前插入的三道关卡：

    关卡 1：硬拒绝清单（rm -rf /、sudo 等，一律禁止）
    关卡 2：规则匹配（写到工作区之外？破坏性命令？）
    关卡 3：用户审批（暂停下来，等待用户确认）

    +----------+      +-------+      +--------------+      +---------------+
    |   User   | ---> |  LLM  | ---> | Permission   | ---> | Tool Dispatch |
    |  prompt  |      |       |      | 1. deny list |      | execute       |
    +----------+      +---+---+      | 2. rules     |      +-------+-------+
                          ^          | 3. approval  |              |
                          |          +------+-------+              |
                          |                 | deny                 |
                          |                 v                      v
                          |          +-------------------------------+
                          +----------+ tool_result: denied or output |
                                     +-------------------------------+

agent 循环里只新增了一行代码：

    if not check_permission(block):
        continue

在 s02（多工具版）基础上构建。用法：

    python s03_permission/code.py
    需要：pip install anthropic python-dotenv，并在 .env 中配置 ANTHROPIC_API_KEY
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

SYSTEM = f"You are a coding agent at {WORKDIR}. All destructive operations require user approval."


# -- 来自 s02：工具实现 --

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


# -- 来自 s02（未改动）：工具定义与分发 --

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


# -- s03 新增：三道关卡的权限流水线 --

# 关卡 1：硬拒绝清单 —— 一律禁止
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"]

def check_deny_list(command: str) -> str | None:
    """关卡 1：检查命令是否命中硬拒绝清单。

    Args:
        command: 模型要执行的 bash 命令字符串。

    Returns:
        命中时返回阻止原因；未命中返回 None（放行）。
    """
    # 子串匹配：命令里包含清单中任意一条即拦截
    for pattern in DENY_LIST:
        if pattern in command:
            return f"Blocked: '{pattern}' is on the deny list"
    # 全部未命中：放行
    return None


# 关卡 2：规则匹配 —— 视上下文而定的检查
# 正则含义：只匹配"作为独立命令出现"的 rm 或 del
# (?i)             忽略大小写
# (?:^|[;&|()\n])  命令前必须是行首或分隔符（排除 firm、worm 这类单词误伤）
# \s*              命令前允许空白
# (?:rm|del)       命令本体
# (?=\s|$|[;&|()]) 命令后必须是空白/行尾/分隔符（排除 rmdir、delta）
DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)


def contains_destructive_command(command: str) -> bool:
    """判断命令中是否出现独立的 rm / del 删除命令。

    Args:
        command: bash 命令字符串。

    Returns:
        True 表示疑似破坏性删除命令，需要升级审批。
    """
    return bool(DESTRUCTIVE_COMMAND_WORD.search(command))


# 规则表：每条规则 = 适用工具 + 检查函数（lambda）+ 提示消息
# 命中规则不会直接拒绝，而是交给关卡 3 请用户裁决
PERMISSION_RULES = [
    # 规则一：文件类工具的路径解析后逃出工作区 -> 判为越界写
    {"tools": ["read_file", "write_file", "edit_file"],
     "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
     "message": "Writing outside workspace"},
    # 规则二：bash 含独立 rm/del，或出现 rm / > /etc/ / chmod 777 关键字
    {"tools": ["bash"],
     "check": lambda args: contains_destructive_command(args.get("command", "")) or
     any(kw in args.get("command", "") for kw in ["rm ", "> /etc/", "chmod 777"]),
     "message": "Potentially destructive command"},
]

def check_rules(tool_name: str, args: dict) -> str | None:
    """关卡 2：按规则表逐条匹配本次工具调用。

    Args:
        tool_name: 工具名（如 "bash"、"write_file"）。
        args: 模型传给工具的参数字典。

    Returns:
        命中规则时返回提示消息（升级给关卡 3 审批）；未命中返回 None。
    """ 
    for rule in PERMISSION_RULES:
        # 工具名在规则适用范围内，且检查函数判定命中
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None


# 关卡 3：用户审批 —— 命中规则后等待用户确认
def ask_user(tool_name: str, args: dict, reason: str) -> str:
    """关卡 3：在终端展示风险详情，请用户人工裁决。

    Args:
        tool_name: 工具名。
        args: 工具参数（完整展示给用户看）。
        reason: 关卡 2 给出的命中原因。

    Returns:
        "allow"（输入 y/yes）；其他任何输入（含直接回车）均为 "deny"。
    """
    # 黄色高亮提示命中原因，并完整展示将要执行的工具与参数
    print(f"\n\033[33m[permission] {reason}\033[0m")
    print(f"   Tool: {tool_name}({args})")
    # [y/N] 中的大写 N 表示默认值：直接回车 = 拒绝
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"


# 流水线：三道关卡依次串联
def check_permission(block) -> bool:
    """权限流水线：关卡 1 -> 关卡 2 -> 关卡 3 依次过检。

    Args:
        block: 模型的 tool_use 块（含工具名 block.name 与参数 block.input）。

    Returns:
        True 放行执行；False 拒绝（agent_loop 会把 "Permission denied."
        作为工具结果回传给模型）。
    """
    # 关卡 1 只针对 bash：命中硬拒绝清单 -> 红色提示，直接拦截，不询问
    if block.name == "bash":
        reason = check_deny_list(block.input.get("command", ""))
        if reason:
            print(f"\n\033[31m[blocked] {reason}\033[0m")
            return False
    # 关卡 2：命中规则不直接拒绝，升级到关卡 3 请用户裁决
    reason = check_rules(block.name, block.input)
    if reason:
        decision = ask_user(block.name, block.input, reason)
        if decision == "deny":
            return False
    # 三道关卡全部通过：放行
    return True


# -- Agent 循环：与 s02 相同，只是插入了 check_permission() --

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
            print(f"\033[36m> {block.name}\033[0m")

            # s03 的改动：执行前先过一遍权限流水线
            if not check_permission(block):
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": "Permission denied."})
                continue

            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            print(str(output)[:200])
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("s03: Permission")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            # \001/\002 告诉 Readline：这些 ANSI 转义符的显示宽度为零。
            query = input("\001\033[36m\002s03 >> \001\033[0m\002")
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
