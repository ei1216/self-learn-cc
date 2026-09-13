#!/usr/bin/env python3
"""
s01_agent_loop.py - Agent 循环

AI 编程助手的全部秘密浓缩为一个模式：

    while True:
        response = LLM(messages, tools)
        if response contains no tool_use:
            break
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (循环继续)

这就是核心循环：把工具结果回传给模型，
直到模型自己决定停下。后续章节会在它周围
加入策略、钩子和生命周期控制。

messages 列表随循环的演变（理解本模式的关键）：

    [user: 提问]                                    <- 入口追加
    [user, assistant(tool_use)]                     <- 第 1 轮模型响应
    [user, assistant(tool_use), user(tool_result)]  <- 工具结果伪装成 user 消息回传
    [..., assistant("最终回答")]                     <- 最后一轮：纯文本，循环结束

两个 API 硬性约定：
1. assistant 回复（含 tool_use 块）必须原样追加进 messages，
   否则 tool_use 与 tool_result 配不上对，下一轮请求直接报错。
2. tool_result 以 user 角色回传，tool_use_id 必须与请求一一对应。

用法：
    pip install anthropic python-dotenv
    ANTHROPIC_API_KEY=... python s01_agent_loop/code.py
"""

import os
import subprocess

try:
    import readline
    # #143 针对 macOS libedit 的 UTF-8 退格修复
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

# 从 .env 文件加载环境变量；override=True 表示 .env 里的值覆盖已有环境变量
load_dotenv(override=True)

# 配置了自定义网关（ANTHROPIC_BASE_URL）时，删掉可能冲突的 AUTH_TOKEN，
# 避免客户端同时携带两种凭证导致鉴权失败
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# API 客户端；模型名从环境变量 MODEL_ID 读取（缺失会直接 KeyError）
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# 系统提示词：告诉模型"你是谁、在哪、用什么工具、怎么做事"——只行动，不解释
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# -- 工具定义：只用 bash --
# 这是给模型看的"工具说明书"：name/description 帮模型决定何时调用，
# input_schema 用 JSON Schema 约束参数格式。模型只能"请求"调用，
# 真正执行权始终在本地代码（run_bash）手里。
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


# -- 工具执行 --
def run_bash(command: str) -> str:
    """在本地 shell 中执行一条命令，返回合并后的文本输出。

    Args:
        command: 要执行的命令字符串（来自模型的 tool_use 参数）。

    Returns:
        stdout 与 stderr 拼接后的文本（最长 50000 字符）；
        无输出、被拦截或出错时返回对应的提示字符串。
        本函数不抛异常：错误信息本身会作为"工具结果"喂回模型，
        让它看到失败原因后自行调整。
    """
    # 危险命令黑名单：子串匹配，命中直接拒绝执行
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # shell=True: 经系统 shell 执行（Windows 下是 cmd.exe），dir、2>nul 等语法因此可用
        # capture_output: 捕获 stdout/stderr；text=True: 解码成 str
        # errors="replace": 解不了的字符替换为占位符，不崩溃；timeout: 120 秒强制超时
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, errors="replace", timeout=120)
        # stdout 在前、stderr 在后直接拼接（无分隔符），再去掉首尾空白
        out = (r.stdout + r.stderr).strip()
        # 截断到 50000 字符，防止一条命令的输出撑爆模型上下文；空输出用字面提示占位
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# -- 核心模式：一个不断调用工具直到模型停止的 while 循环 --
def agent_loop(messages: list):
    """Agent 的心脏：请求模型 -> 执行工具 -> 回传结果，周而复始。

    Args:
        messages: 对话历史列表，会被就地追加 assistant/tool_result 消息；
                  循环结束时，最后一条就是模型的最终文字回复。
    """
    # 循环没有轮数上限，唯一的正常出口是下面"模型不再请求工具"的 return
    while True:
        # 把完整历史 + 工具清单发给模型；它本轮要么回文本，要么回 tool_use 请求
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        # 追加 assistant 回合消息
        # 必须原样入史（包括其中的 tool_use 块），否则 tool_use 与
        # 后面的 tool_result 配不上对，下一轮请求会被 API 拒绝
        messages.append({"role": "assistant", "content": response.content})

        # 如果模型没有调用工具，说明任务完成
        # 从回复的内容块中筛出 tool_use（模型可能同一轮里又说话又要用工具）
        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]

        # 本轮只有纯文本：模型认为任务完成，循环到此结束
        if not tool_calls:
            return

        # 执行每个工具调用，收集结果
        # 结果必须以 user 角色回传，tool_use_id 与请求一一配对（API 硬性要求）
        results = []
        for block in tool_calls:
            # 终端黄色回显模型要执行的命令，让人能实时看到 Agent 的动作
            print(f"\033[33m$ {block.input['command']}\033[0m")
            output = run_bash(block.input["command"])
            # 本地只预览前 200 字符；完整输出会通过 results 发给模型
            print(output[:200])
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })

        # 把工具结果回传，循环继续
        # 伪装成一条 user 消息追加进历史；回到 while 顶部再次请求模型时，
        # 它就能"看到"命令输出，决定继续用工具还是给出最终回答
        messages.append({"role": "user", "content": results})


# -- 程序入口 --
if __name__ == "__main__":
    print("s01: Agent Loop")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    # 对话历史：整个会话共用一份，agent_loop 会就地往里追加消息，
    # 所以模型每一轮都能看到之前所有问答
    history = []
    while True:
        try:
            # \001/\002 告诉 Readline：这些 ANSI 转义符的显示宽度为零。
            # 不加的话，用退格/方向键编辑输入行时光标位置会算错
            query = input("\001\033[36m\002s01 >> \001\033[0m\002")
        except (EOFError, KeyboardInterrupt):
            break
        # q / exit / 空输入退出，其他输入作为新一轮提问交给 Agent
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        # 打印模型最终的文本回复
        # agent_loop 返回时，history 末尾就是 assistant 的最终回复（纯文本块）
        response_content = history[-1]["content"]
        # 最终回复的 content 是内容块列表；只有是列表才逐块检查（防止意外类型）
        if isinstance(response_content, list):
            for block in response_content:
                # 只打印 text 类型块（模型说的话），跳过 tool_use 等其他块；
                # getattr 带 None 默认值，块对象缺 type 属性时也不会崩溃
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()
