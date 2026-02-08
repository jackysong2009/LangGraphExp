from typing import TypedDict, Literal, Callable, List

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama


# =========================
# 1) 定义工作流状态结构
# =========================
class RouteState(TypedDict):
    user_input: str
    route: Literal["translate", "summarize", "unknown"]
    output: str


# =========================
# 2) 初始化语言模型
# =========================
llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
    base_url="http://192.168.1.60:11434"
)


# =========================
# 3) 路由判断（意图分类）
# =========================
def route_decision(state: RouteState) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个意图分类器。根据用户输入判断其意图是翻译还是总结。如果无法判断则返回'unknown'。"),
        ("human", "用户输入：{text}\n只输出分类标签（translate, summarize, unknown）。")
    ])
    chat = prompt | llm
    result = chat.invoke({"text": state["user_input"]}).content.strip().lower()

    if "translate" in result:
        return "translate"
    elif "summarize" in result:
        return "summarize"
    else:
        return "unknown"


# =========================
# 4) 工具函数本体
# =========================
def translate_tool(text: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个翻译助手，将输入文本翻译成英文。"),
        ("human", "请翻译以下内容：{text}")
    ])
    chat = prompt | llm
    return chat.invoke({"text": text}).content


def summarize_tool(text: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个总结助手，将输入文本进行简要总结。"),
        ("human", "请总结以下内容：{text}")
    ])
    chat = prompt | llm
    return chat.invoke({"text": text}).content


# =========================
# 5) 更优雅：工具工厂 + Executor工厂
# =========================
def make_tool(fn: Callable[[str], str], name: str, description: str) -> Tool:
    # 只在这里写 Tool.from_function，其他地方复用
    return Tool.from_function(fn, name=name, description=description)


def make_executor(single_tool: Tool, *, verbose: bool = True) -> AgentExecutor:
    # 统一 prompt（也可以按需传参定制）
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能助手，会根据用户需求调用合适的工具完成任务。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(
        llm,
        tools=[single_tool],      # 给 LLM 看
        prompt=agent_prompt,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=[single_tool],      # 给执行层用 —— 同一份对象，绝不重复定义
        verbose=verbose
    )
    return executor


# =========================
# 6) 构建工具 & Executors（复用同一份 Tool）
# =========================
translate_tool_obj = make_tool(
    translate_tool,
    name="translate_tool",
    description="用于将文本翻译成英文"
)
summarize_tool_obj = make_tool(
    summarize_tool,
    name="summarize_tool",
    description="用于对文本进行总结"
)

translate_executor = make_executor(translate_tool_obj, verbose=True)
summarize_executor = make_executor(summarize_tool_obj, verbose=True)


# =========================
# 7) 图节点
# =========================
def translate_node(state: RouteState) -> RouteState:
    result = translate_executor.invoke({"input": state["user_input"]})
    return {
        "user_input": state["user_input"],
        "route": "translate",
        "output": result["output"]
    }


def summarize_node(state: RouteState) -> RouteState:
    result = summarize_executor.invoke({"input": state["user_input"]})
    return {
        "user_input": state["user_input"],
        "route": "summarize",
        "output": result["output"]
    }


def unknown_node(state: RouteState) -> RouteState:
    return {
        "user_input": state["user_input"],
        "route": "unknown",
        "output": "无法识别用户意图。"
    }


# =========================
# 8) 构建LangGraph状态图
# =========================
builder = StateGraph(RouteState)

builder.add_node("router", lambda x: x)
builder.add_node("translate", translate_node)
builder.add_node("summarize", summarize_node)
builder.add_node("unknown", unknown_node)

builder.add_conditional_edges("router", route_decision, {
    "translate": "translate",
    "summarize": "summarize",
    "unknown": "unknown"
})

builder.set_entry_point("router")

# 终点（更通用/更稳的写法）
builder.add_edge("translate", END)
builder.add_edge("summarize", END)
builder.add_edge("unknown", END)

graph = builder.compile()


# =========================
# 9) 测试
# =========================
input_state: RouteState = {
    "user_input": "请帮我总结这句话：人工智能正在迅速发展，改变着各行各业的面貌。",
    "route": "unknown",
    "output": ""
}

final_state = graph.invoke(input_state)
print("分发路径:", final_state["route"])
print("智能体返回结果:", final_state["output"])