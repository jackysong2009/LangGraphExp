from typing import TypedDict, List, Any, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool


# 定义状态结构
class FlowState(TypedDict):
    input: Any
    logs: List[str]
    messages: Annotated[list, add_messages]


# 工具函数1：文本规范化
@tool
def normalize_text(text: str) -> str:
    """Normalize text by stripping whitespace and converting to lowercase."""
    return text.strip().lower()


# 工具函数2：关键词提取
@tool
def extract_keywords(text: str) -> List[str]:
    """Extract words longer than 3 characters from text."""
    return [word for word in text.split() if len(word) > 3]


# 工具函数3：关键词计数
@tool
def count_keywords(keywords: List[str]) -> int:
    """Count unique keywords."""
    return len(set(keywords))


# 节点1：标准化文本
def node_normalize(state: FlowState) -> dict:
    norm = normalize_text.invoke(state["input"])
    return {
        "input": norm,
        "logs": state["logs"] + ["normalize"],
        "messages": [
            {"role": "system", "content": f"标准化结果：{norm}"}
        ],
    }


# 节点2：提取关键词
def node_extract(state: FlowState) -> dict:
    keywords = extract_keywords.invoke(state["input"])
    return {
        "input": keywords,
        "logs": state["logs"] + ["extract"],
        "messages": [
            {"role": "system", "content": f"关键词：{keywords}"}
        ],
    }


# 节点3：关键词计数
def node_count(state: FlowState) -> dict:
    kw_list = state["input"]
    count = count_keywords.invoke({"keywords": kw_list})
    return {
        "input": count,
        "logs": state["logs"] + ["count"],
        "messages": [
            {"role": "system", "content": f"关键词总数：{count}"}
        ],
    }


# 构建图结构
builder = StateGraph(FlowState)

builder.add_node("normalize", node_normalize)
builder.add_node("extract", node_extract)
builder.add_node("count", node_count)

builder.set_entry_point("normalize")
builder.add_edge("normalize", "extract")
builder.add_edge("extract", "count")
builder.add_edge("count", END)

graph = builder.compile()


# 执行测试流程
input_text = " LangChain integrates tools and agents into reasoning workflows. "
initial_state = {
    "input": input_text,
    "logs": [],
    "messages": [],
}

result = graph.invoke(initial_state)


# 打印结果日志
for msg in result["messages"]:
    print(f"{msg.type}: {msg.content}")

print("最终结果：", result["input"])
print("调用序列：", result["logs"])