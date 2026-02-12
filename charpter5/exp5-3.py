from langgraph.graph import StateGraph, END
from typing import TypedDict

# 定义总状态结构
class GlobalState(TypedDict):
    input: str
    preprocessed: str
    analyzed: str
    result: str

# 子流程状态结构
class SubState(TypedDict):
    input: str
    preprocessed: str

# 子图逻辑：预处理节点
def clean_text(state: SubState) -> SubState:
    cleaned = state["input"].strip().lower()
    return {
        **state,
        "preprocessed": cleaned
    }

# 构建子图
subgraph = StateGraph(SubState)
subgraph.add_node("clean_text", clean_text)
subgraph.set_entry_point("clean_text")
subgraph.add_edge("clean_text", END)
compiled_subgraph = subgraph.compile()

# 主图分析节点
