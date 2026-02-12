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
def analyze(state: GlobalState) -> GlobalState:
    text = state["preprocessed"]
    summary = f"分析结果：文本长度为{len(text)}, 开始：{text[:5]}..."
    return {
        **state,
        "analyzed": summary
    }

# 结果整理节点
def finalize(state: GlobalState) -> GlobalState:
    result = f"分析完成：{state['analyzed']}"
    return {
        **state,
        "result": result
    }

# 构建主图
main_graph = StateGraph(GlobalState)
main_graph.add_node("subgraph", compiled_subgraph)
main_graph.add_node("analyze", analyze)
main_graph.add_node("finalize", finalize)

main_graph.set_entry_point("subgraph")
main_graph.add_edge("subgraph", "analyze")  # 子图完成后进入
main_graph.add_edge("analyze", "finalize")  # analyze完成后进入finalize
main_graph.add_edge("finalize", END)  # finalize完成后流程结束

# 编译执行图
compiled_main = main_graph.compile()
result = compiled_main.invoke({"input": "  Hello LangGraph Subflow!   "})

print("使用子图管理子流程的输出结果：")
print(result)