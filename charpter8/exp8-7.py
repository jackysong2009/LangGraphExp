from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Union, Callable, Dict, Any

# 定义图状态类型
class WorkflowState(TypedDict):
    input: str
    summary: Union[str, None]
    decision: Union[str, None]
    output: Union[str, None]

# 这是我们准备使用的几个处理函数
def summarize_node(state: WorkflowState) -> Dict:
    text = state["input"]
    summary = text[:50] +"..." if len(text)>50 else text
    return {"summary": summary}

def decision_node(state: WorkflowState) -> Dict:
    if "error" in state["summary"].lower():
        return {"decision": "handler_error"}
    return {"decision": "generate_output"}

def error_handler_node(state: WorkflowState) -> Dict:
    return {"output": f"错误处理完成：{state['summary']}"}

def generate_output_node(state: WorkflowState) -> Dict:
    return {"output": f"结果输出：{state['summary']}"}

# 节点函数注册表
FUNCTION_REGISTRY: Dict[str, Callable] = {
    "summarize": summarize_node,
    "decide": decision_node,
    "error_handler": error_handler_node,
    "output_generator": generate_output_node
}

# DSL定义：用于描述节点结构、链接关系与调度逻辑
graph_dsl = {
    "entry": "summarize",
    "nodes": {
        "summarize": {
            "func": "summarize",
            "next": "decide"
        },
        "decide": {
            "func": "decide",
            "branch": {
                "handler_error": "error_handler",
                "generate_output": "output_generator"
            }
        },
        "error_handler": {
            "func": "error_handler",
            "next": "END"
        },
        "output_generator": {
            "func": "output_generator",
            "next": "END"
        }
    }
}

# 构建器：根据DSL自动构建LangGraph图
def build_graph_from_dsl(dsl: Dict[str, Any]) -> Callable:
    graph = StateGraph(WorkflowState)
    node_objects = {}

    # 注册所有节点
    for node_name, node_info in dsl["nodes"].items():
        func = FUNCTION_REGISTRY[node_info["func"]]
        runnable = RunnableLambda(func)
        graph.add_node(node_name, runnable)
        node_objects[node_name] = node_info

    # 配置连接关系
    for node_name, node_info in node_objects.items():
        if "next" in node_info:
            next_node = node_info["next"]
            if next_node == "END":
                graph.add_edge(node_name, END)
            else:
                graph.add_edge(node_name, next_node)
        elif "branch" in node_info:
            graph.add_conditional_edges(
                node_name,
                lambda state: state["decision"],
                node_info["branch"]
            )
    
    graph.set_entry_point(dsl["entry"])
    return graph.compile()

# 构建图
workflow = build_graph_from_dsl(graph_dsl)

# 执行图
result = workflow.invoke({"input": "这是一个包含error关键字的一场摘要测试文本。"})

# 输出结果
print(result)