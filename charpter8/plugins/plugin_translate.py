from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda

class WorkflowState(TypedDict):
    input: str
    output: str

def build_graph() -> StateGraph:
    """构建并返回翻译处理子图"""
    graph = StateGraph(WorkflowState)

    def translate_process(state: WorkflowState) -> WorkflowState:
        # 模拟翻译功能
        result = f"[Translate插件] 已将 '{state['input']}' 模拟翻译为目标语言"
        
        # 累加之前的输出记录
        current_output = state["output"]
        new_output = f"{current_output}\n -> {result}" if current_output else result
        
        return {"input": state["input"], "output": new_output}
    
    graph.add_node("translate_node", RunnableLambda(translate_process))
    graph.set_entry_point("translate_node")
    graph.set_finish_point("translate_node")
    
    return graph
