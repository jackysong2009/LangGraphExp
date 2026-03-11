from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda

# 必须与主程序保持一致的状态定义
class WorkflowState(TypedDict):
    input: str
    output: str

def build_graph() -> StateGraph:
    """构建并返回数学处理子图"""
    graph = StateGraph(WorkflowState)

    def math_process(state: WorkflowState) -> WorkflowState:
        # 尝试进行数学计算（例如求平方）
        try:
            num = int(state["input"])
            result = f"[Math插件] 计算成功，{num} 的平方是 {num * num}"
        except ValueError:
            result = f"[Math插件] 计算跳过，'{state['input']}' 不是有效的数字"
        
        # 累加之前的输出记录
        current_output = state["output"]
        new_output = f"{current_output}\n -> {result}" if current_output else result
        
        return {"input": state["input"], "output": new_output}
    
    graph.add_node("math_node", RunnableLambda(math_process))
    graph.set_entry_point("math_node")
    graph.set_finish_point("math_node")
    
    return graph
