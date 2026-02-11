from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from langchain_ollama import ChatOllama
import random

# 定义工作流状态结构
class JumpState(TypedDict):
    input_text: str
    output_text: str
    score: float
    retries: int
    route: Literal["generate", "review", "output"]

# 初始化语言模型
llm = ChatOllama(model="qwen3:8b", temperature=0.5, base_url="http://192.168.1.60:11434")

# 生成内容的节点
def generate_node(state: JumpState) -> JumpState:
    text = state["input_text"]
    response = llm.invoke(f"缩句：{text}").content
    return {
        **state,
        "output_text": response,
        "route": "review"
    }

# 评估生成摘要的质量节点，随机打分，分数过低则跳回generate节点重试
def review_node(state: JumpState) -> JumpState:
    score = round(random.uniform(0.3, 1.0), 2)  # 模拟评估打分
    print(f"评估得分: {score}")
    if score < 0.6 and state["retries"] < 2:
        # 分数过低，跳回生成节点重试
        return {
            **state,
            "score": score,
            "retries": state["retries"] + 1,
            "route": "generate"
        }
    else:
        # 分数合格，进入输出节点
        return {
            **state,
            "score": score,
            "route": "output"
        }
    
# 输出结果节点
def output_node(state: JumpState) -> JumpState:
    return {
        **state,
        "route": "end"
    }

# 路由控制函数，根据当前状态的route字段决定下一个节点
def jump_router(state: JumpState) -> str:
    return state["route"]

# 构建LangGraph状态图
builder = StateGraph(JumpState)
builder.add_node("generate", generate_node)
builder.add_node("review", review_node)
builder.add_node("output", output_node)

# 添加条件跳转控制逻辑
builder.add_conditional_edges("generate", jump_router, {"review": "review"})
builder.add_conditional_edges("review", jump_router, {"generate": "generate", "output": "output"})
builder.add_conditional_edges("output", jump_router, {"end": END})

# 设置入口节点
builder.set_entry_point("generate")
graph = builder.compile()

# 执行状态图
initial_state: JumpState = {
    "input_text": "要站在推进强国建设、民族复兴伟业的战略高度，立足客观条件，发挥比较优势，坚持稳中求进、梯度培育，推动我国未来产业发展不断取得新突破。",
    "output_text": "",
    "score": 0.0,
    "retries": 0,
    "route": "generate"
}
final_state = graph.invoke(initial_state)
print("最终内容输出:", final_state["output_text"])
print("最终评估得分:", final_state["score"])

