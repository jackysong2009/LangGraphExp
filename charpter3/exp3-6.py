from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from typing import TypedDict
import random

# 定义工作流状态结构
class LoopState(TypedDict):
    input_text: str
    output_text: str
    score: float
    retry_count: int

# 初始化语言模型
llm = ChatOllama(model="qwen3:8b", temperature=0, base_url="http://192.168.1.60:11434")

# 内容生成节点
def generate_summary(state: LoopState) -> LoopState:
    content = state["input_text"]
    response = llm.invoke(f"请根据以下内容生成摘要：\n{content}").content
    return {
        "input_text": content,
        "output_text": response,
        "score": 0.0,  # 初始评分
        "retry_count": state["retry_count"]
    }

# 重新评估节点，打分评分机制（或可用LLM评分）
def evaluate_summary(state: LoopState) -> LoopState:
    score = random.uniform(0.4, 1.0)  # 模拟评分，实际可用LLM进行评分
    print(f"评估摘要，当前评分：{score:.2f}")
    return {
        **state,
        "score": score,
        "retry_count": state["retry_count"] + 1
    }

# 路由函数：若分数过低且未超过3次重试，则自循环回生成节点
def review_decision(state: LoopState) -> str:
    if state["score"] < 0.7 and state["retry_count"] < 3:
        print("评分过低，重新生成摘要...")
        return "generate"
    else:
        print("摘要质量达标，结束流程。")
        return "end"
    
# 构建状态图
builder = StateGraph(LoopState)
builder.add_node("generate", generate_summary)
builder.add_node("evaluate", evaluate_summary)

# 添加跳转逻辑：generate -> evaluate -> generate（如果评分过低）或evaluate -> END（如果评分达标或重试次数过多）
builder.add_edge("generate", "evaluate")
builder.add_conditional_edges("evaluate", review_decision, {
    "generate": "generate",
    "end": END
})

# 设置入口节点
builder.set_entry_point("generate")
graph = builder.compile()

# 初始化状态，开始执行
initial_state = {
    "input_text": "Langgraph是一个高度模块化的框架，可用于构建复杂的语言模型工作流和条件编排工作流。",
    "output_text": "",
    "score": 0.0,
    "retry_count": 0
}
final_state = graph.invoke(initial_state)

print("最终得分：", final_state["score"])
print("最终摘要：", final_state["output_text"])
print("重试次数：", final_state["retry_count"])
