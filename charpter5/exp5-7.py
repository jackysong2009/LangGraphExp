from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from typing import TypedDict, Annotated, Literal

# 定义状态结构，包含评论内容、评分值、交互记录
class ReviewState(TypedDict):
    review: str
    score: int
    messages: Annotated[list[BaseMessage], add_messages]

# 工具函数：对评论内容进行评分，简单示例根据关键词打分
@tool(description="根据评论内容评分")
def score_review(review: str) -> int:
    
    if "差" in review or "不好" in review:
        return 20
    elif "一般" in review:
        return 60
    else:
        return 85
    
# 定义节点函数1: 接收评论输入并记录
def receive_review(state: ReviewState) -> ReviewState:
    print(f"收到评论: {state['review']}")
    return {"messages": [HumanMessage(content=state["review"])]}

# 定义节点函数2: 使用工具进行评论评分并写入状态
def evaluate_review(state: ReviewState) -> ReviewState:
    score = score_review.invoke(state["review"])
    print(f"评论评分: {score}")
    return {"score": score, "messages": [SystemMessage(content=f"评论评分为: {score}")]}

# 定义节点函数3: 根据评分值进行分支处理
def route_by_score(state: ReviewState) -> Literal["negative", "neutral", "positive"]:
    if state["score"] < 40:
        return "negative"
    elif state["score"] < 70:
        return "neutral"
    else:
        return "positive"

# 各分支处理节点（负面评论、中性评论、正面评论）
def handle_negative(state: ReviewState) -> ReviewState:
    print("处理负面评论")
    return {"messages": [AIMessage(content="我们很抱歉听到您的不满，我们会努力改进。")]}

def handle_neutral(state: ReviewState) -> ReviewState:
    print("处理中性评论")
    return {"messages": [AIMessage(content="感谢您的反馈，我们会继续努力。")]}

def handle_positive(state: ReviewState) -> ReviewState:
    print("处理正面评论")
    return {"messages": [AIMessage(content="非常感谢您的支持！")]}

# 构建状态图
builder = StateGraph(ReviewState)
builder.add_node("receive_review", receive_review)
builder.add_node("evaluate_review", evaluate_review)
builder.add_node("handle_negative", handle_negative)
builder.add_node("handle_neutral", handle_neutral)
builder.add_node("handle_positive", handle_positive)

builder.set_entry_point("receive_review")
builder.add_edge("receive_review", "evaluate_review")  # receive_review完成后进入
builder.add_conditional_edges(
    "evaluate_review",
    route_by_score,
    {
        "negative": "handle_negative",
        "neutral": "handle_neutral",
        "positive": "handle_positive"
    }
)

# 各处理节点完成后流程结束
builder.add_edge("handle_negative", END)
builder.add_edge("handle_neutral", END)
builder.add_edge("handle_positive", END)

# 编译执行图
compiled = builder.compile()

# 执行图
input_text = "这个产品质量不错。"
initial_state = {"review": input_text, "score": 0, "messages": []}
result = compiled.invoke(initial_state) 

# 打印输出消息
for msg in result["messages"]:
    print(f"{msg.__class__.__name__}: {msg.content}")
