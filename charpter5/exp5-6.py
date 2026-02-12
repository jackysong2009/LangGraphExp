from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from typing import TypedDict, Literal, Annotated, Union

# 定义状态结构
class FlowState(TypedDict):
    input: str
    intent: str
    messages: Annotated[list[BaseMessage], add_messages]

# 定义一个简单的LangChain工具用于意图分类
@tool(description="根据用户输入的文本分类意图")
def classify_intent(text: str) -> str:
    if "账" in text or "余额" in text:
        return "account"
    elif "支付" in text or "扣款" in text:
        return "payment"
    else:
        return "other" 
    
# 定义节点函数1: 接收输入并记录
def receive_input(state: FlowState) -> FlowState:
    print(f"收到用户输入: {state['input']}")
    return {"messages": [HumanMessage(content=state["input"])]}

# 定义节点函数2: 使用工具进行意图分类
def detect_intent(state: FlowState) -> FlowState:
    intent = classify_intent.invoke(state["input"])
    print(f"检测到的意图: {intent}")
    return {
        "intent": intent,
        "messages": [SystemMessage(content=f"检测到的意图是: {intent}")]
    }

# 分支判断函数：根据state["intent"]的值路由至不同处理节点
def route_by_intent(state: FlowState) -> Literal["account", "payment", "other"]:
    return state["intent"]

# 各分支处理节点（账号问题、支付问题、其他问题）
def handle_account(state: FlowState) -> FlowState:
    print("处理账号相关问题")
    return {"messages": [AIMessage(content="正在处理账号相关问题...")]}

def handle_payment(state: FlowState) -> FlowState:
    print("处理支付相关问题")
    return {"messages": [AIMessage(content="正在处理支付相关问题...")]}  

def handle_other(state: FlowState) -> FlowState:
    print("处理其他问题")
    return {"messages": [AIMessage(content="正在处理其他问题...")]} 

# 构建状态图
builder = StateGraph(FlowState)
builder.add_node("receive_input", receive_input)
builder.add_node("detect_intent", detect_intent)
builder.add_node("handle_account", handle_account)
builder.add_node("handle_payment", handle_payment)
builder.add_node("handle_other", handle_other)

builder.set_entry_point("receive_input")
builder.add_edge("receive_input", "detect_intent")  # receive_input完成后进入

# 多条件分支逻辑定义
builder.add_conditional_edges("detect_intent", route_by_intent, {
    "account": "handle_account",
    "payment": "handle_payment",
    "other": "handle_other"
})

# 各处理节点完成后流程结束
builder.add_edge("handle_account", END)
builder.add_edge("handle_payment", END)
builder.add_edge("handle_other", END)

# 编译执行图
compiled = builder.compile()

# 执行图
input_text = "我想查询我的账户余额"
initial_state = {"input": input_text, "intent": "", "messages": []}
result = compiled.invoke(initial_state)

# 打印输出
for msg in result["messages"]:
    print(f"{msg.type}: {msg.content}")

