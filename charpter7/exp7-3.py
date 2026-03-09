from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, List

# 定义状态结构，包含当前角色，历史消息与任务信息
class RoleState(TypedDict):
    active_role: Literal["user", "assistant"]
    messages: List[str]
    topic: str
    count: int

# 用户节点：生成用户发言并迁移控制权
def user_node(state: RoleState) -> RoleState:
    content = f"我想了解关于{state['topic']}的更多信息."
    state["messages"].append(f"User:{content}")
    state["active_role"] = "assistant"  #控制权转移
    state["count"] += 1
    return state

# 助理节点：响应用户提问并迁移控制权
def assistant_node(state: RoleState) -> RoleState:
    reply = f"{state['topic']}是一个值得探讨的重要主题，这里是一些相关信息。"
    state["messages"].append(f"Assistant:{reply}")
    state["active_role"] = "user"   #控制权迁移
    state["count"] += 1
    return state

# 控制调整函数：根据当前角色决定跳转目标
def next_role(state: RoleState) -> Literal["user", "assistant", "__end__"]:
    if state["count"] >= 4:
        return "__end__"
    return state["active_role"]

# 构建Langgraph图结构
builder = StateGraph(RoleState)
builder.set_entry_point("user")
builder.add_node("user", user_node)
builder.add_node("assistant", assistant_node)

# 状态迁移规则：根据active_role切换执行节点
builder.add_conditional_edges(
    "user",
    next_role,
    {"assistant": "assistant", "__end__": END}
)

builder.add_conditional_edges(
    "assistant",
    next_role,
    {"user": "user", "__end__": END}
)

graph = builder.compile()

# 测试运行
initial_state = {"active_role": "user", "messages": [], "topic": "LangGraph", "count": 0}
result = graph.invoke(initial_state)

# 输出对话记录
for line in result["messages"]:
    print(line)