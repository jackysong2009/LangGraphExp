from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

# 第一步：定义状态结构，确保输入输出一致性
class ClarifyState(TypedDict):
    user_input: str                     # 用户初始输入指令
    needs_clarification: bool           # 是否需要澄清
    task_description: Optional[str]     # 任务说明
    result: Optional[str]               # 结果

# 节点1：判断是否需要澄清（例如输入太短或缺乏动词）
def assess_input(state:ClarifyState) -> ClarifyState:
    text = state["user_input"]
    # 简单条件：输入长度不足或不含关键词
    unclear = len(text) < 10 or not any(kw in text for kw in ["生成", "分析", "翻译", "总结"])
    state["needs_clarification"] = unclear
    return state

# 节点2：澄清用户意图
def clarify_node(state:ClarifyState) -> ClarifyState:
    #生成提示要求用户重新说明
    state["result"] = "当前指令不明确，请补充具体操作意图。"
    return state

# 节点3：正式处理任务(此处简化为写入描述)
def process_task(state:ClarifyState) -> ClarifyState:
    state["task_description"] = f"任务确认：执行操作--{state['user_input']}"
    state["result"] = "已成功识别任务，开始处理。"
    return state

# 构建图结构
builder = StateGraph(ClarifyState)
builder.add_node("判断", assess_input)
builder.add_node("澄清", clarify_node)
builder.add_node("处理", process_task)

# 设置起始节点
builder.set_entry_point("判断")

# 添加条件跳转逻辑
def jump_to_clarify(state:ClarifyState) -> bool:
    return state["needs_clarification"]

def jump_to_process(state:ClarifyState) -> bool:
    return not state["needs_clarification"]

# 路由函数返回节点名
def route_clarification(state: ClarifyState) -> str:
    return "澄清" if state["needs_clarification"] else "处理"

# 条件边配置
builder.add_conditional_edges("判断", route_clarification)

# 终止流程
builder.add_edge("澄清", END)
builder.add_edge("处理", END)

# 编译图
app = builder.compile()

# 测试输入：意图不明确的情况
state1:ClarifyState = {
    "user_input": "请帮忙",  # 模糊指令
    "needs_clarification": False,
    "task_description": None,
    "result": None
}

# 测试输入：意图明确的情况
state2:ClarifyState = {
    "user_input": "请帮我翻译这句话how are you？",
    "needs_clarification": False,
    "task_description": None,
    "result": None
}

# 执行图流程
print("测试1（模糊输入）输出：\n")
print(app.invoke(state1))
print("测试2（明确输入）输出：\n")
print(app.invoke(state2))