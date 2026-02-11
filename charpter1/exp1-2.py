from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

# 第一步：定义状态结构，确保输入输出一致性
class TaskState(TypedDict):
    user_input: str                 # 用户初始输入指令
    task_type: Optional[str]        # 系统识别的任务类型
    clarification_required: bool    # 是否需要追问澄清
    feedback: Optional[str]         # 系统生成的反馈内容

# 第二步：定义节点函数--任务类型识别器
def classify_task(state:TaskState) -> TaskState:
    input_text = state["user_input"]

    #简单基于关键词判断任务类型
    if "生成" in input_text:
        task_type = "文本生成"
    elif "总结" in input_text or "摘要" in input_text:
        task_type = "文本摘要"
    elif "翻译" in input_text:
        task_type = "机器翻译"
    else:
        task_type = "未知任务"
    
    # 如果无法识别清晰类型，则设置需要追问
    need_clarification = task_type == "未知任务"

    # 构建状态返回
    state["task_type"] = task_type
    state["clarification_required"] = need_clarification
    return state

# 第三步：定义节点函数--反馈响应器
def generate_feedback(state:TaskState) -> TaskState:
    task_type = state["task_type"]
    if state["clarification_required"]:
        feedback = "未能识别任务类型，请补充更多说明。"
    else:
        feedback = f"已识别任务类型为“{task_type}”，即将进入处理流程。"
    
    state["feedback"] = feedback
    return state

# 第四步：构建图结构并连接节点
builder = StateGraph(TaskState)
builder.add_node("分类", classify_task)
builder.add_node("反馈", generate_feedback)

# 添加边：从分类节点跳转到反馈节点
builder.set_entry_point("分类")
builder.add_edge("分类", "反馈")
builder.add_edge("反馈", END)

# 构建可调用的图对象
app = builder.compile()

# 第五步：准备初始状态，启动图执行
initial_state:TaskState = {
    "user_input": "你好，请帮我翻译这段英文内容",
    "task_type" : None,
    "clarification_required": False,
    "feedback": None
}

# 执行流程并返回最终状态
final_state = app.invoke(initial_state)
print("最终状态输出结果：", final_state)