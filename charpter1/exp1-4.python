from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

# 定义状态结构，包含用户文本、质量标记、分数与反馈字段
class ReviewState(TypedDict):
    submission: str
    quality: Optional[str]
    score: Optional[str]
    feedback: Optional[str]

# 节点1：文本预处理
def preprocess_node(state:ReviewState) -> ReviewState:
    # 去除空格和标点（简化处理）
    cleaned = state["submission"].strip().replace("。","").replace("，","")
    state["submission"] = cleaned
    return state

# 节点2：评分与质量判断
def evaluate_node(state:ReviewState) -> ReviewState:
    text = state["submission"]
    length = len(text)
    # 简化评分规则：文本长度>15认为合格，反之不合格
    score = min(length * 3, 100)
    quality = "合格" if length>15 else "不合格"
    state["score"] = score
    state["quality"] = quality
    return state

# 节点3：根据质量生成反馈
def feedback_node(state:ReviewState) -> ReviewState:
    if state["quality"] == "合格":
        state["feedback"] = f"文本通过，得分 {state['score']},可以进入后续流程。"
    else:
        state["feedback"] = f"文本内容较短，仅得分 {state['score']}，建议补充更多信息。"
    return state

# 构建StateGraph流程
builder = StateGraph(ReviewState)

# 添加三个节点
builder.add_node("预处理", preprocess_node)
builder.add_node("评估", evaluate_node)
builder.add_node("反馈", feedback_node)

# 设置起点
builder.set_entry_point("预处理")

# 顺序连接节点
builder.add_edge("预处理","评估")
builder.add_edge("评估","反馈")
builder.add_edge("反馈",END)

# 编译为可执行流程
app = builder.compile()
