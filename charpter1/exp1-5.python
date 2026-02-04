from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

# 定义状态结构，包含输入文本、输出文本、错误标志与异常信息
class SafeState(TypedDict):
    raw_input: Optional[str]
    reversed_text: Optional[str]
    has_error: bool
    error_message: Optional[str]

# 节点1：文本反转处理，带异常捕获
def reverse_text(state: SafeState) -> SafeState:
    try:
        text = state["raw_input"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("输入内容不能为空或非字符串类型")
        # 执行文本反转操作
        reversed_str = text[::-1]
        state["reversed_text"] = reversed_str
        state["has_error"] = False
        state["error_message"] = None
    except Exception as e:
        # 捕获异常并记录状态信息
        state["has_error"] = True
        state["reversed_text"] = None
        state["error_message"] = f"处理异常：{str(e)}"
    return state

# 节点2：基于状态内容生成最终报告
def result_node(state: SafeState) -> SafeState:
    if state["has_error"]:
        print("处理失败，错误信息：", state["error_message"])
    else:
        print("文本反转成功，结果：", state["reversed_text"])
    return state

# 构建图结构
builder = StateGraph(SafeState)

builder.add_node("处理", reverse_text)
builder.add_node("结果", result_node)

builder.set_entry_point("处理")
builder.add_edge("处理", "结果")
builder.add_edge("结果", END)

# 编译执行图
app = builder.compile()

# 测试输入1：正常输入
input1: SafeState = {
    "raw_input": "LangGraph状态图",
    "reversed_text": None,
    "has_error": False,
    "error_message": None
}

# 测试输入2：非法输入(空字符串)
input2: SafeState = {
    "raw_input": "",
    "reversed_text": None,
    "has_error": False,
    "error_message": None
}

# 执行流程
print("示例1（合法输入）")
print(app.invoke(input1))

print("示例2（非法输入）")
print(app.invoke(input2))