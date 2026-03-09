from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Callable
from functools import wraps
import time

# 状态结构：输入文本、日志与消息
class LogState(TypedDict):
    input: str
    logs: List[str]
    messages: list

# 自定义日志装饰器，记录输入输出并附带执行时间
def with_logging(node_name: str):
    def decorator(func: Callable[[LogState], LogState]) -> Callable[[LogState], LogState]:
        @wraps(func)
        def wrapper(state: LogState) -> LogState:
            start_time = time.time()
            input_snapshot = state["input"]
            try:
                result = func(state)
                duration = round(time.time() - start_time, 3)
                state["logs"].append(f"[{node_name}] 输入：{input_snapshot},耗时：{duration}s, 输出：{result['input']}")
                return result
            except Exception as e:
                state["logs"].append(f"[{node_name}] 执行异常：{str(e)}")
                raise
        return wrapper
    return decorator

# 节点1:文本预处理
@with_logging("normalize")
def normalize(state: LogState) -> LogState:
    norm = state["input"].strip().lower()
    state["input"] = norm
    state["messages"].append({"role":"system", "content":f"标准化结果：{norm}"})
    return state

# 节点2:判断是否为命令句式
@with_logging("classify")
def classify_command(state: LogState) -> LogState:
    if state["input"].startswith("please") or state["input"].endswith("!"):
        state["messages"].append({"role":"system", "content":"识别为命令语句"})
    else:
        state["messages"].append({"role":"system", "content":"非命令语句"})
    return state

# 构建图结构
builder = StateGraph(LogState)
builder.add_node("normalize", normalize)
builder.add_node("classify", classify_command)
builder.set_entry_point("normalize")

builder.add_edge("normalize","classify")
builder.add_edge("classify", END)

graph = builder.compile()

# 测试执行
input_text = " Please send me the report. "
initial_state = {"input":input_text, "logs":[], "messages":[]}
result = graph.invoke(initial_state)

# 输出日志与交互信息
for msg in result["messages"]:
    print(f"{msg['role']}: {msg['content']}")

print("日志追踪：")
for log in result["logs"]:
    print(log)