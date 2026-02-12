from langgraph.graph import StateGraph, END
from typing import TypedDict
import asyncio

# 定义状态结构
class ParallelState(TypedDict):
    input: str
    summary: str
    keywords: str
    merged: str

# 并发任务1：提取摘要
async def summarize(state: ParallelState) -> ParallelState:
    content = state["input"]
    await asyncio.sleep(1)  # 模拟处理时间
    return {**state, "summary": f"摘要：{content[:20]}..."}

# 并发任务2：提取关键词
async def extract_keywords(state: ParallelState) -> ParallelState:
    content = state["input"]
    await asyncio.sleep(1)  # 模拟处理时间
    words = [w.strip('.,!?') for w in content.split()]  # 简单示例：提取长度大于4的词
    return {**state, "keywords":",".join(words[:5])}

# 同步屏障节点：合并摘要与关键词
def merge_results(state: ParallelState) -> ParallelState:
    combined = f"Summary: {state['summary']} | Keywords: {state['keywords']}"
    return {**state, "merged": combined}

# 构建状态图
graph = StateGraph(ParallelState)
graph.add_node("summarize", summarize)
graph.add_node("extract_keywords", extract_keywords)
graph.add_node("merge", merge_results)

# 设置并发路径入口
graph.set_entry_point("summarize")
graph.add_edge("summarize", "merge")  # summarize完成后进入merge
graph.add_edge("extract_keywords", "merge")  # extract_keywords完成后进入merge

# 从多个入口启动执行
graph.add_node("start", lambda state: state)  # 起始节点
graph.add_edge("start", "summarize")
graph.add_edge("start", "extract_keywords")

graph.set_entry_point("start")
graph.add_edge("merge", END)  # merge完成后流程结束

# 编译执行图
compiled = graph.compile()

# 执行图
input_data = {"input": "这是一个用于测试并发执行的示例文本，包含多个句子和关键词。"}
result = asyncio.run(compiled.ainvoke(input_data))

print("最终结果：", result)