from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import asyncio

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 定义状态结构
class ParallelState(TypedDict):
    input: str
    summary: Optional[str]
    keywords: Optional[str]
    merged: Optional[str]

# 并发任务1：提取摘要
async def summarize(state: ParallelState) -> ParallelState:
    logging.info("摘要任务开始执行")
    content = state["input"]
    await asyncio.sleep(2)  # 模拟处理时间
    logging.info("摘要任务结束执行")
    return {"summary": f"摘要：{content[:20]}..."}

# 并发任务2：提取关键词
async def extract_keywords(state: ParallelState) -> ParallelState:
    logging.info("提取关键词任务开始执行")
    content = state["input"]
    await asyncio.sleep(1)  # 模拟处理时间
    words = [w.strip('.,!?') for w in content.lower().split()]  # 简单示例
    logging.info("提取关键词任务结束执行")
    return {"keywords":",".join(words[:5])}

# 同步屏障节点：合并摘要与关键词
def merge_results(state: ParallelState) -> ParallelState:
    if "summary" not in state or "keywords" not in state:
        return {}  # 还没齐，先不更新 merged
    combined = f"Summary: {state['summary']} | Keywords: {state['keywords']}"
    return {"merged": combined}

# 构建状态图
graph = StateGraph(ParallelState)
graph.add_node("summarize", summarize)
graph.add_node("extract_keywords", extract_keywords)
graph.add_node("merge", merge_results)

# 设置并发路径入口
# graph.set_entry_point("summarize")
# graph.add_edge("summarize", "merge")  # summarize完成后进入merge
# graph.add_edge("extract_keywords", "merge")  # extract_keywords完成后进入merge

# 从多个入口启动执行
graph.add_node("start", lambda state: state)  # 起始节点
graph.add_edge("start", "summarize")
graph.add_edge("start", "extract_keywords")

graph.add_edge("summarize", "merge")  # summarize完成后进入merge
graph.add_edge("extract_keywords", "merge")  # extract_keywords完成后进入merge

graph.set_entry_point("start")
graph.add_edge("merge", END)  # merge完成后流程结束

# 编译执行图
compiled = graph.compile()

# 执行图
input_data = {"input": "这是一个用于测试并发执行的示例文本，包含多个句子和关键词。"}
result = asyncio.run(compiled.ainvoke(input_data))

print("最终结果：", result)