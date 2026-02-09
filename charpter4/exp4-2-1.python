from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from langchain_ollama import ChatOllama
import asyncio

class StreamState(TypedDict):
    user_input: str
    stream_output: Optional[str]

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
    base_url="http://192.168.1.60:11434",
    streaming=True
)

async def llm_stream_node(state: StreamState) -> StreamState:
    content = state["user_input"]
    prompt = f"请根据以下内容生成摘要：\n{content}"

    buffer: list[str] = []

    print("流式生成中：", end="", flush=True)
    async for chunk in llm.astream(prompt):
        text = getattr(chunk, "content", "")
        if text:
            print(text, end="", flush=True)   # 实时输出
            buffer.append(text)
    print()  # 换行

    return {
        "user_input": content,
        "stream_output": "".join(buffer)
    }

builder = StateGraph(StreamState)
builder.add_node("stream", llm_stream_node)
builder.add_edge("stream", END)
builder.set_entry_point("stream")
graph = builder.compile()

async def main():
    initial_state: StreamState = {
        "user_input": "LangChain与LangGraph联合使用可以实现多节点、状态驱动的语言模型智能体工作流，适用于复杂任务的分布式调度与工具链管理。",
        "stream_output": None
    }
    final_state = await graph.ainvoke(initial_state)
    print("最终汇总结果：")
    print(final_state["stream_output"])

asyncio.run(main())