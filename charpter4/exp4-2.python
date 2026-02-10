from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from langchain_ollama import ChatOllama
import asyncio

# 定义工作流状态结构
class StreamState(TypedDict):
    user_input: str
    stream_output: Optional[str]

# 初始化语言模型
llm = ChatOllama(model="qwen3:8b", temperature=0, base_url="http://127.0.0.1:11434", streaming=True)

# 异步流式节点函数：实时接收生成结果并缓冲回填
async def llm_stream_node(state: StreamState) -> StreamState:
    content = state["user_input"]
    prompt = f"请根据以下内容生成摘要：\n{content}"
    
    buffer = []

    # 利用异步迭代获取逐步输出内容
    async for chunk in llm.astream(prompt):
        if hasattr(chunk, "content") and chunk.content:
            buffer.append(chunk.content)
    
    return {
        "user_input": content,
        "stream_output": "".join(buffer)  # 将缓冲内容合并为完整输出
    }

# 构建LangGraph流程
builder = StateGraph(StreamState)
builder.add_node("stream", llm_stream_node)
builder.add_edge("stream", END)  # 流式生成完成后流程结束
# 设置入口节点
builder.set_entry_point("stream")
graph = builder.compile()

# 异步运行主逻辑
async def main():
    initial_state = {
        "user_input": "LangChain与LangGraph联合使用可以实现多节点、状态驱动的语言模型智能体工作流，适用于复杂任务的分布式调度与工具链管理。",
        "stream_output": None
    }
    final_state = await graph.ainvoke(initial_state)
    print("流式生成结果：")
    print(final_state["stream_output"])

# 运行异步主函数
asyncio.run(main())