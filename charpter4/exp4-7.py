from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_ollama import ChatOllama
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import messages_from_dict, messages_to_dict,AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 定义包含历史的状态结构
class MemoryState(TypedDict):
    user_input: str
    chat_history: List[dict]  # 存储历史对话消息的列表
    intermediate_response: Optional[str]
    final_response: Optional[str]

# 初始化Agent工具
def summarizer_tool(text: str) -> str:
    return f"总结：{text[:20]}..."  # 简单示例，实际可调用更复杂的总结工具

def improver_tool(summary: str) -> str:
    return f"改进建议：{summary}。以增强表达效果。"  # 简单示例，实际可调用更复杂的改进工具

summarizer = Tool.from_function(summarizer_tool, name="summarizer", description="总结生成工具")
improver = Tool.from_function(improver_tool, name="improver", description="优化内容工具")

# 初始化模型
llm = ChatOllama(model="qwen3:8b", temperature=0, base_url="http://localhost:11434")

# 创建第一个Agent，接收历史内容并生成总结
memory_1 = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent_1 = create_tool_calling_agent(llm, tools=[summarizer], prompt=ChatPromptTemplate.from_messages([
    ("system", "你是一个总结助手。你【必须且只能】调用一次工具 summarizer 来生成总结。拿到工具返回后，直接输出【最终总结】给用户，不要再次调用任何工具。最终输出用中文，且不要包含工具调用痕迹。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
]))
executor_1 = AgentExecutor(agent=agent_1, tools=[summarizer], memory=memory_1, verbose=True)

# 创建第二个Agent，继续基于历史上下文进行优化
memory_2 = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent_2 = create_tool_calling_agent(llm, tools=[improver], prompt=ChatPromptTemplate.from_messages([
    ("system", "你是一个内容优化助手。你【必须且只能】调用一次工具 improver 来生成改进建议。拿到工具返回后，直接输出【最终改进建议】给用户，不要再次调用任何工具。最终输出用中文，且不要包含工具调用痕迹。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
]))
executor_2 = AgentExecutor(agent=agent_2, tools=[improver], memory=memory_2, verbose=True)

# 节点函数：Agent1生成总结并写入状态
def node_summarize(state: MemoryState) -> MemoryState:
    memory_1.chat_memory.messages = messages_from_dict(state["chat_history"])  # 恢复历史对话
    result = executor_1.invoke({"input": state["user_input"]})
    updated_history = messages_to_dict(memory_1.chat_memory.messages)  # 获取更新后的历史
    return {
        **state,
        "intermediate_response": result["output"],
        "chat_history": updated_history
    }

# 节点函数：Agent2优化总结并写入状态
def node_improve(state: MemoryState) -> MemoryState:
    memory_2.chat_memory.messages = messages_from_dict(state["chat_history"])  # 恢复历史对话
    result = executor_2.invoke({"input": state["intermediate_response"]})
    updated_history = messages_to_dict(memory_2.chat_memory.messages)  # 获取更新后的历史
    return {
        **state,
        "final_response": result["output"],
        "chat_history": updated_history
    }

# 构建LangGraph流程
builder = StateGraph(MemoryState)
builder.add_node("summarize", node_summarize)
builder.add_node("improve", node_improve)
builder.add_edge("summarize", "improve")  # summarize完成后进入improve
builder.add_edge("improve", END)  # improve完成后流程结束
builder.set_entry_point("summarize")
graph = builder.compile()

# 执行流程
initial_state = {
    "user_input": "LangGraph与LangChain结合后能够构建更负责的多Agent任务系统，具备更强的流程调度与节点复用能力。",
    "chat_history": messages_to_dict([HumanMessage(content="开始任务")]),  
    "intermediate_response": None,
    "final_response": None
}

final_state = graph.invoke(initial_state)
print("用户输入：", final_state["user_input"])
print("最终总结：", final_state["intermediate_response"])
print("最终改进建议：", final_state["final_response"])