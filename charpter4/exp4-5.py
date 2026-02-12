from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 定义状态结构，message 字段用于Agent间传递内容
class AgentMessageState(TypedDict):
    task_request: str
    agent_message: Optional[str]
    task_feedback: Optional[str]

# Agent1: 生成任务处理建议
def planning_tool(input_text: str) -> str:
    return f"建议，应将任务'{input_text}'拆分为若干小步骤执行。"

# Agent2: 读取建议并生成反馈说明
def feedback_tool(advice: str) -> str:
    return f"收到建议：'{advice}'。我认为这个建议可行，可以按照这个思路执行。"

# 注册为LangChain工具
planner = Tool.from_function(planning_tool, name="planner", description="生成任务处理建议")
responder = Tool.from_function(feedback_tool, name="responder", description="读取建议并生成反馈说明")

# 构建两个Agent
llm = ChatOllama(model="qwen3:8b", temperature=0, base_url="http://localhost:11434")
prompt_1 = ChatPromptTemplate.from_messages([
    ("system",
     "你是任务规划助手。你【必须且只能】调用一次工具 planner 获取建议。"
     "拿到工具返回后，直接输出【最终建议】给用户，不要再次调用任何工具。"
     "最终输出用中文，且不要包含工具调用痕迹。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
prompt_2 = ChatPromptTemplate.from_messages([
    ("system",
     "你是反馈助手。你【必须且只能】调用一次工具 responder 获取反馈。"
     "拿到工具返回后，直接输出【最终反馈】给用户，不要再次调用任何工具。"
     "最终输出用中文，且不要包含工具调用痕迹。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent_1 = create_tool_calling_agent(llm, tools=[planner], prompt=prompt_1)
agent_2 = create_tool_calling_agent(llm, tools=[responder], prompt=prompt_2)

executor_1 = AgentExecutor(
    agent=agent_1, 
    tools=[planner], 
    verbose=True
    )
executor_2 = AgentExecutor(
    agent=agent_2, 
    tools=[responder], 
    verbose=True
    )

# LangGraph节点1:Agent1处理请求并写入中间信息
def node_generate_message(state: AgentMessageState) -> AgentMessageState:
    result = executor_1.invoke({"input": state["task_request"]})
    return {
        **state,
        "agent_message": result["output"]
    }

# LangGraph节点2:Agent2读取Agent1的建议并生成反馈
def node_feedback_response(state: AgentMessageState) -> AgentMessageState:
    result = executor_2.invoke({"input": state["agent_message"]})
    return {
        **state,
        "task_feedback": result["output"]
    }

# 构建LangGraph流程
builder = StateGraph(AgentMessageState)
builder.add_node("generate", node_generate_message)
builder.add_node("feedback", node_feedback_response)
builder.add_edge("generate", "feedback")
builder.add_edge("feedback", END)
builder.set_entry_point("generate")
graph = builder.compile()

# 执行流程
initial_state = {"task_request": "6年级小学生寒假英语提升。", "agent_message": None, "task_feedback": None}
final_state = graph.invoke(initial_state)
print("任务请求：", final_state["task_request"])  # 输出任务请求
print("Agent1建议：", final_state["agent_message"])  # 输出Agent1生成
print("Agent2反馈：", final_state["task_feedback"])  # 输出Agent2生成的反馈