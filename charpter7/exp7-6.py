from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from typing import TypedDict
import asyncio

# 定义图状态
class AskState(TypedDict):
    question: str
    answer: str
    sub_question: str
    sub_answer: str

# 初始化LLM模型
llm = ChatOllama(model="qwen3:8b", temperature=0, base_url="http://127.0.0.1:11434")

# 工具函数：百科查询工具
@tool
def wiki_tool(query: str) -> str:
    """wiki tool"""
    if "Einstein" in query:
        return "Einstein was a theoretical physicist who developed the theory of relativity."
    return "No information found."

tools = [wiki_tool]
agent_executor = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant"
)

# Step1:判断是否缺失信息
async def check_need_info(state: AskState) -> AskState:
    if "Einstein" in state["question"]:
        return {
            "question": state["question"],
            "sub_question": "Who is Einstein?",
            "sub_answer": "",
            "answer": ""
        }
    else:
        return {
            **state,
            "answer": "This question can be answered directly."
        }
    
# Step2:提问并调用工具（Self_Ask)
async def query_tool(state: AskState) -> AskState:
    result = wiki_tool.invoke({"query": state["sub_question"]})
    return {
        **state,
        "sub_answer": result
    }

# Step3:整合子答案生成最终回复
async def answer_with_tool_result(state: AskState) -> AskState:
    full_answer = f"{state['sub_answer']} Therefore, the answer to '{state['question']}' is based on that."
    return {
        **state,
        "answer": full_answer
    }

# 构建图流程
graph = StateGraph(AskState)
graph.add_node("check_need_info", RunnableLambda(check_need_info))
graph.add_node("query_tool", RunnableLambda(query_tool))
graph.add_node("answer", RunnableLambda(answer_with_tool_result))

# 边控制：如果需要补全信息则跳转工具，否则直接进入answer
def route_edge(state: AskState) -> str:
    if state.get("sub_question"):
        return "query_tool"
    return "answer"

graph.set_entry_point("check_need_info")
graph.add_conditional_edges("check_need_info", route_edge)
graph.add_edge("query_tool", "answer")
graph.set_finish_point("answer")

compiled_graph = graph.compile()

# 执行一次
import asyncio
async def main():
    input_state = {
        "question": "Explain the contribution of Einstein",
        "answer": "",
        "sub_question": "",
        "sub_answer": ""
    }
    final_state = await compiled_graph.ainvoke(input_state)
    print("最终回答：", final_state["answer"])

asyncio.run(main())
