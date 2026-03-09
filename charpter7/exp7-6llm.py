from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict
import asyncio
import json

# -----------------------------
# 1. 定义状态
# -----------------------------
class AskState(TypedDict):
    question: str
    answer: str
    sub_question: str
    sub_answer: str
    need_tool: bool


# -----------------------------
# 2. 初始化本地 Ollama 大模型
# -----------------------------
llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
    base_url="http://127.0.0.1:11434",
)

parser = StrOutputParser()


# -----------------------------
# 3. 定义工具
# -----------------------------
@tool
def wiki_tool(query: str) -> str:
    """Search simple encyclopedia facts."""
    if "Einstein" in query and ("contribution" in query or "major" in query):
        return (
            "Einstein's major contributions include the special theory of relativity, "
            "the general theory of relativity, and his explanation of the photoelectric effect, "
            "which helped establish quantum theory."
        )
    if "Einstein" in query:
        return "Einstein was a theoretical physicist."
    return "No information found."


# -----------------------------
# 4. 节点1：让 LLM 判断是否需要工具，并生成子问题
# -----------------------------
async def check_need_info(state: AskState) -> AskState:
    prompt = f"""
You are a planner for a QA workflow.

Decide whether the user question needs an external tool lookup.
Return ONLY valid JSON with two keys:
- need_tool: true or false
- sub_question: a concise English sub-question string, or empty string if not needed

User question:
{state["question"]}
""".strip()

    raw = await (llm | parser).ainvoke(prompt)
    print("LLM raw planner output:", raw)

    # 尝试解析 JSON；失败则给一个保底逻辑
    try:
        data = json.loads(raw)
        need_tool = bool(data.get("need_tool", False))
        sub_question = data.get("sub_question", "")
    except Exception:
        # 保底逻辑，避免模型偶尔输出不规范
        if "Einstein" in state["question"]:
            need_tool = True
            sub_question = "What were Einstein's major contributions to physics?"
        else:
            need_tool = False
            sub_question = ""

    return {
        **state,
        "need_tool": need_tool,
        "sub_question": sub_question,
    }


# -----------------------------
# 5. 节点2：显式调用工具
# -----------------------------
async def query_tool(state: AskState) -> AskState:
    result = wiki_tool.invoke({"query": state["sub_question"]})
    return {
        **state,
        "sub_answer": result
    }


# -----------------------------
# 6. 节点3：让 LLM 基于工具结果组织最终答案
# -----------------------------
async def answer_with_tool_result(state: AskState) -> AskState:
    prompt = f"""
You are a helpful assistant.

Answer the user's question naturally and clearly.

User question:
{state["question"]}

Tool result:
{state["sub_answer"]}

Requirements:
- Write in Chinese
- Use the tool result as the factual basis
- Make the answer complete and easy to understand
""".strip()

    final_answer = await (llm | parser).ainvoke(prompt)

    return {
        **state,
        "answer": final_answer
    }


# -----------------------------
# 7. 条件路由
# -----------------------------
def route_edge(state: AskState) -> str:
    if state.get("need_tool"):
        return "query_tool"
    return "answer"


# -----------------------------
# 8. 构建图
# -----------------------------
graph = StateGraph(AskState)

graph.add_node("check_need_info", RunnableLambda(check_need_info))
graph.add_node("query_tool", RunnableLambda(query_tool))
graph.add_node("answer", RunnableLambda(answer_with_tool_result))

graph.set_entry_point("check_need_info")

graph.add_conditional_edges(
    "check_need_info",
    route_edge,
    {
        "query_tool": "query_tool",
        "answer": "answer"
    }
)

graph.add_edge("query_tool", "answer")
graph.set_finish_point("answer")

compiled_graph = graph.compile()


# -----------------------------
# 9. 运行
# -----------------------------
async def main():
    input_state = {
        "question": "Explain the contribution of Einstein",
        "answer": "",
        "sub_question": "",
        "sub_answer": "",
        "need_tool": False
    }

    final_state = await compiled_graph.ainvoke(input_state)

    print("sub_question:", final_state["sub_question"])
    print("sub_answer:", final_state["sub_answer"])
    print("最终回答：", final_state["answer"])


asyncio.run(main())
