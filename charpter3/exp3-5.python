from typing import TypedDict, Literal
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# 定义工作流状态结构
class RouteState(TypedDict):
    user_input: str
    route: Literal["translate", "summarize", "unknown"]
    output: str

# 初始化语言模型
llm = ChatOllama(model="qwen3:8b", temperature=0, base_url="http://192.168.1.60:11434")

# 路由判断函数（意图分类）
def route_decision(state: RouteState) -> str:
    prompt = ChatPromptTemplate.from_messages([
       ("system", "你是一个意图分类器。根据用户输入判断其意图是翻译还是总结。如果无法判断则返回'unknown'。"),
       ("human", "用户输入：{text}\n只输出分类标签（translate, summarize, unknown）。")
    ])
    chat = prompt | llm
    result = chat.invoke({"text": state["user_input"]}).content.strip().lower()
    if "translate" in result:
        return "translate"
    elif "summarize" in result:
        return "summarize"
    else:
        return "unknown"
    
# 定义翻译工具与Agent
def translate_tool(text: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个翻译助手，将输入文本翻译成英文。"),
        ("human", "请翻译以下内容：{text}")
    ])
    chat = prompt | llm
    return chat.invoke({"text": text}).content

translate_agent = create_tool_calling_agent(
    llm,
    tools=[Tool.from_function(translate_tool, name="translate_tool", description="用于将文本翻译成英文")],
    prompt=ChatPromptTemplate.from_messages([
        ("system", "你是一个智能助手，会根据用户需求调用合适的工具完成任务。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])  
)

translate_executor = AgentExecutor(agent=translate_agent, tools=[Tool.from_function(translate_tool, name="translate_tool", description="用于将文本翻译成英文")], verbose=True)

# 定义总结工具与Agent
def summarize_tool(text: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个总结助手，将输入文本进行简要总结。"),
        ("human", "请总结以下内容：{text}")
    ])
    chat = prompt | llm
    return chat.invoke({"text": text}).content  

summarize_agent = create_tool_calling_agent(
    llm,
    tools=[Tool.from_function(summarize_tool, name="summarize_tool", description="用于对文本进行总结")],
    prompt=ChatPromptTemplate.from_messages([
        ("system", "你是一个智能助手，会根据用户需求调用合适的工具完成任务。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
)   

summarize_executor = AgentExecutor(agent=summarize_agent, tools=[Tool.from_function(summarize_tool, name="summarize_tool", description="用于对文本进行总结")], verbose=True)

# 翻译节点
def translate_node(state: RouteState) -> RouteState:
    result = translate_executor.invoke({"input": state["user_input"]})
    return {
        "user_input": state["user_input"],
        "route": "translate",
        "output": result["output"]
    }

# 总结节点
def summarize_node(state: RouteState) -> RouteState:
    result = summarize_executor.invoke({"input": state["user_input"]})
    return {
        "user_input": state["user_input"],
        "route": "summarize",
        "output": result["output"]
    }

# 未知意图节点
def unknown_node(state: RouteState) -> RouteState:
    return {
        "user_input": state["user_input"],
        "route": "unknown",
        "output": "无法识别用户意图。"
    }

# 构建LangGraph状态图
builder = StateGraph(RouteState)
builder.add_node("router", lambda x: x)
builder.add_node("translate", translate_node)
builder.add_node("summarize", summarize_node)
builder.add_node("unknown", unknown_node)

# 添加条件跳转控制逻辑
builder.add_conditional_edges("router", route_decision, {
    "translate": "translate",
    "summarize": "summarize",
    "unknown": "unknown"
})

# 设置入口节点与终点
builder.set_entry_point("router")
builder.set_finish_point("translate")
builder.set_finish_point("summarize")
builder.set_finish_point("unknown")

graph = builder.compile()

# 执行测试流程
input_state: RouteState = {
    "user_input": "请帮我总结这句话：人工智能正在迅速发展，改变着各行各业的面貌。",
    "route": "unknown",
    "output": ""
}

final_state = graph.invoke(input_state)
print("分发路径:", final_state["route"])
print("智能体返回结果:", final_state["output"])