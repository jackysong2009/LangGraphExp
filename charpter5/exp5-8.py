from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Literal, Union, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.language_models.fake import FakeListLLM

# 定义状态结构
class RouteState(TypedDict):
    input: str
    route: str
    messages: Annotated[list[BaseMessage], add_messages]

# 构造静态 LLM 返回用于测试（替代实际的语言模型）
llm = FakeListLLM(responses=[
    "weather",  # 第一轮会识别为天气
])

# 构造理由判断链，意图分类
router_prompt = PromptTemplate.from_template(
    "请根据用户输入的内容判断应该路由到哪个处理器，选项有：weather, news, chat。用户输入是：{input}"
    "只输出一个词，必须是weather, news, chat中的一个。"
)
router_chain = router_prompt | llm | StrOutputParser()  # 输出直接解析为字符串

# 节点：记录输入
def receive_input(state: RouteState) -> RouteState:
    print(f"收到用户输入: {state['input']}")
    return {"messages": [HumanMessage(content=state["input"])]}

# 节点：动态路由控制器，调用LangChain的router_chain返回route字段
def route_controller(state: RouteState) -> RouteState:
    route_text = router_chain.invoke({"input": state["input"]})
    route = route_text.strip().lower()  # 去除可能的前后空格    
    print(f"路由控制器决定路由到: {route}")
    return {"route": route, "messages": [SystemMessage(content=f"路由控制器决定路由到: {route}")]}

# 分支处理节点
def dispatch_weather(state: RouteState) -> Literal["weather", "news", "chat"]:
    if state["route"] not in ["weather", "news", "chat"]:
        return "chat"
    return state["route"]

# 路由目标节点
def handle_weather(state: RouteState) -> RouteState:
    print("处理天气相关请求")
    return {"messages": [AIMessage(content="这是天气相关的回复。")]}

def handle_news(state: RouteState) -> RouteState:
    print("处理新闻相关请求")
    return {"messages": [AIMessage(content="这是新闻相关的回复。")]}

def handle_chat(state: RouteState) -> RouteState:
    print("处理聊天相关请求")
    return {"messages": [AIMessage(content="这是聊天相关的回复。")]}

# 构建状态图
builder = StateGraph(RouteState)
builder.add_node("receive_input", receive_input)
builder.add_node("route_controller", route_controller)
builder.add_node("dispatch_weather", dispatch_weather)
builder.add_node("handle_weather", handle_weather)
builder.add_node("handle_news", handle_news)
builder.add_node("handle_chat", handle_chat)

builder.set_entry_point("receive_input")
builder.add_edge("receive_input", "route_controller")
builder.add_conditional_edges("route_controller", dispatch_weather, {
    "weather": "handle_weather",
    "news": "handle_news",
    "chat": "handle_chat"
})

builder.add_edge("handle_weather", END)
builder.add_edge("handle_news", END)
builder.add_edge("handle_chat", END)

# 编译执行图
compiled = builder.compile()

# 执行图
input_text = "今天天气怎么样？"
result = compiled.invoke({"input": input_text})

# 打印交互消息
for msg in result["messages"]:
    print(f"{msg.__class__.__name__}: {msg.content}")

