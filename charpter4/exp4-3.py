from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_ollama import ChatOllama
from langchain_core.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from ollama_embeddings_client import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 构建简单向量检索器
docs = [
    Document(page_content="王湘华在石家庄公交公司上班"),
    Document(page_content="王湘华在石家庄桥西区休门街居住"),
    Document(page_content="王湘华最近参加了公司的联欢晚会")
]

embedding = OllamaEmbeddings(model="qwen3-embed:8b0q4km")
vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever()
retriever_tool = create_retriever_tool(retriever, name="doc_lookup", description="根据用户查询从文档库中检索相关内容")

# 构建函数工具
def get_length(text: str) -> str:
    return f"输入文本的长度是 {len(text)} 个字符。"

length_tool = Tool.from_function(
    func=get_length, 
    name="text_length", 
    description="计算输入文本的长度"
)

# 构建语言模型与代理
llm = ChatOllama(model="qwen3:8b", temperature=0, base_url="http://127.0.0.1:11434")
tools = [retriever_tool, length_tool]
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个可以使用工具来回答问题的智能助手。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 定义工作流状态结构
class ToolState(TypedDict):
    query: str
    result: Optional[str]

# 定义节点函数：执行代理调用并将结果写入状态
def tool_node(state: ToolState) -> ToolState:
    output = executor.invoke({"input": state["query"]})
    return {"query": state["query"], "result": output["output"]}

# 构建LangGraph流程
builder = StateGraph(ToolState)
builder.add_node("tool_execution", tool_node)
builder.add_edge("tool_execution", END)
builder.set_entry_point("tool_execution")
graph = builder.compile()

# 执行流程
initial_state = {"query": "请告诉我关于王湘华的信息", "result": None}
final_state = graph.invoke(initial_state)
print("查询内容：", final_state["query"])
print("工具调用结果：", final_state["result"])