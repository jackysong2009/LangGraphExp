from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from ollama_embeddings_client import OllamaEmbeddings

# 定义状态结构，包含查询内容与检索结果
class VectorState(TypedDict):
    query: str
    retrieved_content: Optional[str]

# 构造文档集合
documents = [
    Document(page_content="王湘华在石家庄公交公司上班"),
    Document(page_content="王湘华在石家庄桥西区休门街居住"),
    Document(page_content="王湘华最近参加了公司的联欢晚会")
]


# 初始化嵌入器与构建向量数据库（FAISS）
embedding_model = OllamaEmbeddings(model="qwen3-embedding")
vector_db = FAISS.from_documents(documents, embedding_model)

# 节点函数：执行向量检索并将结果写入状态
def retrieval_node(state: VectorState) -> VectorState:
    query = state["query"]
    results = vector_db.similarity_search(query, k=1)
    top_content = results[0].page_content if results else "未找到相关内容。"
    return {"query": query, "retrieved_content": top_content}

# 构建LangGraph流程
builder = StateGraph(VectorState)
builder.add_node("retriever", retrieval_node)
builder.add_edge("retriever", END)
builder.set_entry_point("retriever")
graph = builder.compile()

# 执行检索流程
initial_state = {"query": "王湘华最近干什么了？", "retrieved_content": None}
final_state = graph.invoke(initial_state)
print("查询内容：", final_state["query"])  # 输出查询内容
print("检索结果：", final_state["retrieved_content"])  # 输出检索结果