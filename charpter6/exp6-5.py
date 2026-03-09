from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import TypedDict
from fastapi import FastAPI
from prometheus_client import start_http_server, Summary, Counter, make_asgi_app
import time
import random
import asyncio

# 定义状态结构体
class FlowState(TypedDict):
    input: str
    output: str

# 创建Prometheus监控指标
NODE_EXECUTION_TIME = Summary('node_execution_time_seconds','执行节点耗时',['node_name'])
NODE_EXCEPTION_COUNT = Counter('node_exception_count','节点异常计数',['node_name'])

# 定义可监控的异步节点包装器
def monitored_node(name: str):
    def decorator(func):
        async def wrapper(state: FlowState) -> FlowState:
            start_time = time.time()
            try:
                result = await func(state)
                NODE_EXECUTION_TIME.labels(node_name=name).observe(time.time() - start_time)
                return result
            except Exception as e:
                NODE_EXCEPTION_COUNT.labels(node_name=name).inc()
                raise e
        return wrapper
    return decorator

# 定义两个示例节点
@monitored_node("fetch_data")
async def fetch_data_node(state: FlowState) -> FlowState:
    await asyncio.sleep(random.uniform(0.2, 0.5))
    if random.random() < 0.1:
        raise RuntimeError("数据获取失败")
    return {"input":state["input"], "output":f"Fetched({state['input']})"}

@monitored_node("process_data")
async def process_data_node(state: FlowState) -> FlowState:
    await asyncio.sleep(random.uniform(0.3, 0.6))
    return {"input":state["input"], "output":state["output"]+" -> Processed"}

# 构建LangGraph图
workflow = StateGraph(FlowState)
workflow.add_node("fetch_data", RunnableLambda(fetch_data_node))
workflow.add_node("process_data", RunnableLambda(process_data_node))
workflow.set_entry_point("fetch_data")
workflow.add_edge("fetch_data","process_data")
workflow.set_finish_point("process_data")
app_graph = workflow.compile()

# 启动Prometheus指标服务
start_http_server(9000)

# 构建FastAPI应用与LangGraph集成
app = FastAPI()
app.mount("/metrics",make_asgi_app())

@app.get("/run/{text}")
async def run_flow(text: str):
    try:
        final_state = await app_graph.ainvoke({"input":text})
        return {"status":"success", "result":final_state}
    except Exception as e:
        return {"statue":"error", "message":str(e)}
