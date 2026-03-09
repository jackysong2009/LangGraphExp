from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import TypedDict, List
import asyncio
import json
import uuid
import os

# 定义状态结构
class PipelineState(TypedDict):
    input: str
    intermediate: str
    output: str

# 记录测试样例的路径
TEST_CASE_DIR = "./test_cases"
os.makedirs(TEST_CASE_DIR, exist_ok=True)

# 流程运行状态记录器
execution_trace: List[dict] = []

# 包装函数用于记录每个节点的输入输出状态
def traced_node(name: str):
    def decorator(func):
        async def wrapper(state: PipelineState) -> PipelineState:
            input_state = state.copy()
            result = await func(state)
            record = {
                "node": name,
                "input": input_state,
                "output": result
            }
            execution_trace.append(record)
            return result
        return wrapper
    return decorator

# 节点定义
@traced_node("parse_input")
async def parse_node(state: PipelineState) -> PipelineState:
    await asyncio.sleep(0.1)
    return {"input": state["input"],"intermediate": state["input"].upper(),"output": ""}

@traced_node("generate_response")
async def respond_node(state: PipelineState) -> PipelineState:
    await asyncio.sleep(0.1)
    output_text = f"Response({state['intermediate']})"
    return {"input": state["input"],"intermediate": state["intermediate"],"output": output_text}

# 构建图结构
workflow = StateGraph(PipelineState)
workflow.add_node("parse_input", RunnableLambda(parse_node))
workflow.add_node("generate_response", RunnableLambda(respond_node))
workflow.set_entry_point("parse_input")
workflow.add_edge("parse_input", "generate_response")
workflow.set_finish_point("generate_response")
compiled_graph = workflow.compile()

# 执行图并保存测试用例
async def run_and_save(input_text: str):
    global execution_trace
    execution_trace.clear()
    final_result = await compiled_graph.ainvoke({"input": input_text})
    trace_id = str(uuid.uuid4())
    with open(f"{TEST_CASE_DIR}/{trace_id}.json", "w") as f:
        json.dump({
            "input": input_text,
            "trace": execution_trace,
            "final": final_result
        }, f, indent=2)
    return trace_id, final_result

# 测试用例回放与断言校验
async def replay_test_case(file_path: str):
    with open(file_path,"r") as f:
        case = json.load(f)
    state = {"input": case["input"], "intermediate": "", "output": ""}
    for step in case["trace"]:
        node_name = step["node"]
        expected_output = step["output"]
        if node_name == "parse_input":
            result = await parse_node(state)
        elif node_name == "generate_response":
            result = await respond_node(state)
        else:
            raise ValueError(f"未知节点：{node_name}")
        assert result == expected_output, f"{node_name}节点输出不一致"
        state = result
    assert state == case["final"], "最终输出不一致"
    print("测试用例通过", file_path)

# 主运行逻辑（只示范一次生成与一次测试）
import asyncio
async def main():
    trace_id, result = await run_and_save("test input")
    print("执行完成，最终输出：", result)
    await replay_test_case(f"{TEST_CASE_DIR}/{trace_id}.json")

asyncio.run(main())
