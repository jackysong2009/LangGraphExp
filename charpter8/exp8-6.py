import os
import importlib.util
from typing import TypedDict, Callable, Dict, Any
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

# 定义状态结构
class WorkflowState(TypedDict):
    input: str
    output: str

# 示例插件子图（可动态加载）
def create_plugin_graph(name: str) -> Callable[[], StateGraph]:
    def build():
        graph = StateGraph(WorkflowState)

        # 定义插件逻辑节点
        def plugin_func(state: WorkflowState) -> WorkflowState:
            result = f"[插件{name}]处理输入：{state['input']}"
            return {"input": state["input"], "output": result}
        
        graph.add_node("plugin_node", RunnableLambda(plugin_func))
        graph.set_entry_point("plugin_node")
        graph.set_finish_point("plugin_node")
        return graph
    return build

# 动态加载插件目录
def load_plugin_modules(plugin_dir: str) -> Dict[str, Callable[[], StateGraph]]:
    plugin_graphs = {}
    for file in os.listdir(plugin_dir):
        path = os.path.join(plugin_dir, file)
        module_name = os.path.splitext(file)[0]
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module,"build_graph"):
            plugin_graphs[module_name] = module.build_graph
    return plugin_graphs

# 主图定义
def build_main_graph(plugin_graphs: Dict[str, Callable[[],StateGraph]]):
    graph = StateGraph(WorkflowState)

    # 初始处理节点：
    def preprocess(state: WorkflowState) -> WorkflowState:
        return {"input": state["input"], "output": ""}
    
    graph.add_node("preprocess", RunnableLambda(preprocess))

    # 动态加载所有插件为子图节点
    for plugin_name, graph_fn in plugin_graphs.items():
        subgraph = graph_fn()
        graph.add_node(plugin_name, subgraph.compile())

    def postprocess(state: WorkflowState) -> WorkflowState:
        return {"input": state["input"], "output":f"最终输出:{state['output']}"}
    
    graph.add_node("postprocess", RunnableLambda(postprocess))

    # 构建流程图：预处理 ->插件1 ->插件2 ->后处理
    graph.set_entry_point("preprocess")
    previous = "preprocess"
    for plugin_name in plugin_graphs.keys():
        graph.add_edge(previous, plugin_name)
        previous = plugin_name
    graph.add_edge(previous, "postprocess")
    graph.set_finish_point("postprocess")

    return graph

# 有两个插件模块位于plugins目录下
# 分别是 plugin_math.py 和 plugin_translate.py
# 每个插件文件需定义 build_graph() 函数，返回 StateGraph 对象
# 示例插件动态加载并构建完整图

# 获取当前脚本所在文件夹的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接出 plugins 的绝对路径（/Users/.../charpter8/plugins）
plugin_dir_path = os.path.join(current_dir, "plugins")
# 使用绝对路径加载插件
plugin_graphs = load_plugin_modules(plugin_dir_path)

main_graph = build_main_graph(plugin_graphs)
app = main_graph.compile()

# 测试输入执行
inputs = {"input": "42"}
result = app.invoke(inputs)
print(result)

