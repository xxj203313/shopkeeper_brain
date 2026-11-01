"""
    导入流程主图
    使用 LangGraph 构建文档导入工作流
"""
import json

from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph

from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.import_process.nodes.bge_embedding import BgeEmbeddingChunksNode
from knowledge.processor.import_process.nodes.ducment_split import DocumentSplitNode
from knowledge.processor.import_process.nodes.entry import EntryNode
from knowledge.processor.import_process.nodes.import_milvus import ImportMilvusNode
from knowledge.processor.import_process.nodes.item_name_recognition import ItemNameRecognitionNode
from knowledge.processor.import_process.nodes.md_img import MarkDownImageNode
from knowledge.processor.import_process.nodes.pdf_to_md import PdfToMdNode
from knowledge.processor.import_process.state import ImportGraphState, create_default_state


def load_file_route(state: ImportGraphState):

    if state.get("is_pdf_read_enabled",False):  #获取属性值不存在，会取默认值。
        return "pdf_to_md_node"
    elif state.get("is_md_read_enabled",False):
        return "md_img_node"
    else: #兜底程序   降级处理
        return "END"


def create_import_graph() -> CompiledStateGraph:

    # 1.定义状态  state.py

    # 2.定义节点   参看 节点文件
    nodes = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": PdfToMdNode(),
        "md_img_node": MarkDownImageNode(),
        "document_split_node": DocumentSplitNode(),
        "item_name_rec_node": ItemNameRecognitionNode(),
        "bge_embedding_node": BgeEmbeddingChunksNode(),
        "import_milvus_node": ImportMilvusNode(),
    }

    # 3.创建图实例
    import_graph = StateGraph(ImportGraphState)
    for key, node in nodes.items():
        import_graph.add_node(key, node)

    # 4.添加边
    import_graph.set_entry_point("entry_node")  # 设置入口节点

    import_graph.add_conditional_edges(
        "entry_node",
        load_file_route,
        {
            "pdf_to_md_node": "pdf_to_md_node",
            "md_img_node": "md_img_node",
            "END": END
        }
    )

    import_graph.add_edge("pdf_to_md_node", "md_img_node")
    import_graph.add_edge("md_img_node", "document_split_node")
    import_graph.add_edge("document_split_node", "item_name_rec_node")
    import_graph.add_edge("item_name_rec_node", "bge_embedding_node")
    import_graph.add_edge("bge_embedding_node", "import_milvus_node")

    # 5.编译节点
    compiled_graph = import_graph.compile()
    return compiled_graph


kb_import_process_graph = create_import_graph()

def run_import_graph(inputState: ImportGraphState) -> ImportGraphState:

    # 创建图输入 state
    init_state = create_default_state(
        import_file_path = inputState.get("import_file_path"),
        file_dir = inputState.get("file_dir")
    )

    #运行图
    final_state = None
    for event in kb_import_process_graph.stream(inputState):  #stream_mode="updates"  增量结果
        for node_name, process_state in  event.items():

            print(f"运行节点: {node_name}")
            final_state = process_state


    return final_state


# 测试
if __name__ == "__main__":

    setup_logging()

    input_state = {

        "import_file_path": r"D:\doc\hak180产品安全手册.pdf",  # 导入文件路径（原始输入）
        "file_dir": r"D:\temp_dir",  # 导入(出)文件目录

    }

    final_state = run_import_graph(input_state)

    print(json.dumps(final_state, indent=4, ensure_ascii=False))
    # 打印图
    kb_import_process_graph.get_graph().print_ascii()
