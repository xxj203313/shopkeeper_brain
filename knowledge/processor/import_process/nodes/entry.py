"""
导入流程的入口节点：
判断文件是pdf还是md
"""
import json
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):

    """入口节点"""

    name: str = "entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState|dict :
        """入口节点处理流程"""
        import_file_path = state.get('import_file_path')
        file_dir = state.get('file_dir')

        if not import_file_path or not file_dir:
            raise ValidationError("数据校验失败",self.name)

        import_file_path_obj = Path(import_file_path)

        suffix = import_file_path_obj.suffix.lower()
        if suffix == ".pdf":
            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = import_file_path
        elif suffix == ".md":
            state['is_md_read_enabled'] = True
            state['md_path'] = import_file_path
        else:
            raise ValidationError("文件类型暂不支持",self.name)

        filte_title = import_file_path_obj.stem
        state['file_title'] = filte_title

        return state

if __name__ == "__main__":
    setup_logging()

    input_state = {
        "import_file_path": r"D:\doc\hak180产品安全手册.pdf",
        "file_dir": r"D:\temp_dir",
    }

    entry = EntryNode()

    node_state =  entry(input_state)

    print(json.dumps(node_state,ensure_ascii=False,indent=4))
