"""
导入流程的文档切分节点：
"""
import json
import os
import re
from typing import Tuple, Dict, List, Any

import sections
from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer



class DocumentSplitNode(BaseNode):
    """文档切分节点"""

    name: str = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        """文档切分处理流程"""

        # 1.获取输入进行数据校验
        md_content, file_title, max_content_length, min_content_length = self._get_input_validation(state)

        # 2.标题切分
        sections: List[Dict[str, Any]] = self._split_by_title(md_content, file_title)
        # print(sections)

        # 3.切分合并：  长切短合
        final_sections: List[Dict[str, Any]] = self.split_and_merge(sections, max_content_length, min_content_length)

        # 4.组装切分
        chunks = self._assemble_chunk(final_sections,)

        # 5.日志备份

        # 5.1. 日志统计
        self._log_summary(md_content, chunks, max_content_length)

        # 5.2. 备份
        self._backup_chunks(state, chunks)

        state['chunks'] = chunks

        return state

    def _get_input_validation(self, state: ImportGraphState) -> Tuple[str, str, int, int]:
        """对输入数据进行校验"""
        md_content = state.get("md_content")
        if not md_content:
            raise StateFieldError(self.name, "md_content", str)

        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(self.name, "file_title", str)

        max_content_length = self.config.max_content_length
        min_content_length = self.config.min_content_length

        if max_content_length < 0 or min_content_length < 0 or max_content_length <= min_content_length:
            raise ValidationError("切分最大值和最小值参数错误", self.name)

        return md_content, file_title, max_content_length, min_content_length

    def _split_by_title(self, md_content, file_title) -> List[Dict[str, Any]]:

        sections: List[Dict[str, Any]] = []
        in_fence = False
        heading_re = re.compile(
            r"^\s*(#{1,6})\s+.+")
        body: List[str] = []
        content_lines = md_content.split("\n")
        current_title = ""
        current_level = 0
        hierarchy = [""] * 7

        def _flush():

            if current_title or body:
                parent_title = ""
                for lev in range(current_level - 1, 0, -1):
                    if hierarchy[lev]:
                        parent_title = hierarchy[lev]
                        break

                if not parent_title:
                    parent_title = current_title if current_title else file_title

                sections.append({
                    "parent_title": parent_title,
                    "title": current_title if current_title else file_title,
                    "body": "\n".join(body),
                    "file_title": file_title
                })

        for index, line in enumerate(content_lines):
            if line.startswith("~~~~") or line.startswith("```"):
                in_fence = not in_fence

            match = heading_re.match(line) if not in_fence else None

            if match:
                _flush()

                level = len(match.group(1))
                current_level = level
                current_title = line
                hierarchy[current_level] = current_title

                for i in range(level + 1, 7):
                    hierarchy[i] = ""
                body = []

            else:
                body.append(line.strip())

        _flush()

        return sections

    def split_and_merge(self, sections: List[Dict[str, Any]],
                        max_content_length: int, min_content_length: int):

        self.log_step("step3", "切分及合并...")

        current_sections = []
        for section in sections:
            current_sections.extend(self.split_long_section(section, max_content_length))

        final_sections = self.merge_short_section(current_sections, min_content_length)
        print(final_sections)

        return final_sections

    def split_long_section(self, section: Dict[str, Any], max_content_length: int = 1000) -> List[Dict[str, Any]]:
        title = section.get("title")
        parent_title = section.get("parent_title")
        body = section.get("body")
        file_title = section.get("file_title")

        if "<table>" in body:
            body = MarkdownTableLinearizer.process(body)
            section["body"] = body

        MAX_TITLE_LENGTH = 50
        if len(title) > MAX_TITLE_LENGTH:
            title = title[:MAX_TITLE_LENGTH]

        title_prefix = title + "\n\n"

        if len(title_prefix) + len(body) <= max_content_length:
            return [section]

        body_length = max_content_length - len(title_prefix)
        if body_length <= 0:
            return [section]

        sub_sections = []

        if len(title_prefix) + len(body) > max_content_length:
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "。", "！", "？", ".", ",", "!", " ", ""],  # 用于切分的分隔符
                chunk_size=body_length,
                chunk_overlap=0,
                keep_separator=False,
            )

            texts = splitter.split_text(body)
            if len(texts) <= 1:
                return [section]

            for index, text in enumerate(texts):
                sub_sections.append({
                    "parent_title": parent_title,
                    "title": section.get("title") + f"-{index + 1}",
                    "body": text,
                    "file_title": file_title,
                    "part": f"{index + 1}"
                })
        return sub_sections

    def merge_short_section(self, current_sections: List[Dict[str, Any]], min_content_length) -> List[Dict[str, Any]]:

        final_sections: List[Dict[str, Any]] = []


        current_section = current_sections[0]

        for next_section in current_sections[1:]:
            same_parent = current_section["parent_title"] == next_section["parent_title"]
            if same_parent and len(current_section['body']) < min_content_length:
                current_section['body'] = current_section['body'].rstrip() + "\n\n" + next_section['body'].lstrip()
                current_section['title'] = current_section['parent_title']
                current_section['part'] = 0
            else:
                final_sections.append(current_section)
                current_section = next_section

        final_sections.append(current_section)

        part_counter = {}

        for final_section in final_sections:
            if "part" in final_section:
                parent_title = final_section.get('parent_title')
                part_counter[parent_title] = part_counter.get(parent_title, 0) + 1
                new_part = part_counter[parent_title]
                final_section['part'] = new_part
                final_section['title'] = final_section['title'] + f"- {new_part}"


        return final_sections


    def _assemble_chunk(self, final_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.log_step("step4", "组装最终的切片信息...")
        chunks = []

        for chunk in final_chunks:
            title = chunk.get('title')
            file_title = chunk.get('file_title')
            parent_title = chunk.get('parent_title')
            body = chunk.get('body')
            content = f"{title}\n\n{body}"

            assemble_chunk = {
                "title": title,
                "file_title": file_title,
                "parent_title": parent_title,
                "content": content,
            }

            if "part" in chunk:
                assemble_chunk['part'] = chunk.get('part')

            chunks.append(assemble_chunk)

        return chunks

    def _log_summary(self, raw_content: str, chunks: List[dict], max_length: int):

        self.log_step("step5", "输出统计")

        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(chunks)}")
        self.logger.info(f"最大切片长度: {max_length}")

        if chunks:
            self.logger.info("章节预览:")
            for i, sec in enumerate(chunks[:5]):
                title = sec.get("title", "")[:30]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(chunks) > 5:
                self.logger.info(f"  ... 还有 {len(chunks) - 5} 个章节")

    def _backup_chunks(self, state: ImportGraphState, sections: List[dict]):

        self.log_step("step6", "备份切片")

        local_dir = state.get("file_dir", "")
        if not local_dir:
            self.logger.debug("未设置 file_dir，跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")



if __name__ == '__main__':
    setup_logging()

    document_node = DocumentSplitNode()

    file_path = r"D:\woekspace\woekspaceAI260706\shopkeeper_brain\temp_dir\hak180产品安全手册\auto\hak180产品安全手册_new.md"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    state = {
        "file_title": "hak180产品安全手册",
        "md_content": content,
        "file_dir": r"D:\woekspace\woekspaceAI260706\shopkeeper_brain\temp_dir\hak180产品安全手册\auto"
    }
    document_node.process(state)


