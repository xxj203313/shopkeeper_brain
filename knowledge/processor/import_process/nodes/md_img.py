"""
导入流程的传入文件为md的节点：
"""
import base64
import json
import re
import time
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import List, Dict, Deque

from openai import OpenAI

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, FileProcessingError, ImageProcessingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients

@dataclass
class ImageContext:

    heading: str
    pre_text: str
    post_text: str


@dataclass
class ImageInfo:
    name: str
    path: str
    context: ImageContext


# 1.MD文件读取 & 备份
class MdFileHandler:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def read_md(self, state) -> tuple[str, Path, Path]:
        """读取MD文件内容"""
        md_path = state.get("md_path", "")
        if not md_path:
            raise StateFieldError(self.node_name, "md_path", str, "属性不存在")

        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError(f"路径不存在：{md_path}", self.node_name)

        with open(md_path_obj, 'r', encoding="utf-8") as f:
            md_content = f.read()

        image_path_obj = md_path_obj.parent / "images"

        return md_content, md_path_obj, image_path_obj

    def backup(self, md_path_obj: Path, new_md_content: str) -> str:
        self.logger.info("【step_5】备份新文件")

        new_file_path = md_path_obj.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )
        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise ImageProcessingError(
                f"文件写入失败: {e}", node_name="md_img_node"
            )
        return str(new_file_path)


class ImageScanner:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def scan_img_dir(self, image_path_obj: Path, md_content, image_extensions, img_content_length) -> List[ImageInfo]:

        imageinfo_list = []


        for image_path in image_path_obj.iterdir():

            if not image_path.is_file():
                continue


            if image_path.suffix.lower() not in image_extensions:
                continue

            ctx: ImageContext = self._find_context(image_path.name, md_content, img_content_length)

            if ctx:
                image_info = ImageInfo(
                    name=image_path.name,
                    path=str(image_path),
                    context=ctx
                )
                imageinfo_list.append(image_info)

        return imageinfo_list

    def _find_context(self, image_name: str, md_content: str, img_content_length: int) -> ImageContext | None:

        line_list: List[str] = md_content.split("\n")

        for idx, line in enumerate(line_list):

            if re.search(r"^!\[.*?\]\(.*?" + re.escape(image_name) + ".*?\)", line):

                heading, pre_index = self._find_heading_above(idx, line_list)
                pre_text_list = line_list[pre_index + 1:idx]

                post_index = self._find_heading_below(idx, line_list)
                post_text_list = line_list[idx + 1:post_index]

                pre_text: str = self._extract_limited_context(pre_text_list, img_content_length,
                                                              direction="front")
                post_text: str = self._extract_limited_context(post_text_list, img_content_length,
                                                               direction="end")

                return ImageContext(
                    heading=heading,
                    pre_text=pre_text,
                    post_text=post_text
                )

        return None

    def _find_heading_above(self, idx, line_list) -> tuple[str, int]:

        for i in range(idx - 1, -1, -1):
            line = line_list[i]
            if re.match(r"^\s*#{1,6}\s+", line):
                return line.strip(), i
        return "", -1

    def _find_heading_below(self, idx, line_list):

        for i in range(idx + 1, len(line_list), 1):
            line = line_list[i]
            if re.match(r"^\s*#{1,6}\s+", line):
                return i
        return len(line_list)

    def _extract_limited_context(self, context_list: List[str], img_content_length: int, direction: str):

        current_paragraph: List[str] = []
        paragraphs: List[str] = []
        for line in context_list:
            is_blank_line = not line.strip()
            is_other_image = re.match(r"!\[.*?\]\(.*?\)", line)
            if is_blank_line or is_other_image:
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph = []
                continue
            current_paragraph.append(line)

        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        if direction == "front":
            paragraphs.reverse()

        total: int = 0
        selected: List[str] = []

        for paragraph in paragraphs:

            if total + len(paragraph) >= img_content_length and selected:
                break

            total += len(paragraph)
            selected.append(paragraph)

        if direction == "front":
            selected.reverse()

        return "\n\n".join(selected)

class VLMSummarizer:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def summarize_all(self, file_title, imageinfo_list, vl_model, requests_per_minute) -> Dict[str, str]:

        summarizes: Dict[str, str] = {}

        requests_timestamp: Deque[float] = Deque()

        try:
            client = AIClients.get_openai()
        except Exception as e:

            for image_info in imageinfo_list:
                summarizes[image_info.name] = "默认摘要"

            return summarizes

        for image_info in imageinfo_list:

            self._enforce_rate_limit(requests_timestamp, requests_per_minute)  # 默认时间窗口60秒

            summarizes[image_info.name] = self.summarize_one(file_title, image_info, vl_model, client)

        return summarizes

    def summarize_one(self, file_title: str, image_info: ImageInfo, vl_model: str, client: OpenAI) -> str:

        parts = [p for p in (image_info.context.heading, image_info.context.pre_text, image_info.context.post_text) if
                 p]

        final_context = "\n".join(parts)

        with open(image_info.path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')

        try:
            resp = client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{file_title}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            },
                        },
                    ],
                }],
            )

            return resp.choices[0].message.content.strip()
        except Exception as e:

            return "默认摘要"

    def _enforce_rate_limit(self, requests_timestamp: Deque[float], requests_per_minute: int = 3, window: int = 60):
        """限流函数   时间滑动窗口"""
        now = time.time()

        # 1.每次请求VLM大模型前，移除窗口外(>60秒)的时间戳
        while requests_timestamp and now - requests_timestamp[0] > 60:
            requests_timestamp.popleft()  # 进入队列最早元素，已经超过时间窗口，移除

        # 2.检查窗口内请求数是否大于上限阈值
        if len(requests_timestamp) >= requests_per_minute:
            # 3.达到上限,等待最早请求过期。
            sub = 60 - (now - requests_timestamp[0])  # 剩余时间
            time.sleep(sub)

            now = time.time()
            # 再次清理超过时间窗口的请求时间戳
            while requests_timestamp and now - requests_timestamp[0] > 60:
                requests_timestamp.popleft()

                # 4.记录本次请求时间戳
        requests_timestamp.append(now)  # 往队列里增加当前请求的时间戳

class ImageUploader:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def upload_and_replace(self, file_title, md_content, imageinfo_list, summarizes: Dict[str, str], minio_bucket,
                           endpoint_minio_url) -> str:

        image_name_urls: Dict[str, str] = {}

        image_name_urls: Dict[str, str] = self._upload_all(file_title, imageinfo_list, minio_bucket, endpoint_minio_url)

        new_md_content = self._replace_in_md(md_content, summarizes, image_name_urls)

        return new_md_content

    def _upload_all(self, file_title: str, imageinfo_list: List[ImageInfo], minio_bucket: str,
                    endpoint_minio_url: str) -> Dict[str, str]:

        image_name_urls: Dict[str, str] = {}
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:

            for image_info in imageinfo_list:
                image_name_urls[image_info.name] = image_info.path
            return image_name_urls

        for image_info in imageinfo_list:
            try:
                minio_client.fput_object(
                    bucket_name=minio_bucket,
                    object_name=f"{file_title}/{image_info.name}",
                    file_path=image_info.path,
                    content_type="image/jpeg"
                )

                image_name_urls[image_info.name] = f"{endpoint_minio_url}/{minio_bucket}/{file_title}/{image_info.name}"
            except Exception as e:

                image_name_urls[image_info.name] = image_info.path

        return image_name_urls

    @staticmethod
    def _replace_in_md(
            md_content: str,
            summaries: Dict[str, str],
            remote_urls: Dict[str, str],
    ) -> str:

        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

        # ![默认](images/e67add46b6982ad7f2f380cfc07da1916e398db04fc01d23b8fad0ad72fe18d0.jpg)
        # ![万用表RS-12直流电流测量接线示意图（10A档位）](http://192.168.6.170:9000/knowledge-base-files/hak180产品安全手册/01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg)
        def replacer(match: re.Match) -> str:
            original_path = match.group(2).strip()
            file_name_in_md = Path(
                original_path).name  # e67add46b6982ad7f2f380cfc07da1916e398db04fc01d23b8fad0ad72fe18d0.jpg
            for img_name, summary in summaries.items():
                if img_name == file_name_in_md:
                    return f"![{summary}]({remote_urls[img_name]})"
            return match.group(0)

        return pattern.sub(replacer, md_content)  # 匹配正则表达式，则调用replacer函数，替换图片的摘要和地址


class MarkDownImageNode(BaseNode):
    """MD中图片的处理节点"""

    name: str = "md_img_node"

    def __init__(self):
        super().__init__()
        self.md_file_handler = MdFileHandler(self.logger, self.name)
        self.image_scanner = ImageScanner(self.logger, self.name)
        self.vlm_summarizer = VLMSummarizer(self.logger, self.name)
        self.image_uploader = ImageUploader(self.logger, self.name)

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:

        # 1.读取md文件内容
        md_content, md_path_obj, image_path_obj = self.md_file_handler.read_md(state)
        if not image_path_obj.exists():
            state['md_content'] = md_content
            return state

        # 2.图片扫描
        imageinfo_list: List[ImageInfo] = self.image_scanner.scan_img_dir(image_path_obj,
                                                                          md_content,
                                                                          self.config.image_extensions,
                                                                          self.config.img_content_length)

        # 3.VLM 生成摘要  qwen3-vl-flash
        summarizes: dict[str, str] = self.vlm_summarizer.summarize_all(state['file_title'], imageinfo_list,
                                                                       self.config.vl_model,
                                                                       self.config.requests_per_minute)

        # 4.上传图片到Minio & 替换( 摘要+URL)
        new_md_content: str = self.image_uploader.upload_and_replace(
            state['file_title'],  # 作为上传图片的父路径名称
            md_content,
            imageinfo_list,
            summarizes,
            self.config.minio_bucket,
            self.config.get_minio_base_url()
        )

        # 5.新md文件保存
        self.md_file_handler.backup(md_path_obj, new_md_content)

        # 6.更新state并返回结果数据
        state['md_content'] = new_md_content

        return state


if __name__ == "__main__":
    setup_logging()

    node = MarkDownImageNode()
    state = {
        "file_title": "hak180产品安全手册",
        "md_path": r"D:\woekspace\woekspaceAI260706\shopkeeper_brain\temp_dir\hak180产品安全手册\auto\hak180产品安全手册.md"
    }
    node.process(state)

