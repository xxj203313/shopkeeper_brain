import os

from dotenv import load_dotenv
from openai import OpenAI
import base64

# 初始化客户端
load_dotenv()
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://ws-ggputu6mjvg3gzer.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)

# 图片转 Base64
with open("D:/doc/output/万用表RS-12的使用/auto/images/84c37b209829d15820d5bbe76bbc98e1bf9eddc58bd9c983fc710cb2747d341b.jpg", "rb") as f:
    base64_image = base64.b64encode(f.read()).decode("utf-8")
print(base64_image)
# 调用 VLM
response = client.chat.completions.create(
    # model="qwen3.6-plus",  # 视觉模型
    model="qwen3-vl-plus", #视觉模型
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",        # 告诉 API：这是一段文字
                    "text": "请描述这张图片的内容"
                },
                {
                    "type": "image_url",   # 告诉 API：这是一张图片
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    max_tokens=100
)

summary = response.choices[0].message.content
print(summary)