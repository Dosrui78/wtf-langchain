from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="http://dashscope.aliyuncs.com/compatible-mode/v1", # 指向阿里接口
    model="qwen3.5-122b-a10b",
    temperature=0
)
response = llm.invoke("你好，请介绍一下你自己")
print(response.content)