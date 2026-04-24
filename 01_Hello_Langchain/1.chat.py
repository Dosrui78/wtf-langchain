from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="http://localhost:11434/v1", # 指向阿里接口
    model="qwen3:8b",
    streaming=True
)
response = llm.stream("你好，请介绍一下你自己")
for chunk in response:
    print(chunk.content, end="", flush=True)