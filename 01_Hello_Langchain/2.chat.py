from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
load_dotenv()

messages = [
    SystemMessage(content="你是一个边塞诗人"),
    AIMessage(content="葡萄美酒夜光杯，欲饮琵琶马上催。醉卧沙场君莫笑，古来征战几人回？"),
    HumanMessage(content="请参照刚才那首诗的形式继续写一首唐诗"),
]
llm = ChatOpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="http://localhost:11434/v1", # 指向阿里接口
    model="qwen3:8b",
    streaming=True
)
response = llm.stream(messages)
for chunk in response:
    print(chunk.content, end="", flush=True)