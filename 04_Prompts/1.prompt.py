import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
template = """
我的邻居姓{lastname}，刚生了个{gender}，帮忙起名字，请简略回答。
"""

prompt = PromptTemplate.from_template(template)
model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="http://dashscope.aliyuncs.com/compatible-mode/v1", # 指向阿里接口
    model="qwen3.5-122b-a10b",
    temperature=0
)
chain = prompt | model
response = chain.invoke({"lastname":"王", "gender":"女孩"})
print(response.content)