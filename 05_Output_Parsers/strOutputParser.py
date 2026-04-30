import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

parser = StrOutputParser()

model = ChatOpenAI(
    model="qwen3.5-122b-a10b",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_BASE_URL"),
)

prompt = PromptTemplate.from_template("我邻居姓：{lastname}，刚生了{gender}，请起名，仅告知名字无需其他内容。")

chain = prompt | model | StrOutputParser() | model

res: str = chain.invoke({
    "lastname": "王",
    "gender": "女孩"
})
print(res)
