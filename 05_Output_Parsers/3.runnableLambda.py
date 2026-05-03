import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


model = ChatOpenAI(
    model="qwen3.5-122b-a10b",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_BASE_URL"),
)

str_parser = StrOutputParser()

prompt1 = PromptTemplate.from_template(
    "我邻居姓{lastname}，刚生了{gender}，请帮忙起个名字，仅生成一个名字，不用其他无用信息",
)

prompt2 = PromptTemplate.from_template(
    "姓名{name}，请解释它的含义"
)

my_func = RunnableLambda(lambda ai_message: {'name': ai_message.content.strip()})

chain = prompt1 | model | my_func | prompt2 | model | str_parser

for chunk in chain.stream({'lastname': '张', 'gender': '女孩'}):
    print(chunk, end='', flush=True)