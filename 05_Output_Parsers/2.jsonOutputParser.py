import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.messages import AIMessage

str_parser = StrOutputParser()
json_parser = JsonOutputParser()

model = ChatOpenAI(
    model="qwen3.5-122b-a10b",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_BASE_URL"),
)

first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}，刚生了{gender}，请起名，并封装为JSON格式返回给我，"
    "要求key是name，value就是起的名字。请严格按照格式要求返回，不要有其他内容。"
)

second_prompt = PromptTemplate.from_template("请对这个名字{name}进行点评，并给出评分（1-100）。")

chain = first_prompt | model | json_parser | second_prompt | model | str_parser

for chunk in chain.stream({
    "lastname": "王",
    "gender": "女孩"
}):
    print(chunk, end="", flush=True)
