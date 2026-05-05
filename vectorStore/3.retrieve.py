from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.output_parsers import StrOutputParser

input_text = "怎么减肥呢"

vector_store = InMemoryVectorStore(embedding=DashScopeEmbeddings(model="text-embedding-v4"))

model = ChatOpenAI(
    model="qwen3.5-122b-a10b",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_BASE_URL"),
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "根据下方的上下文回答问题，如果上下文中没有信息，请直接回答‘我不知道’。上下文：{context}"),
    ("human", "{question}")
])

vector_store.add_texts(["减肥就是要运动，控制饮食", "在减脂期吃东西很重要，清单少油少盐控制卡路里摄入并运动起来", "游泳是很好的运动", "跑步是很好的运动"])

# 检索向量库
res = vector_store.similarity_search(input_text, k=2)
referen_text = "["
for doc in res:
    referen_text += doc.page_content
referen_text += "]"

def print_prompt(prompt):
    print(prompt.to_string())
    print("=" * 20)
    return prompt

chain = prompt | print_prompt | model | StrOutputParser()

print(chain.invoke({"context": referen_text, "question": input_text}))
