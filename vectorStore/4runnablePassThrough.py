import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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

# langchain中向量存储对象，有一个方法：as_retriever(search_kwargs={"k":2})，可以返回一个Runnable接口的子类实例对象
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# chain
# chain = retriever | prompt | model | StrOutputParser()

def format_func(docs):
    """ 
    格式化检索到的文档，将文档中的元数据提取出来拼接成字符串
    """
    if not docs:return "无相关参考资料"
    re_text = "["
    for doc in docs:
        re_text += doc.page_content
    re_text += "]"
    return re_text

def print_prompt(prompt):
    print(prompt.to_string())
    print("=" * 20)
    return prompt

chain = (
    {"question": RunnablePassthrough(), "context": retriever | format_func} | prompt | print_prompt |model | StrOutputParser()
)

"""
retriever：
    - 输入：用户的问题（字符串） str 
    - 输出：向量库的检索到的文档（Document对象） list[Document]
prompt：
    - 输入：用户提问 + 向量库检索到的文档  dict
    - 输出：完整的提示词  PromptValue
"""

print(chain.invoke(input_text))