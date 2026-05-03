import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

model = ChatOpenAI(
    model="qwen3.5-122b-a10b",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_BASE_URL"),
)

# prompt = PromptTemplate.from_template(
#     "根据会话历史内容回答问题。对话历史：{chat_history}。用户提问：{input}，请回答"
# )

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个会说话的人，可以回答用户的问题。请根据会话历史内容回答问题。对话历史："),
        MessagesPlaceholder("chat_history"),
        ("human", "请回答这个问题：{input}"),
    ]
)

session_configs = {}

def get_history(session_id: str):
    if session_id not in session_configs:
        session_configs[session_id] = InMemoryChatMessageHistory()
    return session_configs[session_id]

def print_prompt(full_prompt):
    print("*" * 20, full_prompt.to_string(), "*" * 20)
    return full_prompt

base_chain = prompt | print_prompt | model | StrOutputParser()

chain_with_memory = RunnableWithMessageHistory(
    base_chain, 
    get_history, 
    input_messages_key="input", 
    history_messages_key="chat_history"
)

session_config = {"configurable": {"session_id": "1"}}

res = chain_with_memory.invoke(
    {"input": "小明有两个蛋"},
    config=session_config
)
print("第一次对话:", res)

res = chain_with_memory.invoke(
    {"input": "小陈只有一个蛋"},
    config=session_config
)
print("第二次对话:", res)

res = chain_with_memory.invoke(
    {"input": "小红没有蛋"},
    config=session_config
)
print("第三次对话:", res)


res = chain_with_memory.invoke(
    {"input": "他们三人总共有几个蛋"},
    config=session_config
)
print("第四次对话:", res)


