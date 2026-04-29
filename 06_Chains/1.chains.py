import os
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSerializable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个边塞诗人，可以作诗。"),
        MessagesPlaceholder("history"),
        ("human", "请再来一首唐诗。"),
    ]
)

model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="http://dashscope.aliyuncs.com/compatible-mode/v1", # 指向阿里接口
    model="qwen3.5-122b-a10b",
    temperature=0
)

history_data = [
    ("human", "你来写一首唐诗"),
    ("ai", "床前明月光，疑似地上霜，举头望明月，低头思故乡"),
    ("human", "好诗好诗，再来一首"),
    ("ai", "锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦"),
]

# 组成链，要求每一个组件都是Runnable接口的子类
chain: RunnableSerializable = chat_prompt_template | model

res = chain.invoke({"history": history_data})
print(res.content)

print("="*20, "流式输出", "="*20)
for chunk in chain.stream({"history": history_data}):
    print(chunk.content, end="", flush=True)