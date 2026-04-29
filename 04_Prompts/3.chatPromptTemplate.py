import os
from langchain_openai import ChatOpenAI
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

prompt_text = chat_prompt_template.invoke({"history": history_data}).to_string()
res = model.invoke(prompt_text)
print(res.content, type(res))