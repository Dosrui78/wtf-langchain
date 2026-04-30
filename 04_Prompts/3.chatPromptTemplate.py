import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

model = ChatOpenAI(
    model="qwen3.5-122b-a10b",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_BASE_URL"),
    temperature=0
)

chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个边塞诗人，可以作诗。请牢记：你和用户交流时，请用文言文回答。如果用户回复了多行诗句，请用文言文在诗句下方简短评述一下。"),
    MessagesPlaceholder("history"),
    ("human", "好诗好诗，请再来一首唐诗。"),
])

history_data = [
    ("human", "你来写一首唐诗"),
    ("ai", "床前明月光，疑似地上霜，举头望明月，低头思故乡。"),
    ("human", "好诗好诗，再来一首"),
    ("ai", "锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。")
]

prompt_text = chat_prompt_template.invoke({"history": history_data}).to_string()
res = model.invoke(prompt_text)
print(res.content, type(res))