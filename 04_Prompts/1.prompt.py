import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="qwen3.5-122b-a10b",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_BASE_URL"),
    temperature=0
)

template = """
你是一位经验丰富的地理老师。请查询位于【{province}】的【{city}】对应的车牌代码简称。
如果【{city}】为空（例如直辖市），请直接返回该省份/直辖市每个简称对应的是该市的哪个区。
请简略输出，如果【{city}】不为空，仅仅输出简称即可，若为空用JSON格式输出对应关系"""

prompt = PromptTemplate.from_template(template)
prompt_str = prompt.format(province="上海", city="")
print(f"生成的Prompt: {prompt_str}")

response = model.invoke(prompt_str)
print(f"模型输出: {response.content}")
