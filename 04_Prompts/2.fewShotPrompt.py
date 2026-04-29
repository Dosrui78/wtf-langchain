import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="http://dashscope.aliyuncs.com/compatible-mode/v1", # 指向阿里接口
    model="qwen3.5-122b-a10b",
    temperature=0
)

example_template = PromptTemplate.from_template("""
单词：{word}
反义词：{antoym}
""")

example_data = [
    {"word": "大", "antoym": "小"},
    {"word": "长", "antoym": "短"},
    {"word": "多", "antoym": "少"},
    {"word": "好", "antoym": "坏"},
    {"word": "开", "antoym": "关"},
]

few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_template, # 示例数据的提示词模板
    examples=example_data, # 示例数据
    prefix="给出定义词的反义词，有如下示例：", # 前缀，用户提供
    suffix="基于示例词告诉我：{input_word}的反义词是？", # 后缀，用户提供
    input_variables=['input_word'] # 输入变量
) 

# 获得最终提示词
prompt_str = few_shot_prompt.format(input_word="冷")
print(prompt_str)
response = model.invoke(input=prompt_str)
print(response)