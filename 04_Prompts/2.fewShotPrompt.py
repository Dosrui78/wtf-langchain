from langchain_core.prompts import few_shot
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="qwen3.5-122b-a10b",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_BASE_URL"),
)

example_prompt = PromptTemplate.from_template("""
姓氏:{surnames}
名人:{Celebrities}
""")

example_data = [
    {"surnames": "诸葛", "Celebrities": "诸葛亮"},
    {"surnames": "张", "Celebrities": "张居正"},
    {"surnames": "欧阳", "Celebrities": "欧阳修"},
    {"surnames": "白", "Celebrities": "白居易"},
    {"surnames": "李", "Celebrities": "李白"},
    {"surnames": "曾", "Celebrities": "曾国藩"},
    {"surnames": "左", "Celebrities": "左宗棠"},
    {"surnames": "朱", "Celebrities": "朱元璋"},
    {"surnames": "毛", "Celebrities": "毛泽东"},
    {"surnames": "冼", "Celebrities": "冼星海"},
    {"surnames": "赵", "Celebrities": "赵云"},
]

few_shot_prompt = FewShotPromptTemplate(
    examples=example_data,
    example_prompt=example_prompt,
    input_variables=["input_surname"],
    prefix="给出定义词对应的名人的名字，示例如下：",
    suffix="请基于示例词告诉我，提起【{input_surname}】这个姓氏，你首先想到的是？",
)

prompt_text = few_shot_prompt.format(input_surname="马")
response = model.invoke(prompt_text)
print(response.content)