from pathlib import Path
from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path=Path(__file__).parent / "jay_lines.json",  # 文件路径
    jq_schema=".",  # jq语法
    text_content=False,  # 抽取的是否是字符串
    json_lines=True,  # 文件内容是否是json lines（每一行都是JSON的文件）
)

docs = loader.load()

for doc in docs:
    print(type(doc), doc)