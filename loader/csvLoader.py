from pathlib import Path
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path=Path(__file__).parent / "stu.csv", 
                encoding="utf-8", 
                csv_args={
                    "delimiter": "，", # 指定分隔符
                    "quotechar": "'",  # 指定带有分隔符文本的引号包围是单引号还是双引号
                    "fieldnames": ["name", "age", "gender", "hobbies"] # 指定列名
                }) 

# 批量加载 .load() -> [Document, Document, Document...]
# docs = loader.load()

# for doc in docs:
#     print(type(doc), doc)


# 懒加载 .lazy_load() -> [Doc1, Doc2, ...]
docs = loader.lazy_load()

for doc in docs:
    print(type(doc), doc)