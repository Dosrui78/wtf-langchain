from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader(file_path=Path(__file__).parent / "text.txt", encoding="utf-8")
docs = loader.load()

spliter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", "。", "", ""],  # 分割符
    chunk_size=500,  # 分段的最大字符数
    chunk_overlap=20,  # 分段之间允许的重叠字符数
    length_function=len  # 计算长度的函数
)

docs = spliter.split_documents(docs)
print(docs)
for doc in docs:
    print("=" * 100)
    print(doc)
    print("=" * 100)