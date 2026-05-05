from pathlib import Path
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader

vector_store = InMemoryVectorStore(embedding=DashScopeEmbeddings())

loader = CSVLoader(file_path=Path(__file__).parent / "stu.csv", encoding="utf-8", source_column="python")

docs = loader.load()

# 向量存储的新增、删除、检索
vector_store.add_documents(
    documents=docs,             # 被添加的文档
    ids=["id" + str(i) for i in range(1, len(docs) + 1)]  # 给添加的文档提供id（字符串）
)

# 向量存储的删除 传入[id, id...]
# vector_store.delete(ids=["id1","id2","id3"])

# 检索 返回类型[Document, Document...]
result = vector_store.similarity_search("Python是否简单易学呀", k=3)

print(result)