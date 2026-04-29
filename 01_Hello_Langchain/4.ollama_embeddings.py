from langchain_community.embeddings import OllamaEmbeddings

# 初始化嵌入模型对象。默认使用：text-embedding-v1
embedding = OllamaEmbeddings(model="qwen3-embedding:4b")

# 测试
print(embedding.embed_query("我喜欢你"))
print(embedding.embed_query(["我们分手吧", "我稀饭你", "今晚去哪吃"]))