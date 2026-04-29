from langchain_community.embeddings import DashScopeEmbeddings

# 初始化嵌入模型对象。默认使用：text-embedding-v1
embedding = DashScopeEmbeddings()

# 测试
print(embedding.embed_query("我喜欢你"))
print(embedding.embed_query(["我们分手吧", "我稀饭你", "今晚去哪吃"]))