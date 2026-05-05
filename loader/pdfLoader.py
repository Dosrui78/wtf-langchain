from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path=Path(__file__).parent / "最新微信养号和防封全教程.pdf",
    mode="single"
)

i = 0
for doc in loader.lazy_load():
    i += 1
    print(doc)
    print("=" * 20, i)