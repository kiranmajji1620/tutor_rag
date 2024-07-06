from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader


def embed_data():
    loader = PyPDFLoader("NIPS-2017-attention-is-all-you-need-Paper.pdf")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

    docs = loader.load()
    final_docs = splitter.split_documents(docs)
    vectors = Chroma.from_documents(final_docs, embeddings)
    return vectors