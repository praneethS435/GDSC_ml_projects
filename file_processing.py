from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def load_documents(file_path):
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format")
    
    documents = loader.load()
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_documents(documents)


def create_vector_store(docs):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)


def process_file(file_path):
    docs = load_documents(file_path)
    split_docs = split_documents(docs)
    vector_store = create_vector_store(split_docs)
    return vector_store
