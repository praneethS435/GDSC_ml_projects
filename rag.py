from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def get_rag_chain(vector_store):
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)
    return qa_chain
