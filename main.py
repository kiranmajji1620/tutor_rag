import streamlit as st
import os

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from prompts import prompt
from model_setup import llm
from embed_setup import embed_data

def retrieve_results(vectors, query):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input' : query})
    return response["answer"]  
def main():
    st.title("Embedding and Retrieval with Streamlit")

    if 'vectors' not in st.session_state:
        with st.spinner("Loading and embedding data..."):
            st.session_state.vectors = embed_data()
        st.success("Data loaded and embedded!")

    vectors = st.session_state.vectors

    query = st.text_input("Enter your query:")
    if query:
        results = retrieve_results(vectors, query)
        st.write("Results:", results)
        
if __name__ == "__main__":
    main()

