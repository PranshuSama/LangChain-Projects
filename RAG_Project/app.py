import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time


from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to .env or export it in your shell.")

if "vector" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings()
    st.session_state.loader = WebBaseLoader("https://docs.langchain.com/langsmith/home")
    st.session_state.docs = st.session_state.loader.load()
    
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size =1000,chunk_overlap = 200)
    st.session_state.final_document = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_document, st.session_state.embeddings)
    


st.title("ChatGroq Demo")
llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
'''
    Answer the question based on the provided text only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question : {input}
    
'''
)

document_chain = create_stuff_documents_chain(llm,prompt)
retriever = st.session_state.vectors.as_retriever()
retriever_chain = create_retrieval_chain(retriever,document_chain)

prompt = st.text_input("Input your prompt here")

if prompt:
    start = time.process_time()
    response = retriever_chain.invoke({"input" : prompt})
    print("Response time :",time.process_time() - start)
    st.write(response['answer'])
    
    # with streamlit document expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-------------------------------")
        
    
