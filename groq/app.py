import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

## load GROQ API key and HUggingFace api key
os.environ["HUGGINGFACE_API_KEY"] = os.getenv('HUGGINGFACE_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("Chatgroq with llama3")

llm = ChatGroq(groq_api_key = groq_api_key,
               model_name = "llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
'''
Answer the question based only on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Question : {input}
'''
)

def vector_embedding():
    if "vector" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("/Users/pranshu/Documents/LangChain/groq/us_census")  # data ingestion
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200) # text spkitter
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectordb = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
    

prompt1 = st.text_input("Enter your que from docu")
if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector DB is ready")

# Only create chains if vectordb exists
if "vectordb" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    real_retriever = st.session_state.vectordb.as_retriever()
    retrieval_chain = create_retrieval_chain(real_retriever, document_chain)

    if prompt1:
        start = time.process_time()
        
        response = retrieval_chain.invoke({"input": prompt1})
        print("Response time:", time.process_time() - start)
        st.write(response['answer'])
        
        # streamlit expander
        with st.expander("document similartiy search"):
            # find the relevant chunks
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("------------------------------")
else:
    if prompt1:
        st.warning("Please click 'Documents Embedding' button first to initialize the vector database.")
