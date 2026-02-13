import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_objectbox.vectorstores import ObjectBox

from dotenv import load_dotenv
load_dotenv()

os.environ['HUGGINGFACE_API_KEY'] = os.getenv('HUGGINGFACE_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("ObjectBox VectorStoreDB ChatGroq Demo")

llm = ChatGroq(
    groq_api_key = groq_api_key,
    model_name = "llama-3.1-8b-instant"
)

prompt = ChatPromptTemplate.from_template(
    '''
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions : {input}
    '''
)

## vectors embeddings

def vector_embedding():
    
    if "vectors" not in st.session_state:
        
        st.session_state.embeddings = HuggingFaceEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("/Users/pranshu/Documents/LangChain/us_census") ## data ingestion
        st.session_state.docs = st.session_state.loader.load() ## loading the data
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size =1000,chunk_overlap = 200)
        st.session_state.final_document = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors = ObjectBox.from_documents(st.session_state.final_document, st.session_state.embeddings,embedding_dimensions = 768)


input_prompt = st.text_input("Enter your question from documents")

if st.button("Document embeddings"):
    vector_embedding()
    st.write("ObjectBox DB is ready")
    
import time
if input_prompt:
    start = time.process_time()
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    
    response = retrieval_chain.invoke({"input" : input_prompt})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])
    
    # with streamlit document expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-------------------------------")