import requests
import streamlit as st

def get_openai_response(input_text):
    response = requests.post(
        "http://localhost:8000/essay/invoke",
        json = {
            'input' : {'topic':input_text}
        }
    )
    
    # Check if request was successful
    if response.status_code != 200:
        return f"Error: Server returned status code {response.status_code}. Response: {response.text}"
    
    try:
        result = response.json()
        # LangServe returns the output directly or in 'output' key
        if 'output' in result:
            return result['output']
        return result
    except Exception as e:
        return f"Error parsing response: {str(e)}. Response text: {response.text}"

def get_ollama_response(input_text):
    response = requests.post(
        "http://localhost:8000/poem/invoke",
        json = {
            'input' : {'topic':input_text}
        }
    )
    result = response.json()
    # LangServe returns the output directly or in 'output' key
    if 'output' in result:
        return result['output']
    return result

## streamlit
st.title("Langchain Demo with LLAMA3.2 API")
input_text = st.text_input("Write an essay on")
input_text1 = st.text_input("Write a poem on")

if input_text:
    st.write(get_openai_response(input_text))

if input_text1:
    st.write(get_ollama_response(input_text1))