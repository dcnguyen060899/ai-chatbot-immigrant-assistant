import streamlit as st

# Import transformer classes for generaiton
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
# Import torch for datatype attributes
import torch
# Import the prompt wrapper...but for llama index
from llama_index.prompts.prompts import SimpleInputPrompt
# Import the llama index HF Wrapper
from llama_index.llms import HuggingFaceLLM
# Bring in embeddings wrapper
from llama_index.embeddings import LangchainEmbedding
# Bring in HF embeddings - need these to represent document chunks
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# Bring in stuff to change service context
from llama_index import set_global_service_context
from llama_index import ServiceContext
# Import deps to load documents
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from pathlib import Path
import pypdf
import time
import os
from langdetect import detect

def detect_language(text):
  try:
    return detect(text)
  except:
    return "unknown"

# Define variable to hold llama2 weights namingfiner
name = "meta-llama/Llama-2-7b-chat-hf"
# Set auth token variable from hugging face
auth_token = "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"

@st.cache_resource(hash_funcs={AutoModelForCausalLM: id, AutoTokenizer: id})
def get_tokenizer_model():
    global model, tokenizer

    # Check if the model and tokenizer have already been loaded
    if 'model' not in globals() or 'tokenizer' not in globals():
        print("Loading model and tokenizer...")

        # Define the model name and authentication token
        name = "meta-llama/Llama-2-7b-chat-hf"
        auth_token = "your_auth_token_here"

        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(name, 
                                                  cache_dir='/content/drive/My Drive/LLM Deployment/LLM Deployment/', 
                                                  use_auth_token=auth_token)

        # Create model
        model = AutoModelForCausalLM.from_pretrained(name, 
                                                     cache_dir='/content/drive/My Drive/LLM Deployment/LLM Deployment/', 
                                                     use_auth_token=auth_token, 
                                                     torch_dtype=torch.float16,
                                                     rope_scaling={"type": "dynamic", "factor": 2}, 
                                                     load_in_8bit=True)
    else:
        print("Model and tokenizer are already loaded.")

    return model, tokenizer

# Get the model and tokenizer
model, tokenizer = get_tokenizer_model()


# Initialize the SimpleInputPrompt with an empty template
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

# Streamlit UI to let the user update the system prompt
# Start with an empty string or a default prompt
default_prompt = ""
user_system_prompt = st.text_area("How can I best assist you?", value="", height=100)
update_button = st.button('Request')

# Initialize the llm object with a placeholder or default system prompt
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=1024,
    system_prompt="",  # Placeholder if your initial prompt is empty
    query_wrapper_prompt=query_wrapper_prompt,  # Placeholder string
    model=model,
    tokenizer=tokenizer
)

# Function to update the system prompt and reinitialize the LLM with the new prompt
def update_system_prompt(new_prompt):
    global llm
    llm.system_prompt = new_prompt


if update_button:
    # Update the system prompt and reinitialize the LLM
    update_system_prompt(user_system_prompt)
    st.success('Requested')

# Create and dl embeddings instance
embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Create new service context instance where llm object is communicate with retrieval augmented system (VectorStoreIndex)
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)

# And set the service context
set_global_service_context(service_context)

# Define a directory for storing uploaded files
UPLOAD_DIRECTORY = "/content/ai-chatbot-immigrant-assistant"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

st.title('PDF Upload and Query Interface')

# File uploader allows user to add PDF
uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
upload_button = st.button('Upload')

if uploaded_file and upload_button:
  for file in uploaded_file:
  # Save the uploaded PDF to the directory
    with open(os.path.join(UPLOAD_DIRECTORY, file.name), "wb") as f:
      f.write(file.getbuffer())
    st.success("File uploaded successfully.")

documents = SimpleDirectoryReader(UPLOAD_DIRECTORY).load_data()
index = VectorStoreIndex.from_documents(documents)


# Setup index query engine using LLM
query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)

# Create centered main title
st.title('👔 SettleSmart 🧩')

# setup a session to hold all the old prompt
if 'messages' not in st.session_state:
  st.session_state.messages = []

# print out the history message
for message in st.session_state.messages:
  st.chat_message(message['role']).markdown(message['content'])


# Create a text input box for the user
# If the user hits enter
prompt = st.chat_input('Input your prompt here')

if prompt:
  detected_language = detect_language(prompt)
  st.write(f"Detected language: {detected_language}")

  st.chat_message('user').markdown(prompt)
  st.session_state.messages.append({'role': 'user', 'content': prompt})

  response = query_engine.query(prompt)

  st.chat_message('assistant').markdown(response)
  st.session_state.messages.append(
      {'role': 'assistant', 'content': response}
  )
