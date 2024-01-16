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
import getpass
import random
import textwrap
from llama_index.readers.deeplake import DeepLakeReader
import openai
from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores import DeepLakeVectorStore
from llama_index.storage.storage_context import StorageContext
import deeplake
from llama_index.llms import OpenAI
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import OpenAIAgent

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

# Define a directory for storing uploaded files and initialize other components
UPLOAD_DIRECTORY = "/content/ai-chatbot-immigrant-assistant"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# Create the Streamlit UI components
st.title('👔 SettleSmart 🧩')

openai.api_key = ''
os.environ["ACTIVELOOP_TOKEN"] = 'eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwNTAyMTQ0MywiZXhwIjoxNzM2NjQzODI4fQ.eyJpZCI6ImRjbmd1eWVuMDYwODk5In0.jUIzxdEZQhsCffeVslM0o84NcVXUI_fzaZkQuYtH3sRDKqKuuNbCDSq_iBwNdR75Am8zfuYzjEM_eC5B-0DVgw'

reader = DeepLakeReader()
query_vector = [random.random() for _ in range(1536)]
documents = reader.load_data(
    query_vector=query_vector,
    dataset_path="hub://dcnguyen060899/SettleMind_AIChatbotImmigrantAssistant_Dataset",
    limit=5,
)

llm = OpenAI(temperature=0.8, model="gpt-4")
service_context = ServiceContext.from_defaults(llm=llm)

dataset_path = 'SettleMind_AIChatbotImmigrantAssistant_Dataset'
vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index_vector_store = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context)

query_engine = index_vector_store.as_query_engine()

query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="consult",
            description=(
                "Provides information about Study Permit for international students.\
                You are a professional international immigration consultant"
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]

agent = OpenAIAgent.from_tools(tools=query_engine_tools, verbose=True)

uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
upload_button = st.button('Upload')

if uploaded_file and upload_button:
  for file in uploaded_file:
  # Save the uploaded PDF to the directory
    with open(os.path.join(UPLOAD_DIRECTORY, file.name), "wb") as f:
      f.write(file.getbuffer())
    st.success("File uploaded successfully.")

# Session state for holding messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('Input your prompt here')

# Initialize a flag in Streamlit's session state to track if files are processed
if 'uploaded_files_processed' not in st.session_state:
    st.session_state.uploaded_files_processed = False

# Handling user input and file upload
if prompt:
    if uploaded_file and not st.session_state.uploaded_files_processed:
        # # Process the uploaded file(s)
        # for file in uploaded_file:
        #     with open(os.path.join(UPLOAD_DIRECTORY, file.name), "wb") as f:
        #         f.write(file.getbuffer())

        # Process documents and set up the query engine
        documents = SimpleDirectoryReader(UPLOAD_DIRECTORY).load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)

        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Query using Llama 7b indexed documents
        response = query_engine.query(prompt)

        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

        # Iterate over the response generator and concatenate the data
        response_content = ""
        for text in response.response_gen:
          response_content += text

        response = agent.chat(response_content)
        
        # Mark the uploaded files as processed
        st.session_state.uploaded_files_processed = True

    else:
        # Directly query the OpenAI Agent
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        response = agent.query(prompt)

        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

    # Reset the upload button state
    if upload_button:
        st.session_state.uploaded_files_processed = False
