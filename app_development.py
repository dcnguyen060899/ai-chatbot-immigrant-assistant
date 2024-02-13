import streamlit as st

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
# Import torch for datatype attributes
import torch

from llama_index.core import ServiceContext

# Import deps to load documents
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from pathlib import Path
import pypdf
import time
import os
import getpass
import random
import textwrap
from llama_index.legacy.readers.deeplake import DeepLakeReader
import openai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.legacy.vector_stores import DeepLakeVectorStore
from llama_index.storage.storage_context import StorageContext
import deeplake
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.embeddings import OpenAIEmbedding
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import MultiModalLLMCompletionProgram
from typing import List
from deeplake.core.vectorstore import VectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

class Information(BaseModel):
    """Data model for immigration related information"""
    url: str
    header: str
    content: str

class InformationList(BaseModel):
    """A list of immigration information for the model to use"""
    immigration_details: List[Information]

# initialize open ai agent model
openai.api_key = st.secrets["openai_api_key"]
# os.environ["ACTIVELOOP_TOKEN"] = ''

# Fetching secrets
# openai_api_key = st.secrets["openai_api_key"]
os.environ['ACTIVELOOP_TOKEN'] = st.secrets["active_loop_token"]
# os.environ['OPENAI_API_KEY'] = st.secrets['openai_api_key']

llm = OpenAI(model='gpt-4', temperature=.7)

reader = DeepLakeReader()
query_vector = [random.random() for _ in range(1536)]
documents = reader.load_data(
    query_vector=query_vector,
    dataset_path="hub://dcnguyen060899/settle_mind_db",
    limit=5,
)

dataset_path = 'settle_mind_db'
vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index_vector_store = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context)

immigration_query_engine = index_vector_store.as_query_engine(similarity_top_k=3, output_cls=InformationList)

immigration_query_engine_tool =QueryEngineTool(
    query_engine=immigration_query_engine,
    metadata=ToolMetadata(
        name="immigration_query_engine_tool",
        description=(
            "This tool is designed to assist with immigration-related queries, "
            "specifically focusing on study permits, visa applications, and university enrollment procedures for Canada. "
            "It can provide detailed steps, required documents, application deadlines, and eligibility criteria. "
            "Usage: input a query like 'How do I apply for a study permit in Canada?' or 'What are the requirements for a university application in Canada?'. "
            "Use this tool to retrieve specific policy information or procedural steps related to immigration and education in Canada."
        ),
    ),
)

# Define a Pydantic model for the immigration response
# class ImmigrationResponse(BaseModel):
#     document_requirements: str = ""
#     application_steps: str = ""
#     cultural_tips: str = ""
#     language_support: str = ""
#     additional_info: str = ""

# def immigration_assistance(nationality: str, user_input: str):
#     """
#     Given the nationality of a user and their specific query, this function returns relevant immigration assistance information.
#     This tool is designed to provide multilingual support for immigration-related queries, 
#     focusing on study permits, visa applications, and general immigration advice for Canada.

#     Parameters:
#     nationality (str): The nationality of the user or their preferred language for communication.
#     user_input (str): The specific query or request from the user.

#     Returns:
#     response: The assistance information related to the user's query.

#     Example:
#     >>> immigration_assistance("Vietnamese", "Yêu cầu tài chính tối thiểu để đủ điều kiện xin giấy phép học tập là gì?")
#     """

#     # Define the GPT-4 model
#     gpt4_language_model = OpenAI(language_model="gpt-4")

#     # Define prompt template based on nationality and user query
#     prompt_template_str = f"""
#     You are an expert in Canadian immigration policies and fluent in multiple languages, including the language preferred by someone from {nationality}.
#     Provide detailed information in a clear and user-friendly manner about the following query:

#     "{user_input}"

#     The response should be tailored to the nationality and language preferences of the user. 
#     Note: If user ask in other languages other than English, response in their languages
#     Example:
#     >>> User's Input: "Tôi là người Việt. Tôi cần tư vấn nhập cư."
#     >>> Response: "Tôi có thể giúp gì bạn?"
#     """

#     # Create a function to generate response based on the user's query and nationality
#     translation_completion_program = MultiModalLLMCompletionProgram.from_defaults(
#         output_parser=PydanticOutputParser(ImmigrationResponse),
#         prompt_template_str=prompt_template_str,
#         llm=gpt4_language_model,
#         verbose=True,
#     )
#     response = translation_completion_program()
    
#     return response

# # Tool for immigration assistance based on nationality and user query
# immigration_assistance_tool = FunctionTool.from_defaults(fn=immigration_assistance)

# service_context = ServiceContext.from_defaults(
#     chunk_size=1024,
#     llm=llm
#     )

# # And set the service context
# set_global_service_context(service_context)

agent = OpenAIAgent.from_tools(
  system_prompt="""You are an advanced AI trained to assist with immigration-related queries by accessing a specialized vector database. 
  Your role is to provide accurate and updated information from this database regarding study permits, visa applications, and university enrollment procedures for Canada.
   When users ask questions, you will directly consult the vector database to find the most relevant and current information available.
    Your responses should be based on the data stored in this database, ensuring they are precise and tailored to the users' needs.
  """,
  tools=[
        immigration_query_engine_tool
    ],
    llm=llm,
  verbose=True)

# Create the Streamlit UI components
st.title('👔 SettleSmart 🧩')

# Session state for holding messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('Input your prompt here')

if prompt:
   # Directly query the OpenAI Agent
   st.chat_message('user').markdown(prompt)
   st.session_state.messages.append({'role': 'user', 'content': prompt})

   response = agent.chat(prompt)
   final_response = response.response

   st.chat_message('assistant').markdown(final_response)
   st.session_state.messages.append({'role': 'assistant', 'content': final_response})
