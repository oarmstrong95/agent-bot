# /**********************************************************************************************************
# Step 1: Import necessary libraries
# /**********************************************************************************************************
# Import general modules in 
import os
from dotenv import load_dotenv

# Import modules required for API call
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate

# Load modules required for RAG system
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader

# Load modules required for Google Search API
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain.tools import StructuredTool
from pydantic import BaseModel

# Load modules required for the chain
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Load modules required for the agent
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

# Load in yaml file for environment level variables
load_dotenv('env')

# Access the OpenAI API key from the environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# /**********************************************************************************************************
# Create an instance of the GoogleSearchAPIWrapper class
# /**********************************************************************************************************
# Intialize the GoogleSearchAPIWrapper class
search = GoogleSearchAPIWrapper(k=1)

# Define the search function class
class GoogleSearchInput(BaseModel):
    query: str

# Define the tool
search = StructuredTool(
    name="google_search",
    description="Search Google for top result",
    func=search.run,
    args_schema=GoogleSearchInput
)

# /**********************************************************************************************************
# Set up the retriever
# /**********************************************************************************************************
# Load the OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Process documents to use later in OpenAI model
loader = CSVLoader(file_path='transactions.csv')
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
docs = text_splitter.split_documents(data)

# Create vector store
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()

# /**********************************************************************************************************
# Create retriever tool
# /**********************************************************************************************************
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "transaction_database_search",
    "Search the database to find categories for known transactions",
)

# Create a list of tools
tools = [search, retriever_tool]

# /**********************************************************************************************************
# Create the agent
# /**********************************************************************************************************
# Create an instance of the ChatOpenAI class
llm = ChatOpenAI(temperature=0.0)

# Define the prompt
prompt = ChatPromptTemplate(
    [
        ("system", "You are an AI developed to classify banking transactions into a pre-defined list of categories. You have access to a database of transactions and can search for relevant information."),
        ("user", "Instructions are to classify the transactions so that every transaction is mapped to a category. The response should be in json format"),
        ("assistant", "Understood, are there any constraints"),
        ("user", "Yes. Please reference the database of previous transactions for context. The categories must be either 'Rent', 'Council tax', 'Food', 'Internet', 'Phone', 'Transport', 'Elecricity', 'Clothes', 'Gym', 'Holiday', 'Savings'. If its a new transaction, please search the web for more information and try to classify it into a category based on the first result. Use the search to try to find the category of the transaction. If you cant infer the category, please return 'Other'."),
        ("user", "Banking transaction: {input}"),
        ("assistant", "{agent_scratchpad}")
    ]
)

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Define the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Invoke the agent
agent_executor.invoke({"input": "Zara"})
agent_executor.invoke({"input": "Briki Coffee"})