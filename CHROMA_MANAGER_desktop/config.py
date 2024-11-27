# config.py

# Add these imports at the top of the file
import requests
from pymongo import MongoClient
from neo4j import GraphDatabase
import chromadb
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Database connection parameters
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PW = os.getenv('NEO4J_PW')
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
CHROMADB_HOST = os.getenv('CHROMADB_HOST', 'localhost')
CHROMADB_PORT = int(os.getenv('CHROMADB_PORT', '8000'))
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://192.168.86.100:11434')

# Define OLLAMA_MODELS and EMBEDDING_MODELS
OLLAMA_MODELS = [
    'llama3.2:3b-instruct-q5_0',
    'llama3.2',
    'llama3.2:3b',
    'llama3.1:8b-instruct-q5_0',
    'llama3.1:latest',
    'llama3-chatqa:latest',
    'llama3-groq-tool-use:latest',
    'llama3.1:70b-instruct-q2_K',
    'llama3.1:8b-instruct-fp16',
    'llama3:latest',
    'bakllava:latest',
    'codegemma:latest',
    'codellama:latest',
    'codestral:latest',
    'gemma2:2b',
    'gemma2:27b',
    'gemma2:9b-instruct-fp16',
    'gemma2:latest',
    'llava-llama3:latest',
    'llava:latest',
    'mistral-small',
    'mistral-nemo:12b-instruct-2407-fp16',
    'mistral-nemo:latest',
    'nemotron-mini',
    'phi3.5:latest',
    'phi3:3.8b-instruct',
    'phi3:latest',
    'solar-pro:22b-preview-instruct-q3_K_S',
    'solar-pro:latest',
    'sqlcoder:latest',
    'zephyr:latest'
]

EMBEDDING_MODELS = [
    'nomic-embed-text:latest',
    'all-minilm:latest',
    'snowflake-arctic-embed:latest'
]

# New GROQ_MODELS
GROQ_MODELS = [
    "llama-3.1-70b-versatile",    # Llama 3.1 70B (Meta)
    "llama-3.1-8b-instant",       # Llama 3.1 8B (Meta)
    "llama3-groq-70b-8192-tool-use-preview",  # Llama 3 Groq 70B Tool Use (Groq)
    "llama3-groq-8b-8192-tool-use-preview",   # Llama 3 Groq 8B Tool Use (Groq)
    "llama3-70b-8192",            # Meta Llama 3 70B (Meta)
    "llama3-8b-8192",             # Meta Llama 3 8B (Meta)
    "llama-guard-3-8b",           # Llama Guard 3 8B (Meta)
    "llava-v1.5-7b-4096-preview", # LLaVA 1.5 7B (Haotian Liu)
    "gemma2-9b-it",               # Gemma 2 9B (Google)
    "gemma-7b-it",                # Gemma 7B (Google)
    "mixtral-8x7b-32768",         # Mixtral 8x7B (Mistral)
]

# New OPENAI_MODELS
OPENAI_MODELS = [
    "gpt-4o-mini",  # Smaller and more affordable version for simpler tasks
    "gpt-4o-mini-2024-07-18",  # Latest snapshot of gpt-4o-mini
    "gpt-4o",  # Flagship high-intelligence model for multi-step tasks
    "chatgpt-4o-latest",  # Continuously updated version of GPT-4o in ChatGPT
    "gpt-4o-2024-08-06",  # Latest snapshot with support for Structured Outputs
    "gpt-4o-2024-05-13",  # Current version of GPT-4o
    "openai-o1-preview",  # New preview model focused on reasoning tasks
    "openai-o1-mini",  # Lightweight version designed for smaller reasoning tasks
    "gpt-4-turbo",  # Model with vision capabilities, faster and cheaper than GPT-4
    "gpt-4-turbo-2024-04-09",  # Latest snapshot of GPT-4 Turbo
    "gpt-4-turbo-preview",  # Preview model for testing new updates to gpt-4-turbo
    "gpt-4-0613",  # Snapshot of GPT-4 with enhanced function calling
    "gpt-3.5-turbo",  # Cheaper, slightly less capable version compared to gpt-4o
    "gpt-3.5-turbo-0125",  # Latest snapshot of GPT-3.5 Turbo
    "gpt-3.5-turbo-instruct",  # Completion-based behavior of GPT-3.5 Turbo
]

def check_ollama():
    try:
        response = requests.get(OLLAMA_URL)
        return response.status_code == 200
    except:
        return False

def check_chromadb():
    try:
        client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
        client.heartbeat()
        return True
    except:
        return False

def check_mongodb():
    try:
        client = MongoClient(MONGODB_URI)
        client.server_info()
        return True
    except:
        return False

def check_neo4j():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PW))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        return True
    except:
        return False
