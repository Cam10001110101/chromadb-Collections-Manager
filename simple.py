import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

# Load configuration settings from the .env file
load_dotenv()

CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH')

# Set up chromadb client
client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# Function to list collections
def list_collections():
    collections = client.list_collections()
    return [collection.name for collection in collections]

# Function to display collection details
def show_details(collection_name):
    try:
        collection = client.get_collection(collection_name)
        print(f"\nCollection Name: {collection_name}")
        print(f"Document Count: {collection.count()}")
        if collection.metadata:
            for key, value in collection.metadata.items():
                print(f"{key}: {value}")
        else:
            print("No metadata available.")
        
        # Display details of the documents in the collection
        raw_data = collection.get(limit=5)
        print("\nRaw Document Data (first 5 documents):", raw_data)
            
    except Exception as e:
        print(f"Error retrieving details for collection '{collection_name}': {str(e)}")

def main():
    print("Available Collections:")
    collections = list_collections()
    if not collections:
        print("No collections found.")
    else:
        for i, collection in enumerate(collections, 1):
            print(f"{i}. {collection}")
            show_details(collection)

if __name__ == "__main__":
    main()
