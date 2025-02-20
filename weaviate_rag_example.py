import os
import weaviate
from weaviate.classes import init, config
from weaviate.collections import Collection
from dotenv import load_dotenv
from typing import List
import requests
import re

# Load environment variables
load_dotenv()

def setup_weaviate_client():
    """Setup and return a Weaviate Cloud client."""
    if not os.getenv("WEAVIATE_URL") or not os.getenv("WEAVIATE_API_KEY"):
        raise ValueError(
            "Please set WEAVIATE_URL and WEAVIATE_API_KEY environment variables. "
            "You can get these from your Weaviate Cloud Console."
        )
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "Please set OPENAI_API_KEY environment variable. "
            "You can get this from your OpenAI dashboard."
        )
    
    # Create authentication configuration
    auth_config = init.ApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))
    
    # Connect to Weaviate
    client = weaviate.connect_to_weaviate(
        url=os.getenv("WEAVIATE_URL"),
        auth_credentials=auth_config,
        headers={
            "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
        }
    )
    
    # Verify connection
    try:
        client.collections.get("_schema")  # Simple check to verify connection
        print("Successfully connected to Weaviate Cloud!")
    except Exception as e:
        raise ConnectionError(
            "Failed to connect to Weaviate Cloud. Please check your credentials and URL. "
            f"Error: {str(e)}"
        )
    
    return client

def download_and_chunk(src_url: str, chunk_size: int = 150, overlap_size: int = 25) -> List[str]:
    """Download text content and chunk it into smaller pieces."""
    response = requests.get(src_url)
    source_text = re.sub(r"\s+", " ", response.text)
    text_words = re.split(r"\s", source_text)
    
    chunks = []
    for i in range(0, len(text_words), chunk_size):
        chunk = " ".join(text_words[max(i - overlap_size, 0): i + chunk_size])
        chunks.append(chunk)
    return chunks

def create_collection(client) -> Collection:
    """Create a Weaviate collection for storing document chunks."""
    collection_name = "DocumentChunks"
    
    # Delete collection if it exists
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)
    
    # Create new collection
    chunks = client.collections.create(
        name=collection_name,
        properties=[
            config.Property(
                name="content",
                data_type=config.DataType.TEXT
            ),
            config.Property(
                name="title",
                data_type=config.DataType.TEXT
            ),
            config.Property(
                name="chunk_index",
                data_type=config.DataType.INT
            ),
        ],
        vectorizer_config=config.Configure.Vectorizer.text2vec_openai(),
        generative_config=config.Configure.Generative.openai()
    )
    return chunks

def import_data(collection: Collection, chunks: List[str], title: str):
    """Import chunked data into Weaviate."""
    chunks_list = []
    for i, chunk in enumerate(chunks):
        data_properties = {
            "title": title,
            "content": chunk,
            "chunk_index": i
        }
        data_object = config.DataObject(properties=data_properties)
        chunks_list.append(data_object)
    
    collection.data.insert_many(chunks_list)

def demonstrate_rag(collection: Collection):
    """Demonstrate different RAG capabilities."""
    print("\n=== Single Object Generation ===")
    response = collection.generate.fetch_objects(
        limit=1,
        single_prompt="Write a haiku based on this text: {content}"
    )
    print("\nHaiku generation:")
    for obj in response.objects:
        print(f"\nChunk index: [{obj.properties['chunk_index']}]")
        print(obj.generated)

    print("\n=== Semantic Search with Group Generation ===")
    response = collection.generate.near_text(
        query="version control systems",
        limit=2,
        grouped_task="Write a tweet explaining these concepts. Use emojis and make it engaging."
    )
    print("\nGenerated tweet:")
    print(response.generated)

def main():
    # Initialize Weaviate client
    client = setup_weaviate_client()
    
    # Create collection
    collection = create_collection(client)
    
    # Download and chunk sample text
    sample_url = "https://raw.githubusercontent.com/progit/progit2/main/book/01-introduction/sections/what-is-git.asc"
    chunks = download_and_chunk(sample_url)
    
    # Import data
    import_data(collection, chunks, "Git Introduction")
    
    # Demonstrate RAG capabilities
    demonstrate_rag(collection)

if __name__ == "__main__":
    main() 