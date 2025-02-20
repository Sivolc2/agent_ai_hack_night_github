import os
import logging
from typing import List, Dict, Any
import yaml
from pathlib import Path
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes import config
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables."""
    load_dotenv()
    
    required_vars = {
        "WCD_URL": os.getenv("WCD_URL"),
        "WCD_API_KEY": os.getenv("WCD_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
    return required_vars

def setup_client(env_vars: Dict[str, str]) -> weaviate.WeaviateClient:
    """Setup Weaviate client."""
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=env_vars["WCD_URL"],
            auth_credentials=Auth.api_key(env_vars["WCD_API_KEY"]),
            headers={
                "X-OpenAI-Api-Key": env_vars["OPENAI_API_KEY"]
            }
        )
        logger.info("Successfully connected to Weaviate!")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {str(e)}")
        raise

def create_profile_collection(client: weaviate.WeaviateClient) -> weaviate.collections.Collection:
    """Create collection for user profiles."""
    collection_name = "UserProfile"
    
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)
        
    return client.collections.create(
        name=collection_name,
        properties=[
            config.Property(name="name", data_type=config.DataType.TEXT),
            config.Property(name="email", data_type=config.DataType.TEXT),
            config.Property(name="interests", data_type=config.DataType.TEXT_ARRAY),
            config.Property(name="skills", data_type=config.DataType.TEXT_ARRAY),
            config.Property(name="experience", data_type=config.DataType.TEXT),
        ],
        vectorizer_config=config.Configure.Vectorizer.text2vec_openai(),
        generative_config=config.Configure.Generative.openai()
    )

def create_document_collection(client: weaviate.WeaviateClient) -> weaviate.collections.Collection:
    """Create collection for text documents."""
    collection_name = "Document"
    
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)
        
    return client.collections.create(
        name=collection_name,
        properties=[
            config.Property(name="content", data_type=config.DataType.TEXT),
            config.Property(name="title", data_type=config.DataType.TEXT),
            config.Property(name="chunk_index", data_type=config.DataType.INT),
        ],
        vectorizer_config=config.Configure.Vectorizer.text2vec_openai(),
        generative_config=config.Configure.Generative.openai()
    )

def chunk_text(text: str, chunk_size: int = 150, overlap: int = 25) -> List[str]:
    """Chunk text into smaller pieces."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def load_user_profiles(collection: weaviate.collections.Collection, yaml_path: str):
    """Load user profiles from YAML file."""
    with open(yaml_path, 'r') as f:
        profiles = yaml.safe_load(f)
    
    with collection.batch.dynamic() as batch:
        for profile in profiles:
            batch.add_object(
                properties=profile
            )
    logger.info(f"Loaded {len(profiles)} user profiles")

def load_text_document(collection: weaviate.collections.Collection, text_path: str):
    """Load and chunk text document."""
    with open(text_path, 'r') as f:
        text = f.read()
    
    chunks = chunk_text(text)
    
    with collection.batch.dynamic() as batch:
        for i, chunk in enumerate(chunks):
            batch.add_object(
                properties={
                    "content": chunk,
                    "title": Path(text_path).stem,
                    "chunk_index": i
                }
            )
    logger.info(f"Loaded {len(chunks)} text chunks")

def main():
    """Main execution function."""
    client = None
    try:
        # Setup
        env_vars = load_environment()
        client = setup_client(env_vars)
        
        # Create collections
        profiles_collection = create_profile_collection(client)
        documents_collection = create_document_collection(client)
        
        # Load data
        load_user_profiles(profiles_collection, "data/user_profile.yaml")
        load_text_document(documents_collection, "data/raw_text.txt")
        
        logger.info("Successfully loaded all data into Weaviate!")
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
    finally:
        if client:
            client.close()
            logger.info("Weaviate client connection closed")

if __name__ == "__main__":
    main()
