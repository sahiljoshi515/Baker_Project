from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API settings
    api_title: str = "Digital Collections Explorer API"
    api_description: str = "API for searching collections using CLIP embeddings"
    api_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # CLIP model settings
    clip_model: str = "openai/clip-vit-base-patch32"
    device: str = "cuda"
    batch_size: int = 32
    
    # Data directories
    collection_type: str = "photographs" # this is the default collection type, will be overwritten by config.json
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    embeddings_dir: str = "data/embeddings"
    thumbnails_dir: str = "data/thumbnails"