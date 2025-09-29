import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

load_dotenv(override=True)

@dataclass
class VectorDBConfig:
    """Configuration for vector database operations"""
    embedding_url: Optional[str] = None
    embedding_dim: int = 4096
    default_collection: str = None

@dataclass
class MilvusConfig:
    """Milvus-specific configuration"""
    url: Optional[str] = None
    db_name: str = "default"
    password: Optional[str] = None

@dataclass
class MySQLConfig:
    """MySQL configuration"""
    user: Optional[str] = None
    password: Optional[str] = None
    host: Optional[str] = None
    port: int = 3306
    database: Optional[str] = None

# Vector Database Config
EMB_URL = os.getenv("EMBEDDING_END_POINT", None)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "4096"))
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "annual_report_0821")

# Milvus Config
MILVUS_URL = os.getenv("MILVUS_URL", None)
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "default")
MILVUS_PW = os.getenv("MILVUS_PW", None)

# MySQL Config
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

# Configuration instances
VECTOR_DB_CONFIG = VectorDBConfig(
    embedding_url=EMB_URL,
    embedding_dim=EMBEDDING_DIM,
    default_collection=DEFAULT_COLLECTION
)

MILVUS_CONFIG = MilvusConfig(
    url=MILVUS_URL,
    db_name=MILVUS_DB_NAME,
    password=MILVUS_PW
)

MYSQL_CONFIG = MySQLConfig(
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    host=MYSQL_HOST,
    port=MYSQL_PORT,
    database=MYSQL_DATABASE
)

