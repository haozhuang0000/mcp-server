import os
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv(override=True)

# Vector Database Config
EMB_URL = os.getenv("EMBEDDING_END_POINT", None)
MILVUS_URL = os.getenv("MILVUS_URL", None)
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", None)
MILVUS_PW = os.getenv("MILVUS_PW", None)

# MySQL Config
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

