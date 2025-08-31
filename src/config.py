import os
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv(override=True)

# LiteLLM Configuration
EMB_URL = os.getenv("EMBEDDING_END_POINT", None)
MILVUS_URL = os.getenv("MILVUS_URL", None)
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", None)
MILVUS_PW = os.getenv("MILVUS_PW", None)