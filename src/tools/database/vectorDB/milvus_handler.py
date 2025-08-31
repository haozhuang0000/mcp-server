#!/usr/bin/env python3
"""
Milvus Database Client

Handles storage and retrieval of document chunks with embeddings.
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import logging
from datetime import datetime
from pymilvus import MilvusClient, DataType, Function, FunctionType
from pymilvus import AnnSearchRequest
from pymilvus import RRFRanker

class MilvusHandler:
    """Milvus vector database client for annual report content"""

    def __init__(self, host: str, port: str = "19530", db_name: str = "default", password: str = None):
        self.host = host
        self.port = port
        self.db_name = db_name
        self.password = password
        self.collection_name = "annual_report_0821"  # Changed from documents_chunks
        self.collection = None

        self.client = MilvusClient(uri=self.host, token=f"root:{self.password}", db_name=self.db_name)
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _initialize_collection(self):
        """Initialize or create the annual report collection"""
        try:
            # Check if collection exists
            if self.client.has_collection(self.collection_name):
                # self.collection = Collection(self.collection_name)
                self.logger.info(f"📚 Using existing collection: {self.collection_name}")
            else:
                # Create new collection
                self._create_collection()

            # Load collection into memory
            # self.collection.load()
            self.logger.info(f"🚀 Collection loaded into memory")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize collection: {e}")
            raise

    def _create_collection(self):
        """Create the annual report collection"""
        schema = self.client.create_schema()

        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="session_name", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="company", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="year", datatype=DataType.VARCHAR, max_length=10)
        # schema.add_field(field_name="item_type", datatype=DataType.VARCHAR, max_length=50)
        # schema.add_field(field_name="item_title", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(field_name="chunk_text", datatype=DataType.VARCHAR, max_length=10000, enable_analyzer=True) ## Chinese character required more bytes to store
        schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
        schema.add_field(field_name="chunk_length", datatype=DataType.INT64)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=4096)  # Updated to 4096
        # schema.add_field(field_name="metadata", datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name="created_at", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="sparse_embedding", datatype=DataType.SPARSE_FLOAT_VECTOR)

        bm25_function = Function(
            name="text_bm25_emb",  # Function name
            input_field_names=["chunk_text"],  # Name of the VARCHAR field containing raw text data
            output_field_names=["sparse_embedding"],
            # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
            function_type=FunctionType.BM25,  # Set to `BM25`
        )
        schema.add_function(bm25_function)

        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="embedding",
            index_name="text_dense_index",
            index_type="IVF_FLAT",
            metric_type="COSINE"
        )

        index_params.add_index(
            field_name="sparse_embedding",
            index_name="text_sparse_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def store_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Store document chunks in Milvus with enhanced metadata"""

        if not chunks:
            self.logger.warning("No chunks to store")
            return 0

        try:
            try:
                self.client.insert(collection_name=self.collection_name, data=chunks)
            except Exception as e:
                print(chunks)
                print(chunks)
            inserted_count = len(chunks)
            self.logger.info(f"✅ Inserted {inserted_count} chunks into Milvus")
            return inserted_count

        except Exception as e:
            self.logger.error(f"❌ Failed to insert chunks: {e}")
            raise

    def hybrid_search_similar_chunks(self,
                                     query_embedding: List[float],
                                     query_text: str,
                                     top_k: int = 10,
                                     filter_expr: str = None,
                                     company: str = None,
                                     year: str = None,
                                     item_types: List[str] = None) -> List[Dict[str, Any]]:

        ranker = RRFRanker(100)
        try:
            # Build filter expression
            filter_conditions = []

            if filter_expr:
                filter_conditions.append(filter_expr)

            if company:
                filter_conditions.append(f'company == "{company}"')

            if year:
                filter_conditions.append(f'year == "{year}"')

            if item_types:
                item_type_filter = " or ".join([f'item_type == "{item_type}"' for item_type in item_types])
                filter_conditions.append(f"({item_type_filter})")

            # Combine filters
            final_filter = " and ".join(filter_conditions) if filter_conditions else None

            # Define search parameters
            search_params = {
                # "metric_type": "L2",
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }

            search_param_1 = {
                "data": [query_embedding],
                "anns_field": "embedding",
                "param": {"nprobe": 10},
                "limit": top_k,
                "expr": final_filter,
            }
            request_1 = AnnSearchRequest(**search_param_1)
            search_param_2 = {
                "data": [query_text],
                "anns_field": "sparse_embedding",
                "param": {"drop_ratio_search": 0.2},
                "limit": top_k,
                "expr": final_filter,
            }
            request_2 = AnnSearchRequest(**search_param_2)

            reqs = [request_1, request_2]

            results = self.client.hybrid_search(
                collection_name=self.collection_name,
                # filter=final_filter,
                reqs=reqs,
                ranker=ranker,
                limit=top_k,
                output_fields=["item_name", "company", "year", "item_type", "item_title",
                                "chunk_text", "chunk_index", "metadata"]
            )

            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        "chunk_id": hit.id,
                        "item_name": hit.entity.get("item_name"),
                        "company": hit.entity.get("company"),
                        "year": hit.entity.get("year"),
                        "item_type": hit.entity.get("item_type"),
                        "item_title": hit.entity.get("item_title"),
                        "chunk_text": hit.entity.get("chunk_text"),
                        "chunk_index": hit.entity.get("chunk_index"),
                        "metadata": json.loads(hit.entity.get("metadata", "{}")),
                        "similarity_score": float(hit.score)
                    }
                    formatted_results.append(result)

            self.logger.info(f"🔍 Found {len(formatted_results)} similar chunks")
            if final_filter:
                self.logger.info(f"   🎯 Filter applied: {final_filter}")

            return formatted_results

        except Exception as e:
            self.logger.error(f"❌ Search failed: {e}")
            raise