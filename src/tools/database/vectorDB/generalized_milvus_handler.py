#!/usr/bin/env python3
"""
Generalized Milvus Database Client

Handles storage and retrieval of document chunks with configurable schemas and embeddings.
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import logging
from datetime import datetime
from pymilvus import MilvusClient, DataType, Function, FunctionType
from pymilvus import AnnSearchRequest
from pymilvus import RRFRanker

from .schema_config import CollectionConfig, SchemaFactory
from ..base_database import VectorDatabase

class GeneralizedMilvusHandler(VectorDatabase):
    """Generalized Milvus vector database client for various document types"""

    def __init__(self,
                 host: str,
                 port: str = "19530",
                 db_name: str = "default",
                 password: str = None,
                 collection_name: str = "documents",
                 embedding_dim: int = 4096,
                 schema_type: str = "document"):
        """
        Initialize the Milvus handler with configurable parameters

        Args:
            host: Milvus host
            port: Milvus port
            db_name: Database name
            password: Password for authentication
            collection_name: Name of the collection to work with
            embedding_dim: Dimension of the embedding vectors
            schema_type: Type of schema to use ('document', 'annual_report', or 'custom')
        """
        self.host = host
        self.port = port
        self.db_name = db_name
        self.password = password
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.schema_type = schema_type
        self.collection = None
        self.schema_config = None

        self.client = MilvusClient(uri=self.host, token=f"root:{self.password}", db_name=self.db_name)

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize schema configuration
        self._setup_schema()

    def _setup_schema(self):
        """Setup the schema configuration based on schema_type"""
        if self.schema_type == "document":
            self.schema_config = SchemaFactory.create_document_schema(
                self.collection_name, self.embedding_dim
            )
        elif self.schema_type == "annual_report":
            self.schema_config = SchemaFactory.create_annual_report_schema(
                self.collection_name, self.embedding_dim
            )
        else:
            raise ValueError(f"Unsupported schema type: {self.schema_type}")

    def set_custom_schema(self, schema_config: CollectionConfig):
        """Set a custom schema configuration"""
        self.schema_config = schema_config
        self.collection_name = schema_config.name

    def initialize_collection(self):
        """Initialize or create the collection based on the schema configuration"""
        try:
            if self.client.has_collection(self.collection_name):
                self.logger.info(f"📚 Using existing collection: {self.collection_name}")
            else:
                self._create_collection()

            self.logger.info(f"🚀 Collection ready: {self.collection_name}")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize collection: {e}")
            raise

    def _create_collection(self):
        """Create the collection using the schema configuration"""
        if not self.schema_config:
            raise ValueError("Schema configuration not set")

        schema = self.client.create_schema()

        # Add fields
        for field_config in self.schema_config.fields:
            kwargs = {
                "field_name": field_config.name,
                "datatype": field_config.datatype,
                "is_primary": field_config.is_primary,
                "auto_id": field_config.auto_id
            }

            if field_config.max_length:
                kwargs["max_length"] = field_config.max_length
            if field_config.dim:
                kwargs["dim"] = field_config.dim
            if field_config.enable_analyzer:
                kwargs["enable_analyzer"] = field_config.enable_analyzer

            schema.add_field(**kwargs)

        # Add functions (like BM25)
        if self.schema_config.functions:
            for func_config in self.schema_config.functions:
                function = Function(
                    name=func_config["name"],
                    input_field_names=func_config["input_field_names"],
                    output_field_names=func_config["output_field_names"],
                    function_type=func_config["function_type"]
                )
                schema.add_function(function)

        # Prepare indexes
        index_params = self.client.prepare_index_params()
        for index_config in self.schema_config.indexes:
            kwargs = {
                "field_name": index_config.field_name,
                "index_name": index_config.index_name,
                "index_type": index_config.index_type,
                "metric_type": index_config.metric_type
            }
            if index_config.params:
                kwargs["params"] = index_config.params

            index_params.add_index(**kwargs)

        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

        self.logger.info(f"✅ Created collection: {self.collection_name}")

    def store_data(self, data: List[Dict[str, Any]]) -> int:
        """Store data in the collection"""
        if not data:
            self.logger.warning("No data to store")
            return 0

        try:
            self.client.insert(collection_name=self.collection_name, data=data)
            inserted_count = len(data)
            self.logger.info(f"✅ Inserted {inserted_count} records into {self.collection_name}")
            return inserted_count

        except Exception as e:
            self.logger.error(f"❌ Failed to insert data: {e}")
            raise

    def search(self,
              query_embedding: List[float] = None,
              query_text: str = None,
              top_k: int = 10,
              filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar items (implements VectorDatabase interface)
        """
        if query_embedding is None and query_text is None:
            raise ValueError("Either query_embedding or query_text must be provided")

        # Convert filters dict to filter expressions
        filter_expr = None
        additional_filters = filters or {}

        return self.hybrid_search(query_embedding, query_text, top_k, filter_expr, additional_filters)

    def hybrid_search(self,
                     query_embedding: List[float],
                     query_text: str,
                     top_k: int = 10,
                     filter_expr: str = None,
                     additional_filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with configurable filtering

        Args:
            query_embedding: Dense vector for similarity search
            query_text: Text for sparse (BM25) search
            top_k: Number of results to return
            filter_expr: Raw filter expression
            additional_filters: Key-value pairs for additional filtering
        """
        ranker = RRFRanker(100)

        try:
            # Build filter expression
            filter_conditions = []

            if filter_expr:
                filter_conditions.append(filter_expr)

            if additional_filters:
                for key, value in additional_filters.items():
                    if isinstance(value, str):
                        filter_conditions.append(f'{key} == "{value}"')
                    else:
                        filter_conditions.append(f'{key} == {value}')

            final_filter = " and ".join(filter_conditions) if filter_conditions else None

            # Determine field names for dense and sparse embeddings
            dense_field = "embedding"
            sparse_field = "sparse_embedding"
            content_field = "content" if self.schema_type == "document" else "chunk_text"

            # Configure search parameters
            search_param_1 = {
                "data": [query_embedding],
                "anns_field": dense_field,
                "param": {"nprobe": 10},
                "limit": top_k,
                "expr": final_filter,
            }
            request_1 = AnnSearchRequest(**search_param_1)

            search_param_2 = {
                "data": [query_text],
                "anns_field": sparse_field,
                "param": {"drop_ratio_search": 0.2},
                "limit": top_k,
                "expr": final_filter,
            }
            request_2 = AnnSearchRequest(**search_param_2)

            reqs = [request_1, request_2]

            # Get field names for output (excluding embedding fields)
            desc = self.client.describe_collection(self.collection_name)['fields']
            fields = [i['name'] for i in desc if 'embedding' not in i['name']]

            results = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=reqs,
                ranker=ranker,
                limit=top_k,
                output_fields=fields
            )

            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        "id": hit.id,
                        "similarity_score": float(hit.score)
                    }

                    # Add all entity fields
                    for field_name in fields:
                        value = hit.entity.get(field_name)
                        if field_name == "metadata" and isinstance(value, str):
                            try:
                                result[field_name] = json.loads(value)
                            except:
                                result[field_name] = value
                        else:
                            result[field_name] = value

                    formatted_results.append(result)

            self.logger.info(f"🔍 Found {len(formatted_results)} results")
            if final_filter:
                self.logger.info(f"   🎯 Filter applied: {final_filter}")

            return formatted_results

        except Exception as e:
            self.logger.error(f"❌ Search failed: {e}")
            raise

    def query_data(self,
                  filter_expr: str = None,
                  output_fields: List[str] = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """Query data from the collection without vector search"""
        try:
            if output_fields is None:
                # Get all non-embedding fields
                desc = self.client.describe_collection(self.collection_name)['fields']
                output_fields = [i['name'] for i in desc if 'embedding' not in i['name']]

            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr or "",
                output_fields=output_fields,
                limit=limit
            )

            self.logger.info(f"📊 Retrieved {len(results)} records")
            return results

        except Exception as e:
            self.logger.error(f"❌ Query failed: {e}")
            raise

    def get_unique_values(self, field_name: str) -> List[str]:
        """Get unique values for a specific field"""
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                filter=f"{field_name} != ''",
                output_fields=[field_name],
                limit=10000
            )

            unique_values = list(set(row[field_name] for row in results if row[field_name]))
            self.logger.info(f"📋 Found {len(unique_values)} unique values for {field_name}")
            return unique_values

        except Exception as e:
            self.logger.error(f"❌ Failed to get unique values for {field_name}: {e}")
            raise

    def delete_collection(self):
        """Delete the current collection"""
        try:
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
                self.logger.info(f"🗑️ Deleted collection: {self.collection_name}")
            else:
                self.logger.info(f"Collection {self.collection_name} does not exist")
        except Exception as e:
            self.logger.error(f"❌ Failed to delete collection: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            if not self.client.has_collection(self.collection_name):
                return {"exists": False}

            stats = self.client.get_collection_stats(self.collection_name)
            return {
                "exists": True,
                "row_count": stats.get("row_count", 0),
                "collection_name": self.collection_name,
                "schema_type": self.schema_type
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to get collection stats: {e}")
            return {"exists": False, "error": str(e)}