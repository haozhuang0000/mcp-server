from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pymilvus import DataType, Function, FunctionType

@dataclass
class FieldConfig:
    """Configuration for a single field in the collection"""
    name: str
    datatype: DataType
    max_length: Optional[int] = None
    is_primary: bool = False
    auto_id: bool = False
    dim: Optional[int] = None
    enable_analyzer: bool = False

@dataclass
class IndexConfig:
    """Configuration for an index"""
    field_name: str
    index_name: str
    index_type: str
    metric_type: str
    params: Optional[Dict[str, Any]] = None

@dataclass
class CollectionConfig:
    """Complete configuration for a collection"""
    name: str
    fields: List[FieldConfig]
    indexes: List[IndexConfig]
    functions: Optional[List[Dict[str, Any]]] = None

class SchemaFactory:
    """Factory for creating different schema configurations"""

    @staticmethod
    def create_document_schema(collection_name: str, embedding_dim: int = 4096) -> CollectionConfig:
        """Create a general document schema suitable for various document types"""
        fields = [
            FieldConfig("doc_id", DataType.INT64, is_primary=True, auto_id=True),
            FieldConfig("content", DataType.VARCHAR, max_length=10000, enable_analyzer=True),
            FieldConfig("metadata", DataType.VARCHAR, max_length=2000),
            FieldConfig("source", DataType.VARCHAR, max_length=500),
            FieldConfig("created_at", DataType.VARCHAR, max_length=50),
            FieldConfig("embedding", DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldConfig("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
        ]

        indexes = [
            IndexConfig("embedding", "dense_index", "IVF_FLAT", "COSINE"),
            IndexConfig("sparse_embedding", "sparse_index", "SPARSE_INVERTED_INDEX", "BM25",
                       {"inverted_index_algo": "DAAT_MAXSCORE"})
        ]

        functions = [{
            "name": "text_bm25_emb",
            "input_field_names": ["content"],
            "output_field_names": ["sparse_embedding"],
            "function_type": FunctionType.BM25
        }]

        return CollectionConfig(collection_name, fields, indexes, functions)

    @staticmethod
    def create_annual_report_schema(collection_name: str, embedding_dim: int = 4096) -> CollectionConfig:
        """Create schema for annual reports (backward compatibility)"""
        fields = [
            FieldConfig("chunk_id", DataType.INT64, is_primary=True, auto_id=True),
            FieldConfig("session_name", DataType.VARCHAR, max_length=100),
            FieldConfig("company", DataType.VARCHAR, max_length=100),
            FieldConfig("year", DataType.VARCHAR, max_length=10),
            FieldConfig("chunk_text", DataType.VARCHAR, max_length=10000, enable_analyzer=True),
            FieldConfig("chunk_index", DataType.INT64),
            FieldConfig("chunk_length", DataType.INT64),
            FieldConfig("embedding", DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldConfig("created_at", DataType.VARCHAR, max_length=50),
            FieldConfig("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
        ]

        indexes = [
            IndexConfig("embedding", "text_dense_index", "IVF_FLAT", "COSINE"),
            IndexConfig("sparse_embedding", "text_sparse_index", "SPARSE_INVERTED_INDEX", "BM25",
                       {"inverted_index_algo": "DAAT_MAXSCORE"})
        ]

        functions = [{
            "name": "text_bm25_emb",
            "input_field_names": ["chunk_text"],
            "output_field_names": ["sparse_embedding"],
            "function_type": FunctionType.BM25
        }]

        return CollectionConfig(collection_name, fields, indexes, functions)

    @staticmethod
    def create_custom_schema(collection_name: str,
                           fields: List[Dict[str, Any]],
                           embedding_dim: int = 4096) -> CollectionConfig:
        """Create a custom schema from field definitions"""
        field_configs = []
        for field in fields:
            field_configs.append(FieldConfig(**field))

        # Default indexes for embedding fields
        indexes = []
        for field in field_configs:
            if field.datatype == DataType.FLOAT_VECTOR:
                indexes.append(IndexConfig(field.name, f"{field.name}_index", "IVF_FLAT", "COSINE"))
            elif field.datatype == DataType.SPARSE_FLOAT_VECTOR:
                indexes.append(IndexConfig(field.name, f"{field.name}_index", "SPARSE_INVERTED_INDEX", "BM25",
                                         {"inverted_index_algo": "DAAT_MAXSCORE"}))

        return CollectionConfig(collection_name, field_configs, indexes)