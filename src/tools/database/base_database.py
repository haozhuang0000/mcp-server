from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorDatabase(ABC):
    """Abstract base class for vector database operations"""

    @abstractmethod
    def initialize_collection(self, collection_name: str = None) -> None:
        """Initialize or create a collection"""
        pass

    @abstractmethod
    def store_data(self, data: List[Dict[str, Any]]) -> int:
        """Store data in the database"""
        pass

    @abstractmethod
    def search(self,
              query_embedding: List[float] = None,
              query_text: str = None,
              top_k: int = 10,
              filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar items"""
        pass

    @abstractmethod
    def query_data(self,
                  filter_expr: str = None,
                  output_fields: List[str] = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """Query data without vector search"""
        pass

    @abstractmethod
    def get_unique_values(self, field_name: str) -> List[str]:
        """Get unique values for a field"""
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str = None) -> None:
        """Delete a collection"""
        pass

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        pass

class TabularDatabase(ABC):
    """Abstract base class for tabular database operations"""

    @abstractmethod
    def connect(self) -> None:
        """Establish database connection"""
        pass

    @abstractmethod
    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """Execute a SQL query"""
        pass

    @abstractmethod
    def insert_data(self, table_name: str, data: List[Dict[str, Any]]) -> int:
        """Insert data into a table"""
        pass

    @abstractmethod
    def update_data(self, table_name: str, data: Dict[str, Any], where_clause: str) -> int:
        """Update data in a table"""
        pass

    @abstractmethod
    def delete_data(self, table_name: str, where_clause: str) -> int:
        """Delete data from a table"""
        pass

    @abstractmethod
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get table schema information"""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close database connection"""
        pass

class DatabaseFactory:
    """Factory for creating database instances"""

    @staticmethod
    def create_vector_db(db_type: str, **kwargs) -> VectorDatabase:
        """Create a vector database instance"""
        if db_type.lower() == "milvus":
            from .vectorDB.generalized_milvus_handler import GeneralizedMilvusHandler
            return GeneralizedMilvusHandler(**kwargs)
        else:
            raise ValueError(f"Unsupported vector database type: {db_type}")

    @staticmethod
    def create_tabular_db(db_type: str, **kwargs) -> TabularDatabase:
        """Create a tabular database instance"""
        if db_type.lower() == "mysql":
            from .tabularDB.generalized_mysql_handler import GeneralizedMySQLHandler
            return GeneralizedMySQLHandler(**kwargs)
        else:
            raise ValueError(f"Unsupported tabular database type: {db_type}")