from src.server import mcp
from src.tools.database.vectorDB import a_embed_query
from src.config import VECTOR_DB_CONFIG, MILVUS_CONFIG, MYSQL_CONFIG
from src.llm import llm
from src.tools.database.base_database import DatabaseFactory

from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
load_dotenv()

class FilterField(BaseModel):
    """Dynamic filter field model"""
    filters: Dict[str, str] = Field(description="Key-value pairs for filtering data")

@mcp.tool()
async def search_vector_database(query: str,
                                collection_name: str = None,
                                schema_type: str = "document",
                                top_k: int = 10,
                                filters: Dict[str, str] = None):
    """
    Search a vector database with configurable schema and filters

    Args:
        query: Search query text
        collection_name: Name of the collection to search (optional, uses default if not provided)
        schema_type: Type of schema ('document', 'annual_report', or 'custom')
        top_k: Number of results to return
        filters: Dictionary of field-value pairs for filtering
    """
    try:
        # Use default collection if not provided
        if not collection_name:
            collection_name = VECTOR_DB_CONFIG.default_collection

        # Create vector database instance
        vector_db = DatabaseFactory.create_vector_db(
            "milvus",
            host=MILVUS_CONFIG.url,
            db_name=MILVUS_CONFIG.db_name,
            password=MILVUS_CONFIG.password,
            collection_name=collection_name,
            embedding_dim=VECTOR_DB_CONFIG.embedding_dim,
            schema_type=schema_type
        )

        # Initialize collection
        vector_db.initialize_collection()

        # Get embedding for the query
        embed_result = await a_embed_query(query)

        # Perform search
        results = vector_db.search(
            query_embedding=embed_result['vector'],
            query_text=embed_result['text'],
            top_k=top_k,
            filters=filters or {}
        )

        return {
            "query": query,
            "collection": collection_name,
            "schema_type": schema_type,
            "total_results": len(results),
            "results": results
        }

    except Exception as e:
        return {
            "error": f"Search failed: {str(e)}",
            "query": query,
            "collection": collection_name
        }

@mcp.tool()
async def smart_search_with_filter_extraction(query: str,
                                             collection_name: str = None,
                                             schema_type: str = "document",
                                             top_k: int = 10):
    """
    Intelligent search that automatically extracts filters from the query

    Args:
        query: Natural language search query
        collection_name: Name of the collection to search
        schema_type: Type of schema ('document', 'annual_report', or 'custom')
        top_k: Number of results to return
    """
    try:
        # Use default collection if not provided
        if not collection_name:
            collection_name = VECTOR_DB_CONFIG.default_collection

        # Create vector database instance
        vector_db = DatabaseFactory.create_vector_db(
            "milvus",
            host=MILVUS_CONFIG.url,
            db_name=MILVUS_CONFIG.db_name,
            password=MILVUS_CONFIG.password,
            collection_name=collection_name,
            embedding_dim=VECTOR_DB_CONFIG.embedding_dim,
            schema_type=schema_type
        )

        # Initialize collection
        vector_db.initialize_collection()

        # Get available filter fields based on schema type
        if schema_type == "annual_report":
            available_fields = vector_db.get_unique_values("company")
            filter_prompt = f"""
            Extract filtering criteria from the user's query. Available companies: {', '.join(available_fields)}
            Look for company names and years in the query.
            """
        else:
            filter_prompt = """
            Extract any filtering criteria from the user's query such as source, date, category, etc.
            """

        # Use LLM to extract filters
        output_parser = JsonOutputParser(pydantic_object=FilterField)

        PROMPT = PromptTemplate(template=f"""
        {filter_prompt}

        <USER_QUERY>
        {{query}}
        </USER_QUERY>

        Format the output according to these instructions:
        {{format_instructions}}
        """,
        input_variables=['query', 'format_instructions']
        )

        chain = PROMPT | llm | output_parser
        extracted_filters = chain.invoke({
            'query': query,
            'format_instructions': output_parser.get_format_instructions()
        })

        # Get embedding for the query
        embed_result = await a_embed_query(query)

        # Perform search with extracted filters
        results = vector_db.search(
            query_embedding=embed_result['vector'],
            query_text=embed_result['text'],
            top_k=top_k,
            filters=extracted_filters.get('filters', {})
        )

        return {
            "query": query,
            "collection": collection_name,
            "schema_type": schema_type,
            "extracted_filters": extracted_filters.get('filters', {}),
            "total_results": len(results),
            "results": results
        }

    except Exception as e:
        return {
            "error": f"Smart search failed: {str(e)}",
            "query": query,
            "collection": collection_name
        }

@mcp.tool()
async def query_tabular_database(sql_query: str, table_name: str = None):
    """
    Execute SQL queries on the configured tabular database

    Args:
        sql_query: SQL query to execute
        table_name: Optional table name for validation
    """
    try:
        # Create tabular database instance
        tabular_db = DatabaseFactory.create_tabular_db(
            "mysql",
            host=MYSQL_CONFIG.host,
            user=MYSQL_CONFIG.user,
            password=MYSQL_CONFIG.password,
            database=MYSQL_CONFIG.database,
            port=MYSQL_CONFIG.port
        )

        # Execute query
        results = tabular_db.execute_query(sql_query)

        # Close connection
        tabular_db.close()

        return {
            "query": sql_query,
            "total_results": len(results),
            "results": results
        }

    except Exception as e:
        return {
            "error": f"SQL query failed: {str(e)}",
            "query": sql_query
        }

@mcp.tool()
async def get_database_info(database_type: str = "vector", collection_name: str = None):
    """
    Get information about the configured databases

    Args:
        database_type: Type of database ('vector' or 'tabular')
        collection_name: Collection name for vector database
    """
    try:
        if database_type == "vector":
            if not collection_name:
                collection_name = VECTOR_DB_CONFIG.default_collection

            vector_db = DatabaseFactory.create_vector_db(
                "milvus",
                host=MILVUS_CONFIG.url,
                db_name=MILVUS_CONFIG.db_name,
                password=MILVUS_CONFIG.password,
                collection_name=collection_name,
                embedding_dim=VECTOR_DB_CONFIG.embedding_dim,
                schema_type="document"
            )

            stats = vector_db.get_collection_stats()
            return {
                "database_type": "vector",
                "collection_info": stats,
                "config": {
                    "embedding_dim": VECTOR_DB_CONFIG.embedding_dim,
                    "default_collection": VECTOR_DB_CONFIG.default_collection
                }
            }

        elif database_type == "tabular":
            tabular_db = DatabaseFactory.create_tabular_db(
                "mysql",
                host=MYSQL_CONFIG.host,
                user=MYSQL_CONFIG.user,
                password=MYSQL_CONFIG.password,
                database=MYSQL_CONFIG.database,
                port=MYSQL_CONFIG.port
            )

            stats = tabular_db.get_database_stats()
            tabular_db.close()

            return {
                "database_type": "tabular",
                "database_info": stats
            }

        else:
            return {"error": f"Unknown database type: {database_type}"}

    except Exception as e:
        return {
            "error": f"Failed to get database info: {str(e)}",
            "database_type": database_type
        }