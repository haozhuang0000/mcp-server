#!/usr/bin/env python3
"""
Generalized MySQL Database Client

Handles CRUD operations for tabular data with configurable connections and schemas.
"""

import pymysql
import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from ..base_database import TabularDatabase

class GeneralizedMySQLHandler(TabularDatabase):
    """Generalized MySQL database client for various table operations"""

    def __init__(self,
                 host: str,
                 user: str,
                 password: str,
                 database: str,
                 port: int = 3306,
                 charset: str = 'utf8mb4'):
        """
        Initialize MySQL handler

        Args:
            host: MySQL host
            user: Database user
            password: Database password
            database: Database name
            port: MySQL port
            charset: Character set
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.charset = charset
        self.connection = None

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def connect(self) -> None:
        """Establish database connection"""
        try:
            self.connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port,
                charset=self.charset,
                cursorclass=pymysql.cursors.DictCursor
            )
            self.logger.info(f"✅ Connected to MySQL database: {self.database}")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to MySQL: {e}")
            raise

    @contextmanager
    def get_cursor(self):
        """Context manager for database cursor"""
        if not self.connection:
            self.connect()

        try:
            with self.connection.cursor() as cursor:
                yield cursor
                self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            self.logger.error(f"❌ Database operation failed: {e}")
            raise

    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, params or [])
                results = cursor.fetchall()
                self.logger.info(f"📊 Query executed successfully, {len(results)} rows returned")
                return results
        except Exception as e:
            self.logger.error(f"❌ Query execution failed: {e}")
            raise

    def insert_data(self, table_name: str, data: List[Dict[str, Any]]) -> int:
        """Insert data into a table"""
        if not data:
            self.logger.warning("No data to insert")
            return 0

        try:
            with self.get_cursor() as cursor:
                # Get column names from the first record
                columns = list(data[0].keys())
                placeholders = ', '.join(['%s'] * len(columns))
                columns_str = ', '.join(f"`{col}`" for col in columns)

                insert_query = f"INSERT INTO `{table_name}` ({columns_str}) VALUES ({placeholders})"

                # Prepare data for insertion
                values_list = []
                for record in data:
                    values = [record.get(col) for col in columns]
                    values_list.append(values)

                # Execute batch insert
                cursor.executemany(insert_query, values_list)
                inserted_count = cursor.rowcount

                self.logger.info(f"✅ Inserted {inserted_count} records into {table_name}")
                return inserted_count

        except Exception as e:
            self.logger.error(f"❌ Failed to insert data into {table_name}: {e}")
            raise

    def update_data(self, table_name: str, data: Dict[str, Any], where_clause: str) -> int:
        """Update data in a table"""
        try:
            with self.get_cursor() as cursor:
                # Build SET clause
                set_clauses = []
                values = []
                for key, value in data.items():
                    set_clauses.append(f"`{key}` = %s")
                    values.append(value)

                set_clause = ', '.join(set_clauses)
                update_query = f"UPDATE `{table_name}` SET {set_clause} WHERE {where_clause}"

                cursor.execute(update_query, values)
                updated_count = cursor.rowcount

                self.logger.info(f"✅ Updated {updated_count} records in {table_name}")
                return updated_count

        except Exception as e:
            self.logger.error(f"❌ Failed to update data in {table_name}: {e}")
            raise

    def delete_data(self, table_name: str, where_clause: str) -> int:
        """Delete data from a table"""
        try:
            with self.get_cursor() as cursor:
                delete_query = f"DELETE FROM `{table_name}` WHERE {where_clause}"
                cursor.execute(delete_query)
                deleted_count = cursor.rowcount

                self.logger.info(f"✅ Deleted {deleted_count} records from {table_name}")
                return deleted_count

        except Exception as e:
            self.logger.error(f"❌ Failed to delete data from {table_name}: {e}")
            raise

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get table schema information"""
        try:
            with self.get_cursor() as cursor:
                # Get column information
                cursor.execute(f"DESCRIBE `{table_name}`")
                columns = cursor.fetchall()

                # Get table statistics
                cursor.execute(f"SELECT COUNT(*) as row_count FROM `{table_name}`")
                stats = cursor.fetchone()

                schema_info = {
                    "table_name": table_name,
                    "columns": columns,
                    "row_count": stats["row_count"] if stats else 0
                }

                self.logger.info(f"📋 Retrieved schema for table {table_name}")
                return schema_info

        except Exception as e:
            self.logger.error(f"❌ Failed to get schema for {table_name}: {e}")
            raise

    def list_tables(self) -> List[str]:
        """List all tables in the database"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                table_names = [list(table.values())[0] for table in tables]

                self.logger.info(f"📋 Found {len(table_names)} tables in database")
                return table_names

        except Exception as e:
            self.logger.error(f"❌ Failed to list tables: {e}")
            raise

    def create_table(self, table_name: str, schema: Dict[str, str]) -> None:
        """Create a new table with the given schema"""
        try:
            with self.get_cursor() as cursor:
                # Build CREATE TABLE statement
                column_definitions = []
                for column_name, column_type in schema.items():
                    column_definitions.append(f"`{column_name}` {column_type}")

                columns_str = ', '.join(column_definitions)
                create_query = f"CREATE TABLE `{table_name}` ({columns_str})"

                cursor.execute(create_query)
                self.logger.info(f"✅ Created table {table_name}")

        except Exception as e:
            self.logger.error(f"❌ Failed to create table {table_name}: {e}")
            raise

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists"""
        try:
            tables = self.list_tables()
            return table_name in tables
        except Exception as e:
            self.logger.error(f"❌ Failed to check if table {table_name} exists: {e}")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self.get_cursor() as cursor:
                # Get database size
                cursor.execute("""
                    SELECT
                        ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) AS size_mb
                    FROM information_schema.tables
                    WHERE table_schema = %s
                """, [self.database])

                size_info = cursor.fetchone()
                tables = self.list_tables()

                stats = {
                    "database_name": self.database,
                    "table_count": len(tables),
                    "size_mb": size_info["size_mb"] if size_info else 0,
                    "tables": tables
                }

                self.logger.info(f"📊 Retrieved database statistics")
                return stats

        except Exception as e:
            self.logger.error(f"❌ Failed to get database stats: {e}")
            raise

    def close(self) -> None:
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.logger.info("🔒 MySQL connection closed")

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()