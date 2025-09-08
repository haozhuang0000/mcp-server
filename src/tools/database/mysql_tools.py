from src.server import mcp
from src.tools.database.tabularDB import MySQLHandler
from src.config import (
    MYSQL_HOST,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DATABASE,
    MYSQL_PORT,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate

from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
from src.llm import llm
from dotenv import load_dotenv
load_dotenv()

@mcp.tool()
async def extract_from_mysql(query: str):
    """
    Extract tabular data from mysql database, the user should provide a query
    """

    ## Initialized class
    mysql_handler = MySQLHandler(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        port=MYSQL_PORT
    )

    class Fields(BaseModel):

        query_to_get_fields_name: str = Field(description="The SQL query to get fields name of the table")
        table_name: str = Field(description="The name of the table")

    class FilterField(BaseModel):
        conditions: str = Field(description="The conditions for the WHERE clause")

    output_parser_agent1 = JsonOutputParser(pydantic_object=Fields)
    output_parser = JsonOutputParser(pydantic_object=FilterField)

    table_description = str({
        'cri_cri_prod_marcus_daily_complete_latest': "CRI Probability Default Data"
    })
    ## First step to get the table name and column names
    Agent1_PROMPT = PromptTemplate(template="""
    You are an expert in SQL. 
    
    Given a user's request:
    1. Identify the relevant table in the database.
    2. Generate a SQL query to List all columns name in the table.

    Ensure the query is syntactically correct and only includes SELECT statements.
    
    <USER_QUERY>
    User's request:
    {query}
    </USER_QUERY>
    
    <TABLE_INFO>
    Your are provided with the following table descriptions, key is the table name and value is the description:
    {table_description}
    Please note that the provided table name is the actual table name, you should not make any changes.
    </TABLE_INFO>
    
    <FORMAT>
    Format the output according to these instructions:
    {format_instructions}
    </FORMAT>
    """,
    input_variables=['query', 'table_description','format_instructions']
    )

    chain = Agent1_PROMPT | llm | output_parser_agent1
    results = chain.invoke({
        'query': query,
        'table_description': table_description,
        'format_instructions': output_parser_agent1.get_format_instructions()
    })
    table_name, query_to_get_fields_name = results['table_name'], results['query_to_get_fields_name']
    print(f"Identified Table: {table_name}, Columns: {query_to_get_fields_name}")

    column_names = list(mysql_handler.fetch_df(query=query_to_get_fields_name)['COLUMN_NAME'])

    ## Second step to get the conditions
    PROMPT = PromptTemplate(template="""
    You are an expert in SQL. Given a user's request, generate a SQL query to extract the required data from a MySQL database.
    Ensure the query is syntactically correct and only includes SELECT statements.

    User's request:
    {query}

    You are provided with the following information:
    1. The table name: {table_name}
    2. The column names: {column_names}

    Format the output according to these instructions:
    {format_instructions}
    """,
    input_variables=['query', 'table_name', 'column_names', 'format_instructions']
    )

    chain = PROMPT | llm | output_parser
    results = chain.invoke({
        'query': query,
        'format_instructions': output_parser.get_format_instructions(),
        'table_name': table_name,
        'column_names': ', '.join(column_names)
    })

    conditions = results['conditions']
    
    sql_query = f"SELECT {', '.join(column_names)} FROM {table_name} WHERE {conditions};"
    print(f"Generated SQL Query: {sql_query}")
    db_results = mysql_handler.fetch_df(sql_query)
    return db_results

