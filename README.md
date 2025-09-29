# MCP Server with LangChain and Milvus

A Model Context Protocol (MCP) server implementation that integrates LangChain and Milvus vector database for intelligent agent workflows and semantic search capabilities.

## Features

- **MCP Integration**: Full Model Context Protocol support for seamless AI agent communication
- **LangChain Compatibility**: Built-in support for LangChain tools and LangGraph agents
- **Flexible Vector Database**: Configurable schemas for different document types and use cases
- **Multiple Database Support**: Abstract interfaces for vector (Milvus) and tabular (MySQL) databases
- **Dynamic Schema Creation**: Support for custom collection schemas with configurable fields
- **Intelligent Search**: Automatic filter extraction from natural language queries
- **RESTful API**: HTTP-based MCP server with streamable client support

## Prerequisites

- Python 3.12+
- Conda (recommended for environment management)

## Installation

### 1. Install uv
Linux / Mac
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Windows
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install dependencies
```bash
uv sync
```

## Quick Start

Here's a basic example of how to use the MCP server with LangChain:

```
cd src
python main.py
```

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.tools import load_mcp_tools

async with streamablehttp_client("http://MCP_SERVER_ADDRESS:PORT/mcp/") as (read, write, _):
    async with ClientSession(read, write) as session:
        # Initialize the connection
        await session.initialize()
        
        # Load available tools from MCP server
        tools = await load_mcp_tools(session)
        
        # Create a reactive agent with the tools
        agent = create_react_agent("openai:gpt-4.1", tools)
        
        # Execute a query
        response2 = await agent.ainvoke({
            "messages": """please help find the total revenue for company: Singapore Airlines in year: 2024, you should search from vector database, the collection name is annual_report_0821"""
        })
        
        print(response2['messages'][-1].content)
```

## Configuration

### Environment Variables
Create a `.env` file in your project root:

```env
# MCP Server Configuration
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=8000

# Vector Database Configuration
EMBEDDING_END_POINT=<FASTAPI_HOSTED_EMBEDDING_MODEL>
EMBEDDING_DIM=4096
DEFAULT_COLLECTION=documents

# Milvus Configuration
MILVUS_URL=localhost
MILVUS_DB_NAME=default
MILVUS_PW=your_milvus_password

# MySQL Configuration
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=your_database_name

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
```

### Schema Types

The generalized system supports multiple schema types:

1. **Document Schema** (default): General-purpose schema for various document types
2. **Annual Report Schema**: Specialized for financial reports with company/year filtering
3. **Custom Schema**: User-defined schemas for specific use cases

### Using Different Schemas

```python
# Document schema (default)
vector_db = DatabaseFactory.create_vector_db(
    "milvus",
    collection_name="my_documents",
    schema_type="document"
)

# Annual reports schema
vector_db = DatabaseFactory.create_vector_db(
    "milvus",
    collection_name="annual_reports",
    schema_type="annual_report"
)

# Custom schema
custom_schema = SchemaFactory.create_custom_schema(
    "custom_collection",
    fields=[...],  # Your custom fields
    embedding_dim=4096
)
vector_db.set_custom_schema(custom_schema)
```

## Acknowledgments

- [Model Context Protocol](https://github.com/modelcontextprotocol/python-sdk) for the foundational protocol
- [LangChain](https://github.com/langchain-ai/langchain-mcp-adapters) for AI agent frameworks & MCP Adapter
- [Milvus](https://github.com/milvus-io/milvus) for vector database capabilities