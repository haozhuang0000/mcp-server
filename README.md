# MCP Server with LangChain and Milvus

A Model Context Protocol (MCP) server implementation that integrates LangChain and Milvus vector database for intelligent agent workflows and semantic search capabilities.

## Features

- **MCP Integration**: Full Model Context Protocol support for seamless AI agent communication
- **LangChain Compatibility**: Built-in support for LangChain tools and LangGraph agents
- **Milvus Vector Database**: High-performance vector storage and similarity search
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
        response = await agent.ainvoke({
            "messages": "What is the headquarters of NVIDIA? Year=2024, and company=NVIDIA"
        })
        
        print(response)
```

## Configuration

### Environment Variables
Create a `.env` file in your project root:

```env
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=8000
MILVUS_HOST=localhost
MILVUS_PORT=19530
OPENAI_API_KEY=your_openai_api_key_here

MYSQL_HOST=
MYSQL_PORT=
MYSQL_USER=
MYSQL_PASSWORD=
MYSQL_DATABASE=
MYSQL_TABLE=

EMBEDDING_END_POINT=<FASTAPI_HOSTED_EMBEDDING_MODEL>
```

## Acknowledgments

- [Model Context Protocol](https://github.com/modelcontextprotocol/python-sdk) for the foundational protocol
- [LangChain](https://github.com/langchain-ai/langchain-mcp-adapters) for AI agent frameworks & MCP Adapter
- [Milvus](https://github.com/milvus-io/milvus) for vector database capabilities