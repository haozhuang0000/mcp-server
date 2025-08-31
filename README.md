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

### 1. Create and activate environment
```bash
conda create -n mcp python=3.12
conda activate mcp
```

### 2. Install dependencies
```bash
pip install mcp langgraph langchain_mcp_adapters langchain pydantic python-dotenv starlette pymilvus
```

## Quick Start

Here's a basic example of how to use the MCP server with LangChain:

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
```

## Acknowledgments

- [Model Context Protocol](https://github.com/modelcontextprotocol/python-sdk) for the foundational protocol
- [LangChain](https://github.com/langchain-ai/langchain-mcp-adapters) for AI agent frameworks & MCP Adapter
- [Milvus](https://github.com/milvus-io/milvus) for vector database capabilities