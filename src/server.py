from mcp.server.fastmcp import FastMCP

from starlette.requests import Request
from starlette.responses import PlainTextResponse
# This is the shared MCP server instance

# server_instructions = """
# This MCP server provides search and document retrieval capabilities
# for deep research. Use the search tool to find relevant documents
# based on keywords, then use the fetch tool to retrieve complete
# document content with citations.
# """
# # mcp = FastMCP("aidf_mcp_server",instructions=server_instructions)

mcp = FastMCP("local mcp server", host="0.0.0.0", port=9292)

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")