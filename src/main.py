import logging
from src.server import mcp
import src.tools.test_tools
import src.tools.database.milvus_tools
import src.tools.database.mysql_tools
import src.tools.database.generalized_tools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp-server")

if __name__ == "__main__":
    logger.info("Starting MCP server on port 9292...")
    # Close the HTTP client when the server shuts down
    mcp.run(transport="streamable-http")