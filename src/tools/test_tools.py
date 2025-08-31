from src.server import mcp
import random

@mcp.tool()
def hello_world_tools(name) -> str:
    """
    this is the hello user tool, when a user give you his / her name you should return this
    """
    return f'Hello {name}! Please extend this msg...'

@mcp.tool()
def roll_dice(n_dice: int) -> list[int]:
    """Roll `n_dice` 6-sided dice and return the results."""
    return [random.randint(100, 600) for _ in range(n_dice)]