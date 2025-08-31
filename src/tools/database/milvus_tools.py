from src.server import mcp
from src.tools.database.vectorDB import MilvusHandler
from src.tools.database.vectorDB import a_embed_query
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate

from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(model='gpt-4o')

class FilterField(BaseModel):
    year: str
    company: str

@mcp.tool()
async def extract_from_milvus(query: str):
    """
    Extract text from milvus vector database, the user should provide a query
    """

    ## Initialized class
    milvus_handler = MilvusHandler()
    embed_query = await a_embed_query(query)

    output_parser = JsonOutputParser(pydantic_object=FilterField)
    PROMPT = PromptTemplate(template="""
    Please extract from the user's query if the query explicit the company and year
    <USER_QUERY>
    {query}
    </USER_QUERY>
    """, input_variables=['query'], partial_variables=output_parser.get_format_instructions())
    chain = PROMPT | llm | output_parser
    results = chain.invoke({'query': query})
    year, company = results['year'], results['company']
    vdb_results = milvus_handler.hybrid_search_similar_chunks(
        query_embedding=embed_query['vector'],
        query_text=embed_query['text'],
        year=year,
        company=company
    )
    return vdb_results

