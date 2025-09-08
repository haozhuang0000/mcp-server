from src.server import mcp
from src.tools.database.vectorDB import MilvusHandler
from src.tools.database.vectorDB import a_embed_query
from src.config import MILVUS_URL, MILVUS_DB_NAME, MILVUS_PW
from src.llm import llm

from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate

from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

class FilterField(BaseModel):
    year: str
    company: str

@mcp.tool()
async def extract_from_milvus(query: str):
    """
    Extract text from milvus vector database, the user should provide a query
    """

    ## Initialized class
    milvus_handler = MilvusHandler(host=MILVUS_URL, db_name=MILVUS_DB_NAME, password=MILVUS_PW)
    embed_query = await a_embed_query(query)
    unique_company_name = milvus_handler.extract_unique_company_name()
    output_parser = JsonOutputParser(pydantic_object=FilterField)

    PROMPT = PromptTemplate(template="""
    Please extract from the user's query if the query explicit the company and year
    
    The company name must be select from the list provided.
    
    <USER_QUERY>
    {query}
    </USER_QUERY>
    
    <PROVIDED_COMPANY_NAME>
    {company_name}
    </PROVIDED_COMPANY_NAME>
    
    Format the output according to these instructions:
    {format_instructions}
    """,
    input_variables=['query', 'company_name','format_instructions']
    )

    chain = PROMPT | llm | output_parser
    results = chain.invoke({
        'query': query,
        'company_name': ', '.join(unique_company_name),
        'format_instructions': output_parser.get_format_instructions()
    })

    year, company = results['year'], results['company']
    print(year, company)
    vdb_results = milvus_handler.hybrid_search_similar_chunks(
        query_embedding=embed_query['vector'],
        query_text=embed_query['text'],
        year=year,
        company=company
    )
    return vdb_results

