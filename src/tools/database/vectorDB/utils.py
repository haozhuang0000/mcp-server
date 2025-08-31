from typing import List, Dict, Optional
import re
import httpx
from src.config import EMB_URL

async def a_embed_query(texts: str) -> Optional[List[List[float]]]:
    data = {'input': texts, 'type': 'query'}
    timeout = httpx.Timeout(600, connect=10)
    async with httpx.AsyncClient(verify=False, timeout=timeout) as client:
        try:
            print("Sending request")
            response = await client.post(EMB_URL, json=data)
            print("Received response")
            return response.json()
        except Exception as e:
            print(str(e))
            traceback.print_exc()
            return None