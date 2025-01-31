import os

import aiohttp

from schemas.gpt_result import GPTResult, GPTSource


class YAGPTException(Exception):
    pass


async def query_gpt(question: str, site: str) -> GPTResult:
    YA_GPT_URL = os.environ.get("YA_GPT_URL")
    API_KEY = os.environ.get("API_KEY")

    payload = {
        "messages": [
            {
                "role": "user",
                "content": question,
            }
        ],
        "site": site,
    }

    headers = {
        'Authorization': f'Api-Key {API_KEY}',
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url=YA_GPT_URL, json=payload, headers=headers) as response:
            if response.status == 200:
                res = await response.json()
                sources = [GPTSource(key=key, url=source['url'], title=source['title']) for key, source in res["used_sources"].items()]
                return GPTResult(
                    source_confidence=0.5 if site == "itmo.ru" else 0.25,
                    content=res['message']['content'],
                    used_sources=sources,
                )
            else:
                raise YAGPTException(f'Failed to query yagpt: {response.content}')
