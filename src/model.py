import asyncio
from typing import List

from schemas.gpt_result import GPTResult
from src.complex_algorythm import get_correct_answer
from src.gpt import query_gpt

SOURCES = [
    "itmo.ru",
    "minobrnauki.gov.ru",
    "ria.ru",
]


async def process_query(query: str):
    question, answers = parse_question(query)

    if len(answers) == 0:
        return None, "", []

    tasks = [query_gpt(question, source) for source in SOURCES]
    results: List[GPTResult] = await asyncio.gather(*tasks)

    answer, correct_results, prior_result_idx = get_correct_answer(answers, results)

    sources_merged = []
    for res in correct_results:
        sources_merged.extend(res.used_sources)
    sources_merged = sources_merged[:3]

    used_sources_urls = [source.url for source in sources_merged]
    used_sources_titles = [source.title for source in sources_merged]

    reasoning = correct_results[prior_result_idx].content + " Подтверждается " + ", ".join(used_sources_titles)

    return answer + 1, reasoning, used_sources_urls


def parse_question(query: str):
    parts = query.split("\n")
    return parts[0], parts[1:]
