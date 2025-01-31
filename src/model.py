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

    tasks = [query_gpt(question, source) for source in SOURCES]
    results: List[GPTResult] = await asyncio.gather(*tasks)

    if len(answers) == 0:
        answer, correct_results, prior_result_idx = None, results, results.index(max(results, key=lambda x: x.source_confidence))
    else:
        answer, correct_results, prior_result_idx = get_correct_answer(answers, results)
        answer += 1

    sources_merged = []
    for res in correct_results:
        sources_merged.extend(res.used_sources)
    sources_merged = sources_merged[:3]

    used_sources_urls = [source.url for source in sources_merged]
    used_sources_titles = [source.title for source in sources_merged]

    reasoning = correct_results[prior_result_idx].content + " Ответ дан с помощью YandexGPT. Источники: " + ", ".join(used_sources_titles)

    return answer, reasoning, used_sources_urls


def parse_question(query: str):
    if "?" not in query:
        return query, []
    parts = query.split("?")
    question = parts[0]
    if '\n' not in parts[1]:
        return question, []

    answers = parts[1].strip('\n')
    answers = answers.split('\n')
    return question, answers
