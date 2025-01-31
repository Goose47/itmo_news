from typing import List
import editdistance
import numpy as np
import re

from schemas.gpt_result import GPTResult


def split_into_words(text):
    cleaned_text = re.sub(r'[^\w\s]', ' ', text)
    words = cleaned_text.split()
    return words


def get_correct_answer(answers: List[str], results: List[GPTResult]) -> (int, List[GPTResult], int):
    answer_confidences = [0] * len(answers)
    chosen_answers = [0] * len(results)
    confident_sources = []
    for result_idx, result in enumerate(results):
        words = split_into_words(result.content)
        answer_distribution = [0] * len(answers)
        source_confidence = 0
        for answer_idx, answer in enumerate(answers):
            words_answer = split_into_words(answer)
            result_word = ' '.join(words_answer)
            answer_probability = 0
            for i in range(len(words) - len(words_answer) + 1):
                target_word = ' '.join(words[i:i+len(words_answer)])
                distance = editdistance.eval(target_word, result_word)
                probability = (1 - distance / max(len(target_word), len(result_word))) ** 2
                answer_probability = max(answer_probability, probability)
            answer_confidence = answer_probability * result.source_confidence
            source_confidence = max(source_confidence, answer_probability)
            answer_confidences[answer_idx] += answer_confidence
            answer_distribution[answer_idx] = answer_probability
        chosen_answers[result_idx] = np.argmax(answer_distribution)
        if source_confidence == 1:
            confident_sources.append(result_idx)

    if len(confident_sources) == 0:
        confident_sources.append(np.argmax(list(map(lambda x: x.source_confidence, results))))

    prior_idx = 0
    for source in confident_sources:
        if results[source].source_confidence >= results[prior_idx].source_confidence:
            prior_idx = source

    return np.argmax(answer_confidences), list(map(lambda x: results[x], confident_sources)), prior_idx
