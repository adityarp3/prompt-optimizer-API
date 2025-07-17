import re
from app.core.optimizer import PromptOptimizer
from app.nl import nl_data

optimizer = PromptOptimizer("gpt")

def test(prompt: str):
    print("_count_tokens")
    print(optimizer._count_tokens(prompt))
    print("_extract_main_intent")
    print(optimizer._extract_main_intent(prompt))
    print("_normalize_commas")
    print(optimizer._normalize_commas(prompt))
    print("_reduce_intensifiers")
    print(optimizer._reduce_intensifiers(prompt))
    print("_remove_request_phrases")
    print(optimizer._remove_request_phrases(prompt))
    print("_remove_filler_words")
    print(optimizer._remove_filler_words(prompt))
    print("_replace_verbose_phrases")
    print(optimizer._replace_verbose_phrases(prompt))
    print("_remove_redundant_pronouns")
    print(optimizer._remove_redundant_pronouns(prompt))
    print("_remove_greetings_and_closings")
    print(optimizer._remove_greetings_and_closings(prompt))
    print("_remove_stylistic_chars")
    print(optimizer._remove_stylistic_chars(prompt))
    print("_prepass_replacements")
    print(optimizer._prepass_replacements(prompt))
    print("_post_cleanup")
    print(optimizer._post_cleanup(prompt))
    print("_clean_whitespace_and_punctuation")
    print(optimizer._clean_whitespace_and_punctuation(prompt))

prompt = "heres a prompt: good day, sunshine. what do you think? can you help, or suggest?"

if __name__ == '__main__':
    test(prompt)