from app.core.optimizer import PromptOptimizer

optimizer = PromptOptimizer("gpt")

def test(prompt: str):
    print("_count_tokens")
    print(optimizer._count_tokens(prompt))

    print("_remove_greetings_and_closings") # rem_gr
    p=optimizer._remove_greetings_and_closings(prompt)
    print(p)

    print("_remove_request_phrases") # rem_req
    p=optimizer._remove_request_phrases(p)
    print(p)

    print("_remove_filler_words") # rem_fill
    p=optimizer._remove_filler_words(p)
    print(p)

    print("_remove_redundant_pronouns") # rem_pron
    p=optimizer._remove_redundant_pronouns(p)
    print(p)

    print("_remove_stylistic_chars") # rem_Style
    p=optimizer._remove_stylistic_chars(p)
    print(p)

    print("_prepass_replacements") # prep_rep
    p=optimizer._prepass_replacements(p)
    print(p)

    print("_replace_verbose_phrases") # rem_verb
    p=optimizer._replace_verbose_phrases(p)
    print(p)

    print("_reduce_intensifiers") # intens_redu
    p=optimizer._reduce_intensifiers(p)
    print(p)

    print("_extract_main_intent")# ext_mi
    p=optimizer._extract_main_intent(p)
    print(p)

    print("_normalize_commas")
    p=optimizer._normalize_commas(p)
    print(p)



    print("_post_cleanup")
    p=optimizer._post_cleanup(p)
    print(p)
    print("_clean_whitespace_and_punctuation")
    p=optimizer._clean_whitespace_and_punctuation(p)
    print(p)
    print(f"tokens after: {optimizer._count_tokens(p)}")

prompt = "so this is a new prompt, i would like to see how good this does, this is your first time testing blind"

if __name__ == '__main__':
    test(prompt)