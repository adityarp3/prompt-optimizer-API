import re
import spacy
from app.nl import nl_data
import tiktoken
from typing import Dict, Any


class PromptOptimizer:
    def __init__(self, model_type: str):
        self.model_type = model_type.lower().strip()
        self.nlp = spacy.load("en_core_web_sm")
        self.model_limits = {
            "gpt": 8192,
            "claude": 100000,
            "deepseek": 32000
        }
        self.tokenizers = {
            "gpt": tiktoken.encoding_for_model("gpt-4"),
            "claude": tiktoken.encoding_for_model("gpt-4"),  # similar tokenization
            "deepseek": tiktoken.encoding_for_model("gpt-3.5-turbo")  # similar = ~ 10-20% off
        }

    def _count_tokens(self, text: str) -> int:
        tokenizer = self.tokenizers[self.model_type]
        return len(tokenizer.encode(text))

    def _normalize_commas(self, text: str) -> str:
        text = re.sub(r'\b(\w+),\s*\1\b', r'\1 \1', text)
        return text

    def _reduce_intensifiers(self, text: str) -> str:
        reductions = nl_data.chatbot_optimizations["intensifier_reductions"]
        result = text
        for verbose, simple in reductions.items():
            result = re.sub(r'\b' + re.escape(verbose) + r'\b', simple, result, flags=re.IGNORECASE)
        return result

    def _remove_request_phrases(self, text: str) -> str:
        request_patterns = nl_data.request_patterns

        result = text
        for pattern in request_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)

        return result

    def _remove_filler_words(self, text: str) -> str:
        fillers = nl_data.fillers
        filler_phrases = [f for f in fillers if ' ' in f]
        filler_single = {f for f in fillers if ' ' not in f}
        result = text
        for phrase in filler_phrases:
            result = re.sub(r'\b' + re.escape(phrase) + r'\b', '', result, flags=re.IGNORECASE)
        doc = self.nlp(result)
        filtered_tokens = []
        for token in doc:
            if token.lemma_.lower() not in filler_single:
                filtered_tokens.append(token.text_with_ws)

        return "".join(filtered_tokens).strip()

    def _replace_verbose_phrases(self, text: str) -> str:
        result = text
        verbose_replacements = nl_data.verbose_replacements
        for verbose, concise in verbose_replacements.items():
            result = re.sub(r'\b' + re.escape(verbose) + r'\b', concise, result, flags=re.IGNORECASE)
        return result

    def _remove_redundant_pronouns(self, text: str) -> str:
        redundant_patterns = nl_data.redundant_patterns
        result = text
        for pattern, replacement in redundant_patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        result = re.sub(r'\s+', ' ', result).strip()
        return result

    def _remove_greetings_and_closings(self, text: str) -> str:
        greetings = nl_data.greetings
        greeting_patterns = nl_data.greeting_patterns

        result = text
        for pattern in greeting_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)

        doc = self.nlp(result)
        filtered_tokens = []
        for token in doc:
            if token.lemma_.lower() not in greetings:
                filtered_tokens.append(token.text_with_ws)

        return "".join(filtered_tokens).strip()

    def _remove_stylistic_chars(self, text: str) -> str:
        result = re.sub(r'[!()";]', '', text)
        result = re.sub(r'[-–—]+', ' ', result)
        result = re.sub(r'[*_~`]', '', result)
        return result

    def _prepass_replacements(self, text: str) -> str:
        pre_patterns = nl_data.pre_patterns
        result = text
        for pattern in pre_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        return result

    def _post_cleanup(self, text: str) -> str:
        cleanup_patterns = nl_data.cleanup_patterns
        result = text
        for pattern, repl in cleanup_patterns:
            result = re.sub(pattern, repl, result, flags=re.IGNORECASE)
        return result.strip()

    def _clean_whitespace_and_punctuation(self, text: str) -> str:
        result = re.sub(r'\s+', ' ', text)
        result = re.sub(r'\s*([.,?:])\s*', r'\1 ', result)
        result = re.sub(r'^[.,?:\s]+|[.,?:\s]+$', '', result)

        if result:
            result = result[0].upper() + result[1:]

        return result.strip()

    def _chatbot_optimize(self, text: str) -> str:
        result = text

        for pattern, replacement in nl_data.chatbot_patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        for full_form, contraction in nl_data.chatbot_optimizations["contractions"].items():
            result = re.sub(r'\b' + re.escape(full_form) + r'\b', contraction, result, flags=re.IGNORECASE)

        for verbose, simple in nl_data.chatbot_optimizations["preposition_simplifications"].items():
            result = re.sub(r'\b' + re.escape(verbose) + r'\b', simple, result, flags=re.IGNORECASE)

        for redundant, simplified in nl_data.chatbot_optimizations["redundant_phrases"].items():
            result = re.sub(r'\b' + re.escape(redundant) + r'\b', simplified, result, flags=re.IGNORECASE)

        for verbose, simple in nl_data.chatbot_optimizations["intensifier_reductions"].items():
            result = re.sub(r'\b' + re.escape(verbose) + r'\b', simple, result, flags=re.IGNORECASE)

        for verbose, simple in nl_data.chatbot_optimizations["conjunction_simplifications"].items():
            result = re.sub(r'\b' + re.escape(verbose) + r'\b', simple, result, flags=re.IGNORECASE)

        for verbose, simple in nl_data.chatbot_optimizations["question_simplifications"].items():
            result = re.sub(r'\b' + re.escape(verbose) + r'\b', simple, result, flags=re.IGNORECASE)

        for verbose, simple in nl_data.chatbot_optimizations["transition_simplifications"].items():
            result = re.sub(r'\b' + re.escape(verbose) + r'\b', simple, result, flags=re.IGNORECASE)

        aggressive_fillers = nl_data.chatbot_optimizations["aggressive_fillers"]
        doc = self.nlp(result)
        filtered_tokens = []
        for token in doc:
            if token.lemma_.lower() not in aggressive_fillers:
                filtered_tokens.append(token.text_with_ws)
        result = "".join(filtered_tokens)

        courtesy_removals = nl_data.chatbot_optimizations["courtesy_removals"]
        doc = self.nlp(result)
        filtered_tokens = []
        for token in doc:
            if token.lemma_.lower() not in courtesy_removals:
                filtered_tokens.append(token.text_with_ws)
        result = "".join(filtered_tokens)

        return result.strip()

    def optimize(self, prompt: str, lvl: str = "aggressive", optim: list = None) -> Dict[str, Any]:
        cleaned = re.sub(r'\s+', ' ', prompt.strip())
        optimized = cleaned

        optimized = self._normalize_commas(optimized)
        optimized = self._post_cleanup(optimized)

        level_configs = {
            "light": ["rem_gr", "rem_req", "intens_redu"],
            "medium": ["rem_gr", "rem_req", "rem_fill", "repl_verb", "prep_rep", "intens_redu"],
            "aggressive": ["rem_gr", "rem_req", "rem_fill", "repl_verb", "prep_rep", "intens_redu", "rem_pron",
                           "rem_style"],
            "chatbot": ["chatbot"]
        }

        optimizations = optim if optim else level_configs.get(lvl, level_configs["aggressive"])

        if "rem_gr" in optimizations:
            optimized = self._remove_greetings_and_closings(optimized)

        if "rem_req" in optimizations:
            optimized = self._remove_request_phrases(optimized)

        if "rem_fill" in optimizations:
            optimized = self._remove_filler_words(optimized)

        if "repl_verb" in optimizations:
            optimized = self._replace_verbose_phrases(optimized)

        if "prep_rep" in optimizations:
            optimized = self._prepass_replacements(optimized)

        if "intens_redu" in optimizations:
            optimized = self._reduce_intensifiers(optimized)

        if "rem_pron" in optimizations:
            optimized = self._remove_redundant_pronouns(optimized)

        if "rem_style" in optimizations:
            optimized = self._remove_stylistic_chars(optimized)

        if "chatbot" in optimizations:
            optimized = self._chatbot_optimize(optimized)

        optimized = self._post_cleanup(optimized)
        optimized = self._clean_whitespace_and_punctuation(optimized)

        original_tokens = self._count_tokens(cleaned)
        optimized_tokens = self._count_tokens(optimized)

        return {
            "optimized_text": optimized,
            "original_tokens": original_tokens,
            "optimized_tokens": optimized_tokens,
            "compression_ratio": optimized_tokens / original_tokens if original_tokens > 0 else 0,
            "applied_optimizations": optimizations
        }
