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
            "claude": 8192,
            "deepseek": 8192
        }
        self.tokenizers = {
            "gpt": tiktoken.encoding_for_model("gpt-4"),
            "claude": tiktoken.encoding_for_model("gpt-4"),  # similar tokenization
            "deepseek": tiktoken.encoding_for_model("gpt-3.5-turbo")  # similar = ~ 10-20% off
        }

    def _count_tokens(self, text: str) -> int:
        tokenizer = self.tokenizers[self.model_type]
        return len(tokenizer.encode(text))

    def _remove_filler_words(self, text: str) -> str:
        fillers = nl_data.fillers
        doc = self.nlp(text)
        filtered_tokens = []
        for token in doc:
            if token.lemma_.lower() not in fillers:
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

    def _remove_greetings(self, text: str) -> str:
        greetings = nl_data.greetings
        doc = self.nlp(text)
        filtered_tokens = []
        for token in doc:
            if token.lemma_.lower() not in greetings:
                filtered_tokens.append(token.text_with_ws)
        return "".join(filtered_tokens).strip()


    def optimize(self, prompt: str, max_tokens: int = None) -> Dict[str, Any]:
        cleaned = re.sub(r'\s+', ' ', prompt.strip())

        optimized = self._remove_greetings(cleaned)
        optimized = self._remove_filler_words(optimized)
        optimized = self._replace_verbose_phrases(optimized)
        optimized = self._remove_redundant_pronouns(optimized)

        return {
            "optimized_text": optimized,
            "original_tokens": self._count_tokens(cleaned),
            "optimized_tokens": self._count_tokens(optimized),
            "compression_ratio": self._count_tokens(optimized) / self._count_tokens(cleaned)
        }