import re
import spacy
from app.nl import nl_data
import tiktoken
from typing import Dict, Any

_GLOBAL_NLP_MODEL = None

def _load_global_nlp_model():
    global _GLOBAL_NLP_MODEL
    if _GLOBAL_NLP_MODEL is None:
        _GLOBAL_NLP_MODEL = spacy.load("en_core_web_sm")
    return _GLOBAL_NLP_MODEL


_load_global_nlp_model()

class PromptOptimizer:
    def __init__(self, model_type: str):
        self.model_type = model_type.lower().strip()
        self.nlp = _GLOBAL_NLP_MODEL

        self.model_limits = {
            "gpt": 8192,
            "claude": 100000,
            "deepseek": 32000
        }
        self._tokenizers_cache = {}
        if "gpt-4" not in self._tokenizers_cache:
            self._tokenizers_cache["gpt-4"] = tiktoken.encoding_for_model("gpt-4")
        if "gpt-3.5-turbo" not in self._tokenizers_cache:
            self._tokenizers_cache["gpt-3.5-turbo"] = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def _count_tokens(self, text: str) -> int:
        tokenizer_key = ""
        if self.model_type == "gpt":
            tokenizer_key = "gpt-4"
        elif self.model_type == "claude":
            tokenizer_key = "gpt-4"
        elif self.model_type == "deepseek":
            tokenizer_key = "gpt-3.5-turbo"
        else:
            print(f"Warning: Unknown model type '{self.model_type}'. Using default 'cl100k_base' tokenizer.")
            return len(tiktoken.get_encoding("cl100k_base").encode(text))

        tokenizer = self._tokenizers_cache.get(tokenizer_key)
        if not tokenizer:
            print(f"Error: Tokenizer for key '{tokenizer_key}' not found in cache.")
            return len(tiktoken.get_encoding("cl100k_base").encode(text))

        return len(tokenizer.encode(text))

    def _extract_main_intent(self, text: str) -> str:
        if not text or len(text.strip()) < 10:
            return text.strip()
        words = text.split()
        if len(words) < 5:
            return text.strip()

        doc = self.nlp(text)

        root = None
        for token in doc:
            if token.dep_ == "ROOT":
                root = token
                break

        if not root:
            return text.strip()

        if root.pos_ not in ("VERB", "AUX") and root.tag_ not in ("MD", "VBZ", "VBP", "VBD", "VBN", "VBG"):
            return text.strip()

        if root.tag_ in ("MD", "VBZ", "VBP") and root.lemma_ in (
                "want", "go", "be", "will", "would", "can", "could", "should"):
            main_verb = None
            for child in root.children:
                if child.dep_ in ("xcomp", "ccomp", "advcl", "acl", "conj", "pcomp", "dobj") and child.pos_ == "VERB":
                    main_verb = child
                    break
            if main_verb:
                root = main_verb
            else:
                if root.lemma_ not in ("do", "help"):
                    return text.strip()

        subjects = [tok for tok in root.lefts if tok.dep_ in ("nsubj", "nsubjpass")]
        dobjs = [tok for tok in root.rights if tok.dep_ in ("dobj", "attr", "pobj")]

        prep_phrases = []
        for child in root.children:
            if child.dep_ == "prep":
                prep_obj = None
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        prep_obj = grandchild
                        break
                if prep_obj:
                    prep_phrases.append(f"{child.text} {self._get_span(prep_obj, doc)}")

        parts = []

        if subjects:
            subj_text = self._get_span(subjects[0], doc)
            if subj_text.lower() not in ("i", "me", "my", "you"):
                parts.append(subj_text)

        verb_lemma = root.lemma_
        if root.pos_ == "AUX" or verb_lemma in ("be", "have", "do", "say", "get"):
            parts.append(root.text)
        else:
            parts.append(verb_lemma)

        if dobjs:
            for dobj in sorted(dobjs, key=lambda t: t.i):
                dobj_text = self._get_span(dobj, doc)
                parts.append(dobj_text)

        if prep_phrases:
            parts.extend(prep_phrases[:2])

        if len(parts) < 2:
            return text.strip()

        result = " ".join(parts)
        result = self._clean_whitespace_and_punctuation(result)

        if len(result.split()) < max(2, len(words) * 0.3):
            return text.strip()

        return result

    def _get_span(self, tok: spacy.tokens.Token, doc: spacy.tokens.Doc) -> str:
        subtree_tokens = list(tok.subtree)
        if not subtree_tokens:
            return tok.text
        return doc[subtree_tokens[0].i: subtree_tokens[-1].i + 1].text

    def _normalize_commas(self, text: str) -> str: # word, word -> word word
        text = re.sub(r'\b(\w+),\s*\1\b', r'\1 \1', text)
        return text

    def _reduce_intensifiers(self, text: str) -> str: # very very -> very
        reductions = nl_data.chatbot_optimizations["intensifier_reductions"]
        result = text
        for verbose, simple in reductions.items():
            result = re.sub(r'\b' + re.escape(verbose) + r'\b', simple, result, flags=re.IGNORECASE)
        return result

    def _remove_request_phrases(self, text: str) -> str: # can you please help with -> help with
        request_patterns = nl_data.request_patterns
        result = text
        for pattern, replacement in request_patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result.strip()

    def _remove_filler_words(self, text: str) -> str:
        fillers_set = nl_data.fillers

        filler_phrases_regex = r'\b(?:' + '|'.join(re.escape(f) for f in fillers_set if ' ' in f) + r')\b'
        if filler_phrases_regex != r'\b(?:)\b':
            text = re.sub(filler_phrases_regex, '', text, flags=re.IGNORECASE)

        doc = self.nlp(text)
        filtered_tokens = []
        for token in doc:
            if token.lemma_.lower() not in fillers_set or ' ' in token.text:
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

    def _remove_greetings_and_closings(self, text: str) -> str: # hey there, help with word, thank you. -> help with word
        result = text
        for pattern in nl_data.greeting_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)

        doc = self.nlp(result)
        filtered_tokens = []
        greetings_set = nl_data.greetings
        for token in doc:
            if token.lemma_.lower() not in greetings_set:
                filtered_tokens.append(token.text_with_ws)

        return "".join(filtered_tokens).strip()

    def _remove_stylistic_chars(self, text: str) -> str: # ========= -> ''
        result = re.sub(r'[!()";]', '', text)
        result = re.sub(r'[-–—]+', ' ', result)
        result = re.sub(r'[*_~`]', '', result)
        return result

    def _prepass_replacements(self, text: str) -> str: # common intro statement -> *remove*
        pre_patterns = nl_data.pre_patterns
        result = text
        for pattern in pre_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        return result

    def _post_cleanup(self, text: str) -> str: # common patterns -> *remove/alter*
        cleanup_patterns = nl_data.cleanup_patterns
        result = text
        for pattern, repl in cleanup_patterns:
            result = re.sub(pattern, repl, result, flags=re.IGNORECASE)
        return result.strip()

    def _clean_whitespace_and_punctuation(self, text: str) -> str: # word   ,  word -> Word word
        result = re.sub(r'\s+', ' ', text)
        result = re.sub(r'\s*([.,?!:])\s*', r'\1 ', result)
        result = re.sub(r'^\s*([.,?!:])\s*', r'\1 ', result)
        result = result.strip()

        if result:
            result = result[0].upper() + result[1:]

        return result.strip()

    def _chatbot_optimize(self, text: str) -> str:
        result = text
        chatbot_opts = nl_data.chatbot_optimizations

        for pattern, replacement in chatbot_opts["chatbot_patterns"]:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        for full_form, contraction in chatbot_opts["contractions"].items():
            result = re.sub(r'\b' + re.escape(full_form) + r'\b', contraction, result, flags=re.IGNORECASE)

        for verbose, simple in chatbot_opts["preposition_simplifications"].items():
            result = re.sub(r'\b' + re.escape(verbose) + r'\b', simple, result, flags=re.IGNORECASE)

        for redundant, simplified in chatbot_opts["redundant_phrases"].items():
            result = re.sub(r'\b' + re.escape(redundant) + r'\b', simplified, result, flags=re.IGNORECASE)

        for verbose, simple in chatbot_opts["intensifier_reductions"].items():
            result = re.sub(r'\b' + re.escape(verbose) + r'\b', simple, result, flags=re.IGNORECASE)

        for verbose, simple in chatbot_opts["conjunction_simplifications"].items():
            result = re.sub(r'\b' + re.escape(verbose) + r'\b', simple, result, flags=re.IGNORECASE)

        for verbose, simple in chatbot_opts["question_simplifications"].items():
            result = re.sub(r'\b' + re.escape(verbose) + r'\b', simple, result, flags=re.IGNORECASE)

        for verbose, simple in chatbot_opts["transition_simplifications"].items():
            result = re.sub(r'\b' + re.escape(verbose) + r'\b', simple, result, flags=re.IGNORECASE)

        doc = self.nlp(result)
        filtered_tokens = []

        aggressive_fillers_set = chatbot_opts["aggressive_fillers"]
        courtesy_removals_set = chatbot_opts["courtesy_removals"]

        lemmas_to_remove = aggressive_fillers_set.union(courtesy_removals_set)

        for token in doc:
            if token.lemma_.lower() not in lemmas_to_remove or ' ' in token.text:
                filtered_tokens.append(token.text_with_ws)

        result = "".join(filtered_tokens).strip()

        return result

    def optimize(self, prompt: str, lvl: str = "aggressive", optim: list = None) -> Dict[str, Any]:
        cleaned = re.sub(r'\s+', ' ', prompt.strip())
        optimized = cleaned

        optimized = self._normalize_commas(optimized)

        level_configs = {
            "light": ["rem_gr", "rem_req", "intens_redu"],
            "medium": ["rem_gr", "rem_req", "rem_fill", "repl_verb", "prep_rep", "intens_redu"],
            "aggressive": ["rem_gr", "rem_req", "rem_fill", "repl_verb", "prep_rep", "intens_redu", "rem_pron",
                           "rem_style", "ext_mi"],
            "chatbot": ["chatbot"]
        }

        optimizations = optim if optim else level_configs.get(lvl, level_configs["aggressive"])

        if "rem_gr" in optimizations:
            optimized = self._remove_greetings_and_closings(optimized)

        if "rem_req" in optimizations:
            optimized = self._remove_request_phrases(optimized)

        if "rem_fill" in optimizations and "chatbot" not in optimizations:
            optimized = self._remove_filler_words(optimized)

        if "rem_pron" in optimizations:
            optimized = self._remove_redundant_pronouns(optimized)

        if "rem_style" in optimizations:
            optimized = self._remove_stylistic_chars(optimized)

        if "prep_rep" in optimizations:
            optimized = self._prepass_replacements(optimized)

        if "repl_verb" in optimizations:
            optimized = self._replace_verbose_phrases(optimized)

        if "intens_redu" in optimizations:
            optimized = self._reduce_intensifiers(optimized)

        if "ext_mi" in optimizations:
            optimized = self._extract_main_intent(optimized)

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