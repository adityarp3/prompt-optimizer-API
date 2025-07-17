import re
import spacy
from app.nl import nl_data
import tiktoken
from typing import Dict, Any, Tuple, List, Set

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

        important_constraints = self._extract_constraints(doc)

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
                "want", "go", "be", "will", "would", "can", "could", "should", "need"):
            main_verb = None
            for child in root.children:
                if child.dep_ in ("xcomp", "ccomp", "advcl", "acl", "conj", "pcomp", "dobj") and child.pos_ == "VERB":
                    main_verb = child
                    break
            if main_verb:
                root = main_verb
            else:
                if root.lemma_ not in ("do", "help", "need"):
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

        if important_constraints:
            parts.extend(important_constraints)

        if len(parts) < 2:
            return text.strip()

        result = " ".join(parts)
        result = self._clean_whitespace_and_punctuation(result)

        min_length = max(3, len(words) * 0.4) if not important_constraints else max(2, len(words) * 0.2)
        if len(result.split()) < min_length:
            return text.strip()

        return result

    def _extract_constraints(self, doc) -> list:
        constraints = []

        constraint_words = {"under", "over", "above", "below", "within", "exactly", "approximately", "about",
                            "at least", "at most", "up to"}

        for token in doc:
            if token.lemma_ in constraint_words:
                constraint_phrase = self._get_full_constraint_phrase(token, doc)
                if constraint_phrase:
                    constraints.append(constraint_phrase)
                    break

        return constraints

    def _get_full_constraint_phrase(self, token, doc) -> str:
        phrase_tokens = [token]

        i = token.i + 1
        while i < len(doc):
            current_token = doc[i]

            if current_token.text in (".", "!", "?", ",") and i > token.i + 2:
                break

            if current_token.like_num or current_token.ent_type_ in ("CARDINAL", "QUANTITY"):
                phrase_tokens.append(current_token)

                if i + 1 < len(doc):
                    next_token = doc[i + 1]
                    if next_token.pos_ in ("NOUN", "PROPN") or next_token.lemma_ in (
                            "word", "words", "character", "characters", "line", "lines", "page", "pages"):
                        phrase_tokens.append(next_token)
                        break
                break

            elif current_token.pos_ in ("DET", "ADP") and current_token.lemma_ in ("a", "an", "the"):
                phrase_tokens.append(current_token)

            elif current_token.pos_ in ("VERB", "ADJ", "ADV") and current_token.lemma_ not in ("least", "most"):
                break

            i += 1
        if len(phrase_tokens) >= 2:
            return " ".join([t.text for t in phrase_tokens])

        return None

    def _get_span(self, tok: spacy.tokens.Token, doc: spacy.tokens.Doc) -> str:
        subtree_tokens = list(tok.subtree)
        if not subtree_tokens:
            return tok.text
        return doc[subtree_tokens[0].i: subtree_tokens[-1].i + 1].text

    def _normalize_commas(self, text: str) -> str:
        text = re.sub(r',{2,}', ',', text)
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
        for pattern, replacement in request_patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result.strip()

    def _remove_filler_words(self, text: str) -> str:
        fillers_set = nl_data.fillers
        quoted_segments = re.findall(r'"[^"]*"', text)

        temp_text = text
        quote_placeholders = {}
        for i, quote in enumerate(quoted_segments):
            placeholder = f"__QUOTE_{i}__"
            quote_placeholders[placeholder] = quote
            temp_text = temp_text.replace(quote, placeholder, 1)

        filler_phrases_regex = r'\b(?:' + '|'.join(re.escape(f) for f in fillers_set if ' ' in f) + r')\b'
        if filler_phrases_regex != r'\b(?:)\b':
            temp_text = re.sub(filler_phrases_regex, '', temp_text, flags=re.IGNORECASE)

        doc = self.nlp(temp_text)
        filtered_tokens = []
        for token in doc:
            if token.lemma_.lower() not in fillers_set or ' ' in token.text or token.text.startswith('__QUOTE_'):
                filtered_tokens.append(token.text_with_ws)

        result = "".join(filtered_tokens).strip()

        for placeholder, quote in quote_placeholders.items():
            result = result.replace(placeholder, quote)

        return result

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

    def _remove_stylistic_chars(self, text: str) -> str:
        quoted_segments = re.findall(r'"[^"]*"', text)
        temp_text = text
        quote_placeholders = {}

        for i, quote in enumerate(quoted_segments):
            placeholder = f"__QUOTE_{i}__"
            quote_placeholders[placeholder] = quote
            temp_text = temp_text.replace(quote, placeholder, 1)

        result = re.sub(r'[!()";]', '', temp_text)
        result = re.sub(r'[-–—]+', ' ', result)
        result = re.sub(r'[*_~`]', '', result)

        for placeholder, quote in quote_placeholders.items():
            result = result.replace(placeholder, quote)

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
        ellipsis_placeholder = "__ELLIPSIS__"

        result = re.sub(r'\.{3,}', ellipsis_placeholder, result)
        result = re.sub(r'\s*([,?!:])\s*(?!")', r'\1 ', result)
        result = re.sub(r'\s*(\.)(?!\.)(?!__ELLIPSIS__)\s*(?!")', r'\1 ', result)
        result = re.sub(r'\s*([.,?!:])\s*(?=")', r'\1', result)
        result = re.sub(r'^\s*([.,?!:])\s*', r'\1 ', result)
        result = result.replace(ellipsis_placeholder, '...')

        result = result.strip()
        if result:
            result = result[0].upper() + result[1:]

        return result.strip()

    def _replace_actions_with_placeholders(self, text: str) -> Tuple[str, Dict[str, str]]:
        temp_text = text
        action_placeholders = {}
        counter = 0

        all_action_patterns = nl_data.chb_action_patterns

        def replacer(match):
            nonlocal counter
            placeholder = f"__ACTION_PH_{counter}__"
            action_placeholders[placeholder] = match.group(0)
            counter += 1
            return placeholder

        for pattern_str in all_action_patterns:
            temp_text = re.sub(pattern_str, replacer, temp_text)

        return temp_text, action_placeholders

    def _restore_actions_from_placeholders(self, text: str, action_placeholders: Dict[str, str]) -> str:
        result = text
        for placeholder, original_action in action_placeholders.items():
            result = result.replace(placeholder, original_action)
        return result

    def _chatbot_optimize(self, text: str) -> str:
        text_with_placeholders, action_map = self._replace_actions_with_placeholders(text)
        result = text_with_placeholders

        chatbot_opts = nl_data.chatbot_optimizations
        protected_elements = self._identify_rp_elements(result)
        result = self._apply_rp_aware_optimizations(result, chatbot_opts, protected_elements)
        result = self._final_rp_cleanup(result, chatbot_opts, protected_elements)
        result = self._restore_actions_from_placeholders(result, action_map)

        return result

    def _identify_rp_elements(self, text: str) -> Dict[str, List[str]]:
        elements = {
            'actions': [],
            'dialogue': [],
            'character_traits': [],
            'emotions': [],
            'narrative_voice': []
        }

        dialogue_matches = re.findall(r'"([^"]+)"', text)
        elements['dialogue'].extend(dialogue_matches)

        character_indicators = nl_data.chb_character_indicators
        for pattern in character_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            elements['character_traits'].extend(matches)

        return elements

    def _apply_rp_aware_optimizations(self, text: str, chatbot_opts: Dict, protected_elements: Dict) -> str:
        result = text
        rp_patterns_to_apply = nl_data.chb_rp_patterns

        for pattern, replacement in rp_patterns_to_apply:
            if any(skip_word in pattern.lower() for skip_word in ['known', 'any', 'the']):
                continue

            try:
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            except re.error:
                continue

        result = self._selective_optimization(result, chatbot_opts)

        result = re.sub(r'\*([^*\n]+)\*\s+\*([^*\n]+)\*', r'*\1 \2*', result)

        return result

    def _selective_optimization(self, text: str, chatbot_opts: Dict) -> str:
        result = text

        for full_form, contraction in chatbot_opts["contractions"].items():
            if not self._is_in_dialogue(full_form, result):
                result = re.sub(r'\b' + re.escape(full_form) + r'\b', contraction, result, flags=re.IGNORECASE)

        for verbose, simple in chatbot_opts["preposition_simplifications"].items():
            if not self._is_character_speech_pattern(verbose, result):
                result = re.sub(r'\b' + re.escape(verbose) + r'\b', simple, result, flags=re.IGNORECASE)

        safe_redundant_phrases = {
            k: v for k, v in chatbot_opts["redundant_phrases"].items()
            if k not in {"known", "any known", "the known"}
        }

        for redundant, simplified in safe_redundant_phrases.items():
            result = re.sub(r'\b' + re.escape(redundant) + r'\b', simplified, result, flags=re.IGNORECASE)

        for verbose, simple in chatbot_opts["intensifier_reductions"].items():
            if not self._is_in_dialogue(verbose, result):
                result = re.sub(r'\b' + re.escape(verbose) + r'\b', simple, result, flags=re.IGNORECASE)

        for verbose, simple in chatbot_opts["conjunction_simplifications"].items():
            result = re.sub(r'\b' + re.escape(verbose) + r'\b', simple, result, flags=re.IGNORECASE)

        for verbose, simple in chatbot_opts["question_simplifications"].items():
            if not self._is_character_questioning_style(verbose, result):
                result = re.sub(r'\b' + re.escape(verbose) + r'\b', simple, result, flags=re.IGNORECASE)

        for verbose, simple in chatbot_opts["transition_simplifications"].items():
            result = re.sub(r'\b' + re.escape(verbose) + r'\b', simple, result, flags=re.IGNORECASE)

        return result

    def _final_rp_cleanup(self, text: str, chatbot_opts: Dict, protected_elements: Dict) -> str:
        original_text = text
        doc = self.nlp(text)
        filtered_tokens = []
        rp_safe_fillers = self._get_rp_safe_fillers(chatbot_opts["removable_fillers"])
        rp_safe_courtesy = self._get_rp_safe_courtesy_removals(chatbot_opts["courtesy_removals"])

        lemmas_to_remove = rp_safe_fillers.union(rp_safe_courtesy)

        for token in doc:
            should_keep = self._should_keep_token_for_rp(token, lemmas_to_remove, protected_elements)

            if should_keep:
                filtered_tokens.append(token.text_with_ws)

        result = "".join(filtered_tokens).strip()

        result = re.sub(r'\.(\s*\.)+', '...', result)
        result = re.sub(r'\.{4,}', '...', result)

        result = re.sub(r'\s+([.,!?;:])', r'\1', result)
        result = re.sub(r'([.,!?;:])\s+(["\'])', r'\1 \2', result)
        result = re.sub(r'(["\'])\s+([.,!?;:])', r'\1\2', result)

        result = re.sub(r'(?<=\w)\s+([\'"])', r'\1', result)
        result = re.sub(r'([\'"])\s+(?=\w)', r'\1', result)

        result = re.sub(r'[ \t]{2,}', ' ', result)
        result = re.sub(r'\n\s*\n', '\n\n', result)

        result = result.strip()

        return result

    def _is_character_speech_pattern(self, phrase: str, text: str) -> bool:
        dialogue_pattern = rf'"[^"]*\b{re.escape(phrase)}\b[^"]*"'
        return bool(re.search(dialogue_pattern, text, re.IGNORECASE))

    def _is_in_dialogue(self, phrase: str, text: str) -> bool:
        dialogue_sections = re.findall(r'"([^"]+)"', text)
        return any(phrase.lower() in section.lower() for section in dialogue_sections)

    def _is_character_questioning_style(self, phrase: str, text: str) -> bool:
        return (self._is_in_dialogue(phrase, text) or
                bool(re.search(
                    rf'\b(?:always|usually|tends to|often)\s+(?:asks?|says?|wonders?)\b.*\b{re.escape(phrase)}\b', text,
                    re.IGNORECASE)))

    def _get_rp_safe_fillers(self, removable_fillers: List[str]) -> Set[str]:
        rp_preserve = nl_data.chb_preserve
        safe_fillers = []
        for filler in removable_fillers:
            if filler in ['umm', 'uhh', 'hmm', 'uhm', 'ummm', 'uhhh', 'uhmm']:
                safe_fillers.append(filler)
        return set(safe_fillers) - rp_preserve

    def _get_rp_safe_courtesy_removals(self, courtesy_removals: List[str]) -> Set[str]:
        rp_preserve = nl_data.chb_preserve_2
        return set() - rp_preserve

    def _should_keep_token_for_rp(self, token, lemmas_to_remove: Set[str], protected_elements: Dict) -> bool:
        if not token.text.strip() or not token.text.isalpha():
            return True

        if 'dialogue' in protected_elements:
            for dialogue_segment in protected_elements['dialogue']:
                if token.text.lower() in dialogue_segment.lower():
                    return True

        if 'character_traits' in protected_elements:
            for trait_phrase in protected_elements['character_traits']:
                if token.text.lower() in trait_phrase.lower():
                    return True
        if token.lemma_.lower() in lemmas_to_remove:
            if token.text.lower() in {'known', 'any', 'the', 'a', 'an', 'this', 'that', 'these', 'those'}:
                return True

            if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'}:
                return True
        return True

    def optimize(self, prompt: str, lvl: str = "aggressive", optim: list = None) -> Dict[str, Any]:
        cleaned = re.sub(r'\s+', ' ', prompt.strip())
        optimized = cleaned

        if "chatbot" not in (optim or []):
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