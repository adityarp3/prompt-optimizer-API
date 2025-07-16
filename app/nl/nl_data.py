fillers = {
            "please", "kindly", "just",
            "could", "would", "can", "may", "might",
            "perhaps", "maybe", "possibly", "potentially",
            "I think", "I believe", "I feel", "in my opinion",
            "you", "me", "I"
        }
verbose_replacements = {
            "in order to": "to",
            "due to the fact that": "because",
            "at this point in time": "now",
            "for the purpose of": "for",
            "with regard to": "about",
            "in spite of the fact that": "although",
            "a large number of": "many",
            "a small number of": "few",
            "a number of": "several",
            "has the ability to": "can",
            "is able to": "can",
            "is required to": "must",
            "it is necessary to": "must",
            "in the event that": "if",
            "take into consideration": "consider",
            "give consideration to": "consider",
            "make a decision": "decide",
            "make use of": "use",
            "conduct an analysis of": "analyze",
            "provide an explanation of": "explain",
            "with the exception of": "except",
            "at this time": "now",
            "at a later date": "later",
        }

redundant_patterns = [
            (r'\byou help me\b', 'help'),
            (r'\byou do this\b', 'do this'),
            (r'\byou can help\b', 'help'),
            (r'\byou should\b', 'should'),
            (r'\byou need to\b', 'need to'),
            (r'\byou must\b', 'must'),
            (r'\bhelp me\b', 'help'),
            (r'\bshow me\b', 'show'),
            (r'\btell me\b', 'tell'),
            (r'\bI want you to\b', ''),
            (r'\bI need you to\b', ''),
            (r'\bcan you\b', 'can'),
            (r'\bwill you\b', 'will'),
        ]

greetings = {
    "hello", "hi", "hey", "greetings",
    "good morning", "good afternoon", "good evening",
    "dear", "to whom it may concern",
    "hi there", "hello there",
    "hey there", "good day",
    "howdy", "yo", "sup"
}