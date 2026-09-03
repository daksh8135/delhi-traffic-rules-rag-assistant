# src/fine_lookup.py

import json

# Words that signal the user is actually asking about a PENALTY, not just mentioning the topic
PENALTY_INDICATOR_WORDS = ["fine", "penalty", "punishment", "challan", "jurmana", "जुर्माना", "सजा"]


class FineLookup:
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.entries = json.load(f)

    def match(self, question):
        question_lower = question.lower()

        # Only consider a lookup match if the question ALSO signals it's asking
        # about a penalty/fine specifically -- not just mentioning the topic.
        asks_about_penalty = any(word in question_lower for word in PENALTY_INDICATOR_WORDS)
        if not asks_about_penalty:
            return None

        for entry in self.entries:
            for keyword in entry["keywords"]:
                if keyword in question_lower:
                    return entry["answer"]
        return None