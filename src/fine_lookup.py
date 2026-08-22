# src/fine_lookup.py

import json

class FineLookup:
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.entries = json.load(f)

    def match(self, question):
        question_lower = question.lower()
        for entry in self.entries:
            for keyword in entry["keywords"]:
                if keyword in question_lower:
                    return entry["answer"]
        return None  # no match found