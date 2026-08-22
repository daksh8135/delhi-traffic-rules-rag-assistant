# src/eval_runner.py

import os
import json
import time
from generator import Generator
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

JUDGE_PROMPT = """You are evaluating the quality of an AI assistant's answer to a traffic law question.

Question: {question}

Retrieved context the assistant had access to when answering:
{context}

Expected answer (reference, may not be word-for-word exact): {expected}

Assistant's actual answer: {actual}

Score the assistant's answer on these three criteria, each from 1 (bad) to 5 (excellent):
1. FAITHFULNESS - Does the answer ONLY use facts present in the retrieved context above? An answer is UNFAITHFUL if it states specific facts, numbers, or sections that are NOT found in the retrieved context, even if those facts happen to be true in general. If the context says "Structured fine lookup table (no retrieval used)", treat the answer as faithful by design (it did not use retrieval-based generation).
2. RELEVANCE - Does the answer actually address the question asked? Note: if the question is off-topic (unrelated to traffic law) and the assistant correctly declines to answer, that IS relevant and correct behavior — score it high.
3. CORRECTNESS - Does the answer align with the expected answer's key facts (allowing paraphrasing)?

Respond ONLY in this exact format, nothing else:
FAITHFULNESS: <score>
RELEVANCE: <score>
CORRECTNESS: <score>
COMMENT: <one short sentence explaining the scores>
"""


def load_eval_set(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def judge_answer(judge_llm, question: str, expected: str, actual: str, context: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, context=context, expected=expected, actual=actual)
    messages = [
        SystemMessage(content="You are a strict, fair evaluator of AI assistant answers."),
        HumanMessage(content=prompt)
    ]
    response = judge_llm.invoke(messages)
    text = response.content

    # Parse the structured response
    scores = {"faithfulness": None, "relevance": None, "correctness": None, "comment": ""}
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("FAITHFULNESS:"):
            scores["faithfulness"] = int(line.split(":")[1].strip())
        elif line.upper().startswith("RELEVANCE:"):
            scores["relevance"] = int(line.split(":")[1].strip())
        elif line.upper().startswith("CORRECTNESS:"):
            scores["correctness"] = int(line.split(":")[1].strip())
        elif line.upper().startswith("COMMENT:"):
            scores["comment"] = line.split(":", 1)[1].strip()

    return scores


def save_results(results: list, output_path: str):
    """Saves whatever results we have so far — called after EVERY question,
    so a crash mid-run doesn't lose previously completed evaluations."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_eval(eval_path: str, output_path: str):
    generator = Generator()
    judge_llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    eval_items = load_eval_set(eval_path)
    results = []

    for item in eval_items:
        print(f"Running {item['id']}: {item['question'][:50]}...")
        try:
            result = generator.ask(item["question"])
            actual_answer = result["answer"]
            context_used = result["context"]
        except Exception as e:
            actual_answer = f"ERROR: {e}"
            context_used = ""

        scores = judge_answer(judge_llm, item["question"], item["expected_answer"], actual_answer, context_used)

        results.append({
            "id": item["id"],
            "question": item["question"],
            "category": item["category"],
            "actual_answer": actual_answer,
            "scores": scores
        })

        # Save progress after EVERY question, not just at the end —
        # protects against losing all results if a later question crashes
        # or hits a rate limit.
        save_results(results, output_path)

        time.sleep(5)  # avoid hitting Groq's rate limit between questions

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    total_f, total_r, total_c, count = 0, 0, 0, 0
    for r in results:
        s = r["scores"]
        if s["faithfulness"] is not None:
            total_f += s["faithfulness"]
            total_r += s["relevance"]
            total_c += s["correctness"]
            count += 1
        print(f"\n[{r['id']}] {r['question'][:60]}")
        print(f"  Faithfulness: {s['faithfulness']}/5 | Relevance: {s['relevance']}/5 | Correctness: {s['correctness']}/5")
        print(f"  Comment: {s['comment']}")

    if count > 0:
        print("\n" + "-" * 70)
        print(f"AVERAGE  Faithfulness: {total_f/count:.2f}/5 | Relevance: {total_r/count:.2f}/5 | Correctness: {total_c/count:.2f}/5")
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    eval_path = os.path.join(BASE_DIR, "eval", "eval_dataset.json")
    output_path = os.path.join(BASE_DIR, "eval", "eval_results.json")

    run_eval(eval_path, output_path)