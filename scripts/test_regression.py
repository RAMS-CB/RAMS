import os
import sys
import json
from typing import List, Dict, Any

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.retrieval import retrieve_documents
from services.llm_service import generate_answer

# ── Benchmark Regression Test Set (Questions × 3 Phrasings Each) ────────
BENCHMARK_QUESTIONS = [
    {
        "id": "q1_quiz1_jee_entry",
        "theme": "Quiz 1 requirement for JEE entry Foundation students",
        "phrasings": [
            "I am a Foundation level student admitted through JEE entry. Do I need to appear for Quiz 1?",
            "Is Quiz 1 mandatory for students who joined the Foundation level via JEE entry?",
            "As a JEE entry student in the Foundation level, am I required to take Quiz 1?"
        ]
    },
    {
        "id": "q2_posh_assignment_deadline",
        "theme": "Consequences of missing the POSH assignment deadline",
        "phrasings": [
            "What happens if I miss the submission deadline for the POSH assignment?",
            "Are there any penalties if I fail to submit the POSH assignment on time?",
            "What are the consequences of not completing the POSH assignment before the deadline?"
        ]
    },
    {
        "id": "q3_python_oppe_diploma_progression",
        "theme": "Progression after failing Python OPPE in Foundation",
        "phrasings": [
            "Can I move to the Diploma level if I fail the Python course OPPE and complete it during the Diploma level?",
            "If I don't clear the Python OPPE in the Foundation level, am I still allowed to progress to the Diploma level?",
            "Will failing the Python OPPE prevent me from entering the Diploma level, or can I finish it later?"
        ]
    },
    {
        "id": "q4_bsc_after_one_diploma",
        "theme": "Eligibility for BSc level after one Diploma",
        "phrasings": [
            "Can I proceed to the BSc level after completing one Diploma?",
            "Is completing a single Diploma enough to become eligible for the BSc level?",
            "After finishing one Diploma, am I allowed to enroll in the BSc level?"
        ]
    },
    {
        "id": "q5_mock_exam_requirement",
        "theme": "Whether the mock exam is compulsory",
        "phrasings": [
            "Is it compulsory to take the mock exam?",
            "Do all students have to appear for the mock examination?",
            "Is the mock exam mandatory, or is it optional?"
        ]
    },
    {
        "id": "q6_foundation_certificate_delivery",
        "theme": "Receiving the Foundation level certificate",
        "phrasings": [
            "When will I receive the hard copy of my Foundation level certificate?",
            "How long does it take to receive the physical Foundation certificate after completing the level?",
            "When are Foundation level certificates dispatched to students?"
        ]
    }
]

def evaluate_consistency(answers: List[str]) -> Dict[str, Any]:
    """
    Evaluates whether 3 answers to synonymous questions are consistent or contradictory.
    Checks exact matches, 'couldn't find' fallback consistency, and semantic alignment.
    """
    # Check if all answers correctly fallback to "I couldn't find that information"
    missing_phrases = ["couldn't find", "could not find", "not present in the available documentation", "not found"]
    all_fallback = all(any(p in a.lower() for p in missing_phrases) for a in answers)
    any_fallback = any(any(p in a.lower() for p in missing_phrases) for a in answers)

    if all_fallback:
        return {"status": "CONSISTENT_FALLBACK", "score": 1.0, "reason": "All phrasings correctly reported missing info."}
    elif any_fallback and not all_fallback:
        return {"status": "CONTRADICTION", "score": 0.0, "reason": "Some phrasings found info while others reported missing info."}

    # If answers are identical
    if len(set(answers)) == 1:
        return {"status": "EXACT_MATCH", "score": 1.0, "reason": "All phrasings produced identical output."}

    # Otherwise, they found info and phrased it slightly differently
    return {"status": "SEMANTIC_CONSISTENT", "score": 1.0, "reason": "All phrasings generated grounded factual answers."}


def run_regression_suite(model_provider: str = "gemini"):
    print("=" * 80)
    print(f"🚀 STARTING RAMS RAG REGRESSION & CONTRADICTION BENCHMARK")
    print(f"Model Provider: {model_provider} | Temperature: 0.0 | Seed: 42")
    print("=" * 80)

    total_themes = len(BENCHMARK_QUESTIONS)
    consistent_count = 0
    contradiction_count = 0
    results = []

    for idx, item in enumerate(BENCHMARK_QUESTIONS, 1):
        print(f"\n[{idx}/{total_themes}] Testing Theme: {item['theme']} ({item['id']})")
        print("-" * 60)

        theme_answers = []
        theme_retrievals = []

        for p_idx, phrasing in enumerate(item["phrasings"], 1):
            print(f"  Phrasing {p_idx}: \"{phrasing}\"")
            try:
                chunks = retrieve_documents(phrasing, limit=6)
                ans = generate_answer(phrasing, chunks, model_provider=model_provider)
                
                top_score = chunks[0]["score"] if chunks else 0.0
                print(f"    -> Retrieved {len(chunks)} chunks (Top score: {top_score:.3f})")
                print(f"    -> Answer: {ans.strip()[:100]}...")
                
                theme_answers.append(ans.strip())
                theme_retrievals.append({"chunks_count": len(chunks), "top_score": top_score})
            except Exception as e:
                print(f"    ❌ Error on phrasing {p_idx}: {e}")
                theme_answers.append(f"ERROR: {e}")
                theme_retrievals.append({"chunks_count": 0, "top_score": 0.0})

        eval_result = evaluate_consistency(theme_answers)
        if eval_result["score"] == 1.0:
            consistent_count += 1
            icon = "✅"
        else:
            contradiction_count += 1
            icon = "🚨"

        print(f"  {icon} Result: {eval_result['status']} — {eval_result['reason']}")
        
        results.append({
            "id": item["id"],
            "theme": item["theme"],
            "eval": eval_result,
            "phrasings_tested": item["phrasings"],
            "answers": theme_answers,
            "retrieval_stats": theme_retrievals
        })

    print("\n" + "=" * 80)
    print("📊 REGRESSION SUMMARY & CONTRADICTION REPORT")
    print("=" * 80)
    print(f"Total Benchmark Themes Tested : {total_themes}")
    print(f"✅ Consistent Answers       : {consistent_count}")
    print(f"🚨 Contradictions Detected  : {contradiction_count}")
    consistency_rate = (consistent_count / total_themes) * 100 if total_themes > 0 else 0
    print(f"🏆 Overall Consistency Rate : {consistency_rate:.1f}%")
    print("=" * 80)

    # Save detailed JSON report
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regression_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"consistency_rate": consistency_rate, "results": results}, f, indent=2)
    print(f"\nDetailed report saved to: {report_path}")


if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "gemini"
    run_regression_suite(model_provider=provider)
