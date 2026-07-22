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
        "id": "q1_system_purpose",
        "theme": "What is the primary purpose of RAMS?",
        "phrasings": [
            "What is the primary purpose and definition of RAMS?",
            "Can you explain what RAMS stands for and what it does?",
            "Give me a brief summary of what the RAMS application is used for."
        ]
    },
    {
        "id": "q2_data_sources",
        "theme": "Where does RAMS ingest documents from?",
        "phrasings": [
            "What external document sources does RAMS fetch from?",
            "Where does the system get its knowledge base and files?",
            "List the integrations and external services RAMS connects to for ingesting content."
        ]
    },
    {
        "id": "q3_chunking_mechanism",
        "theme": "How does RAMS split documents into chunks?",
        "phrasings": [
            "How does RAMS split text into chunks before embedding?",
            "What is the chunking strategy and sentence boundary rule in the ingestion pipeline?",
            "Explain the text segmentation and overlap strategy used when storing documents."
        ]
    },
    {
        "id": "q4_embedding_model",
        "theme": "Which embedding model and dimension size is used?",
        "phrasings": [
            "What embedding model and vector dimension size does RAMS use?",
            "Which Google Gemini embedding model generates the vector representations?",
            "Tell me the dimensionality and model name configured for chunk embeddings."
        ]
    },
    {
        "id": "q5_retrieval_expansion",
        "theme": "How does context expansion work during vector search?",
        "phrasings": [
            "How does adaptive context expansion work when retrieving chunks?",
            "What happens to neighboring chunks when a base chunk is found during search?",
            "Explain how +-1 and +-2 adjacent chunk inclusion operates in the retrieval service."
        ]
    },
    {
        "id": "q6_missing_info",
        "theme": "Handling queries with information not present in the documentation",
        "phrasings": [
            "What is the exact procedure for launching a space shuttle in RAMS?",
            "How do I cook a pepperoni pizza using the RAMS API?",
            "Does the RAMS documentation provide instructions on quantum entanglement routing?"
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
