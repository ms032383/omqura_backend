import json
import requests
import time

API_URL = "http://127.0.0.1:8000/query"
QUESTIONS_FILE = "neurosurgical_test_questions.json"

def run_tests():
    print("=" * 60)
    print("   NEURO-AI CORE ENGINE - PHASE 1 VERIFICATION HARNESS   ")
    print("=" * 60)
    
    try:
        with open(QUESTIONS_FILE, "r") as f:
            questions = json.load(f)
    except FileNotFoundError:
        print(f"Error: {QUESTIONS_FILE} not found. Run from the project root directory.")
        return
    
    total = len(questions)
    passed = 0
    results_report = []

    print(f"Loaded {total} clinical test questions. Querying API at: {API_URL}\n")

    for i, q in enumerate(questions, 1):
        q_id = q.get("question_id", f"Q{i:03d}")
        query_text = q.get("query")
        
        print(f"[{i}/{total}] Testing {q_id}: '{query_text[:60]}...'")
        
        start_time = time.time()
        try:
            response = requests.post(API_URL, json={"query": query_text}, timeout=60)
            elapsed = time.time() - start_time
            
            if response.status_code != 200:
                print(f"  [FAIL] HTTP status {response.status_code}")
                results_report.append((q_id, "HTTP Error", elapsed))
                continue
                
            data = response.json()
            status = data.get("status")
            answer = data.get("answer_text", "")
            citations = data.get("citations", [])
            
            if status != "success":
                print(f"  [FAIL] API Status: {status}")
                results_report.append((q_id, f"API Status: {status}", elapsed))
                continue
                
            # Verify no un-hydrated placeholders remain in the answer
            if "__SOURCE_" in answer:
                print("  [FAIL] Hallucinated/Un-hydrated citation placeholders found!")
                results_report.append((q_id, "Unhydrated citations", elapsed))
                continue
                
            # Verify citations are present
            if not citations:
                print("  [FAIL] Citations list is empty!")
                results_report.append((q_id, "Missing citations", elapsed))
                continue
                
            # Print citations returned
            citation_urls = [c.get("url") for c in citations]
            print(f"  [PASS] ({elapsed:.2f}s) - {len(citations)} citation(s): {citation_urls}")
            passed += 1
            results_report.append((q_id, "PASS", elapsed))
            
        except requests.exceptions.RequestException as e:
            elapsed = time.time() - start_time
            print(f"  [FAIL] Connection error: {e}")
            results_report.append((q_id, "Connection Error", elapsed))
            
        time.sleep(0.1) # brief pause

    print("\n" + "=" * 60)
    print("                      SCORECARD SUMMARY                      ")
    print("=" * 60)
    print(f"{'Question ID':<15} | {'Status/Error':<25} | {'Latency (s)':<12}")
    print("-" * 60)
    for q_id, status, elapsed in results_report:
        print(f"{q_id:<15} | {status:<25} | {elapsed:<12.2f}")
    print("=" * 60)
    
    score_percentage = (passed / total) * 100
    print(f"Total Tests Run: {total}")
    print(f"Tests Passed:    {passed} / {total} ({score_percentage:.1f}%)")
    
    if passed >= 18:
        print("\n[SUCCESS] EXIT CRITERIA MET! Phase 1 completed successfully (Score >= 18/20).")
    else:
        print("\n[WARNING] EXIT CRITERIA NOT MET. Needs >= 18/20 passing tests to exit Phase 1.")

if __name__ == "__main__":
    run_tests()
