import os
import json

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

def load_skills():
    skills_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills_v0.jsonl")
    if not os.path.exists(skills_path):
        return []
    skills = []
    with open(skills_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                skills.append(json.loads(line.strip()))
    return skills

def retrieve_skills(prompt: str, top_k: int = 2) -> list[dict]:
    skills = load_skills()
    if not skills:
        return []
        
    query_tokens = [t.lower() for t in prompt.split()]
    
    if HAS_BM25:
        corpus = []
        for s in skills:
            text = f"{s['name']} {s['description']} {s['tip']}"
            corpus.append([t.lower() for t in text.split()])
            
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [skills[i] for i in top_indices if scores[i] > 0.0]
    else:
        # Simple token-matching fallback
        scored = []
        for s in skills:
            text = f"{s['name']} {s['description']} {s['tip']}".lower()
            overlap = sum(1 for t in query_tokens if t in text)
            scored.append((overlap, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for score, s in scored[:top_k] if score > 0]

def inject_skills_to_prompt(prompt: str, top_k: int = 1) -> str:
    retrieved = retrieve_skills(prompt, top_k=top_k)
    if not retrieved:
        return prompt
        
    tips_str = "\n".join([f"- {s['name']}: {s['tip']}" for s in retrieved])
    injected = f"--- EXPERT KERNEL TIPS ---\n{tips_str}\n--------------------------\n\n{prompt}"
    return injected
