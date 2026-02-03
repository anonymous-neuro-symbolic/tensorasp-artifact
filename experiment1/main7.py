import os
import math
import numpy as np
import clingo

from typing import Dict, List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ============================
# Configuration
# ============================

EPOCHS = 8
NOISE_START = 0.00
NOISE_END = 0.25
LEARNING_RATE = 0.05
ATTENTION_ON = True

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============================
# Utilities
# ============================

def log(title):
    print("\n" + "-" * 25 + title + "-" * 25)

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-9)

# ============================
# External files
# ============================

INTERVIEW_FILE = "data/interview.txt"
ASP_FILE = "asp/program2.lp"

if not os.path.exists(INTERVIEW_FILE):
    with open(INTERVIEW_FILE, "w") as f:
        f.write(
            "The child was on a football field. "
            "He was playing with a ball. "
            "Sometimes he lost focus and looked away."
        )

if not os.path.exists(ASP_FILE):
    with open(ASP_FILE, "w") as f:
        f.write("""
% --- Facts from projections ---
in_place.
low_time :- not high_time.

% --- Involvement rules ---
non_involv :- in_place, low_time, not rel_object.
part_involv :- in_place, low_time, rel_object, not eng_rel.
full_involv :- in_place, high_time, rel_object, eng_rel.

% --- Constraints ---
:- full_involv, non_involv.
""")

# ============================
# Step 1 — Embeddings
# ============================

model = SentenceTransformer(EMBEDDING_MODEL)

def embed_text(text: str):
    return normalize(model.encode(text))

with open(INTERVIEW_FILE) as f:
    interview_text = f.read()

g_atoms = {
    "place_ctx": embed_text("football field"),
    "rel_object_ctx": embed_text("ball"),
    "engagement_ctx": embed_text(interview_text)
}

# ============================
# Step 2 — Attention
# ============================

def attention_aggregate(vectors: Dict[str, np.ndarray], use_attention=True):
    if not use_attention:
        return normalize(np.mean(list(vectors.values()), axis=0))

    keys = list(vectors.keys())
    M = np.stack([vectors[k] for k in keys])
    sim = cosine_similarity(M, M)
    weights = np.exp(sim) / np.sum(np.exp(sim), axis=1, keepdims=True)
    attended = weights @ M
    return normalize(np.mean(attended, axis=0))

# ============================
# Step 3 — Projection (symbolic)
# ============================

thresholds = {
    "rel_object": 0.35,
    "eng_rel": 0.40,
    "high_time": 0.50
}

def project_symbols(embs, theta):
    agg = attention_aggregate(embs, ATTENTION_ON)
    return {
        "rel_object": cosine_similarity([embs["rel_object_ctx"]], [agg])[0, 0] > theta["rel_object"],
        "eng_rel": cosine_similarity([embs["engagement_ctx"]], [agg])[0, 0] > theta["eng_rel"],
        "high_time": cosine_similarity([embs["place_ctx"]], [agg])[0, 0] > theta["high_time"],
    }

# ============================
# Step 4 — ASP via Clingo API
# ============================

def run_clingo(symbols, program_path=ASP_FILE):
    ctl = clingo.Control(["0"])
    ctl.load(program_path)

    facts = [f"{k}." for k, v in symbols.items() if v]
    ctl.add("base", [], "\n".join(facts))
    ctl.ground([("base", [])])

    models = []

    def on_model(model):
        models.append(sorted(str(sym) for sym in model.symbols(shown=True)))

    ctl.solve(on_model=on_model)
    return models

# ============================
# Step 5 — Learning
# ============================

def loss(answer_sets):
    if not answer_sets:
        return 0.5
    atoms = answer_sets[0]
    if "full_involv" in atoms:
        return 1.0
    if "part_involv" in atoms:
        return 0.2
    return 0.5

def update_theta(theta, L):
    theta = theta.copy()
    theta["rel_object"] += LEARNING_RATE * (L - 0.5)
    theta["eng_rel"] += LEARNING_RATE * (L - 0.5)
    return theta

# ============================
# Main Loop
# ============================

losses = []
answer_sets = []

log("STARTING TRAINING")

for epoch in range(EPOCHS):
    noise = NOISE_START + (NOISE_END - NOISE_START) * (epoch / EPOCHS)
    noisy_embs = {
        k: normalize(v + np.random.normal(0, noise, v.shape))
        for k, v in g_atoms.items()
    }

    symbols = project_symbols(noisy_embs, thresholds)
    out = run_clingo(symbols)

    L = loss(out)
    thresholds = update_theta(thresholds, L)

    losses.append(L)
    answer_sets.append(out)

    print(f"[Epoch {epoch}] Noise={noise:.2f} Loss={L:.2f}")
    print(out)

# ============================
# Plot Loss (RQ4 visual)
# ============================

plt.plot(losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning from Answer Sets")
plt.show()

# ============================
# Final RQ Reporting
# ============================

log("RQ1 — Mapping geometric → symbolic")
final_symbols = project_symbols(g_atoms, thresholds)
print("Final projections:", final_symbols)

log("RQ2 — Fixed-point convergence")
print("Answer sets stabilized in last epochs:")
for a in answer_sets[-3:]:
    print(a)

log("RQ3 — NAF ambiguity")
print("Early vs late answer sets:")
print("Epoch 0:", answer_sets[0])
print("Epoch -1:", answer_sets[-1])

log("RQ4 — Overhead / Learning stats")
print(f"Epochs: {EPOCHS}")
print(f"Final thresholds: {thresholds}")
print(f"Loss trajectory: {losses}")

# ============================================================
# NEW SECTION — Natural Language Interpretation (added)
# ============================================================

log("Natural Language Interpretation")

ATOM_INTERPRETATION = {
    "rel_object": "The child is interacting with a relevant object",
    "eng_rel": "The child is socially or contextually engaged",
    "high_time": "The child spent a sustained amount of time",
    "full_involv": "The child was fully involved",
    "part_involv": "The child was partially involved",
}

print("\nAtom → Interpretation")
for atom, meaning in ATOM_INTERPRETATION.items():
    print(f"- {atom}: {meaning}")

final_answer = answer_sets[-1][0] if answer_sets[-1] else []

sentences = []

if "rel_object" in final_symbols and final_symbols["rel_object"]:
    sentences.append("interacted with a relevant object (ball)")
if "eng_rel" in final_symbols and final_symbols["eng_rel"]:
    sentences.append("remained engaged with the activity context")
if "high_time" in final_symbols and final_symbols["high_time"]:
    sentences.append("spent a sustained amount of time in the activity")

summary = ", ".join(sentences)

print("\nInterpretation:")
print(
    "Based on the interview text, the system inferred that the child "
    + summary
    + " (football field)."
)

if "full_involv" in final_answer:
    print("This pattern is consistent with full involvement.")
elif "part_involv" in final_answer:
    print("This pattern is consistent with partial involvement.")
else:
    print("This pattern suggests limited or uncertain involvement.")
