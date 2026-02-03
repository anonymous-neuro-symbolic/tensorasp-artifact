import os
import math
import numpy as np
import clingo


import textwrap

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





# ============================================================
log("Graphical Explanation Interface")

# --- Atom → text span mapping (explanatory, not semantic) ---
TEXT_HIGHLIGHTS = {
    "rel_object": ["ball"],
    "eng_rel": ["playing", "lost focus"],
    "high_time": ["football field"]
}

# --- Prepare highlighted text ---
def highlight_text(text, active_atoms):
    """
    Returns a list of (substring, color) tuples for rendering.
    """
    highlights = []
    remaining = text

    color_map = {
        "rel_object": "tab:blue",
        "eng_rel": "tab:green",
        "high_time": "tab:orange"
    }

    for atom in active_atoms:
        for phrase in TEXT_HIGHLIGHTS.get(atom, []):
            if phrase in remaining:
                before, after = remaining.split(phrase, 1)
                highlights.append((before, "black"))
                highlights.append((phrase, color_map.get(atom, "red")))
                remaining = after

    highlights.append((remaining, "black"))
    return highlights


# --- Build final symbols and explanation ---
active_atoms = [k for k, v in final_symbols.items() if v]

highlighted_chunks = highlight_text(interview_text, active_atoms)

# --- Create UI figure ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax_text, ax_summary = axes

# -------- Left panel: Interview text with highlights --------
ax_text.axis("off")
ax_text.set_title("Interview Text (Highlighted Evidence)", fontsize=25, fontweight="bold")

y = 1.0
for chunk, color in highlighted_chunks:
    wrapped = textwrap.fill(chunk, 60)
    ax_text.text(
        0.01,
        y,
        wrapped,
        fontsize=23,
        va="top",
        color=color,
        transform=ax_text.transAxes
    )
    y -= 0.06 * (wrapped.count("\n") + 1)

# -------- Right panel: Answer set + interpretation --------
ax_summary.axis("off")
ax_summary.set_title("Symbolic Outcome & Interpretation", fontsize=25, fontweight="bold")

summary_lines = []

summary_lines.append("Final Answer Set:")
summary_lines.append(", ".join(final_answer) if final_answer else "∅")
summary_lines.append("")

summary_lines.append("Active Projections:")
for atom in active_atoms:
    summary_lines.append(f"• {atom}: {ATOM_INTERPRETATION.get(atom, '')}")

summary_lines.append("")
summary_lines.append("Natural Language Interpretation:")
summary_lines.append(
    "Based on the interview text, the system inferred that the child "
    + summary
    + " (football field)."
)

if "full_involv" in final_answer:
    summary_lines.append("→ This pattern is consistent with full involvement.")
elif "part_involv" in final_answer:
    summary_lines.append("→ This pattern is consistent with partial involvement.")
else:
    summary_lines.append("→ This pattern suggests limited or uncertain involvement.")

ax_summary.text(
    0.01,
    0.98,
    "\n".join(summary_lines),
    fontsize=23,
    va="top",
    transform=ax_summary.transAxes
)

plt.tight_layout()
plt.show()


import textwrap

def show_interpretation_ui(
    interview_text: str,
    final_symbols: Dict[str, bool],
    final_answer: List[str]
):
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor("white")

    # ---------- Layout ----------
    ax_text = fig.add_axes([0.05, 0.55, 0.90, 0.40])
    ax_logic = fig.add_axes([0.05, 0.30, 0.90, 0.18])
    ax_interp = fig.add_axes([0.05, 0.05, 0.90, 0.20])

    for ax in [ax_text, ax_logic, ax_interp]:
        ax.axis("off")

    # ---------- Interview text ----------
    wrapped_text = textwrap.fill(interview_text, width=90)

    ax_text.text(
        0, 1,
        "Interview Text",
        fontsize=22,
        fontweight="bold",
        va="top"
    )

    ax_text.text(
        0, 0.88,
        wrapped_text,
        fontsize=22,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#cccccc")
    )

    # ---------- Symbolic outcome ----------
    ax_logic.text(
        0, 1,
        "Symbolic Interpretation (Answer Set)",
        fontsize=22,
        fontweight="bold",
        va="top"
    )

    logic_text = ", ".join(final_answer) if final_answer else "No atoms derived"

    ax_logic.text(
        0, 0.65,
        f"Derived atoms: {logic_text}",
        fontsize=22,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="#eef3ff", ec="#99aacc")
    )

    # ---------- Natural-language interpretation ----------
    sentences = []
    if final_symbols.get("rel_object"):
        sentences.append("interacted with a relevant object (ball)")
    if final_symbols.get("eng_rel"):
        sentences.append("remained engaged with the activity")
    if final_symbols.get("high_time"):
        sentences.append("spent a sustained amount of time")

    summary = ", ".join(sentences)

    nl_text = (
        "Based on the interview text, the system inferred that the child "
        + summary
        + " in the context of the football field.\n\n"
    )

    if "full_involv" in final_answer:
        nl_text += "This pattern is consistent with full involvement."
    elif "part_involv" in final_answer:
        nl_text += "This pattern is consistent with partial involvement."
    else:
        nl_text += "This pattern suggests limited or uncertain involvement."

    ax_interp.text(
        0, 1,
        "Natural-Language Summary",
        fontsize=22,
        fontweight="bold",
        va="top"
    )

    ax_interp.text(
        0, 0.75,
        textwrap.fill(nl_text, width=90),
        fontsize=22,
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f0fff4", ec="#88ccaa")
    )

    plt.show()
