import os
import clingo
import numpy as np
from ultralytics import YOLO

# ============================
# Configuration
# ============================

IMAGE_DIR = "images"
ASP_FILE = "asp/person_reasoning.lp"

CONF_THRESHOLD = 0.5          # Unary threshold
AGG_THRESHOLD = 0.65          # Collective threshold

NOISE_STD = 0.08              # Noise level
N_TRIALS = 30                 # Trials per image

# ============================
# Load YOLOv8
# ============================

model = YOLO("yolov8n.pt")

# ============================
# ASP Reasoning
# ============================

def run_clingo(facts):
    ctl = clingo.Control()
    ctl.load(ASP_FILE)
    ctl.add("base", [], "\n".join(facts))
    ctl.ground([("base", [])])

    models = []

    def on_model(model):
        models.append({str(sym) for sym in model.symbols(shown=True)})

    ctl.solve(on_model=on_model)
    return models[0] if models else set()


# ============================
# Projection Modes
# ============================

def unary_projection(confidences):
    """
    Unary projection: per-detection thresholding.
    """
    detected = [c for c in confidences if c >= CONF_THRESHOLD]
    facts = []

    if len(detected) >= 1:
        facts.append("person_detected.")
    if len(detected) >= 2:
        facts.append("multiple_people.")

    return facts


def attention_projection(confidences):
    """
    Attention-dependent projection using a smooth collective score:
    collective_score = 0.5 * mean + 0.5 * min
    """
    facts = []

    if len(confidences) == 0:
        return facts

    mean_conf = np.mean(confidences)
    min_conf = np.min(confidences)

    # Person exists if overall signal exists
    if mean_conf >= 0.4:
        facts.append("person_detected.")

    # Group requires relational collective strength
    if len(confidences) >= 2:
        collective_score = 0.5 * mean_conf + 0.5 * min_conf
        if collective_score >= AGG_THRESHOLD:
            facts.append("multiple_people.")

    return facts


# ============================
# Extract Final Scene Label
# ============================

def extract_scene_label(model_atoms):
    if "crowded" in model_atoms:
        return "crowded"
    if "group_scene" in model_atoms:
        return "group_scene"
    if "single_person_scene" in model_atoms:
        return "single_person_scene"
    if "empty_scene" in model_atoms:
        return "empty_scene"
    return "unknown"


# ============================
# Controlled Ablation Study
# ============================

print("\n==================== CONTROLLED ABLATION STUDY ====================\n")

summary_variability = {"UP": [], "ADP": []}
summary_stability = {"UP": [], "ADP": []}

for img_name in sorted(os.listdir(IMAGE_DIR)):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    print(f"\nImage: {img_name}")

    results = model(img_path)[0]

    raw_confidences = [
        float(box.conf[0])
        for box in results.boxes
        if int(box.cls[0]) == 0
    ]

    print("Original confidences:", raw_confidences)

    for mode in ["UP", "ADP"]:
        labels = []

        for _ in range(N_TRIALS):
            noisy_conf = [
                max(0.0, min(1.0, c + np.random.normal(0, NOISE_STD)))
                for c in raw_confidences
            ]

            if mode == "UP":
                facts = unary_projection(noisy_conf)
            else:
                facts = attention_projection(noisy_conf)

            model_atoms = run_clingo(facts)
            label = extract_scene_label(model_atoms)
            labels.append(label)

        unique_labels = set(labels)
        variability = len(unique_labels)

        # Stability = proportion of dominant label
        most_common = max(set(labels), key=labels.count)
        stability_rate = labels.count(most_common) / N_TRIALS

        summary_variability[mode].append(variability)
        summary_stability[mode].append(stability_rate)

        print(f"\nMode: {mode}")
        print("Distinct labels:", variability)
        print("Stability rate:", round(stability_rate, 3))

# ============================
# Summary Statistics
# ============================

print("\n==================== ABLATION SUMMARY ====================")

for mode in ["UP", "ADP"]:
    avg_var = np.mean(summary_variability[mode])
    avg_stab = np.mean(summary_stability[mode])

    print(f"\nProjection Mode: {mode}")
    print("Average distinct labels:", round(avg_var, 3))
    print("Average stability rate:", round(avg_stab, 3))

print("\nInterpretation:")
print("- UP uses independent per-detection thresholds.")
print("- ADP uses relational collective strength (mean + min).")
print("- Lower variability and higher stability indicate stronger robustness.")
