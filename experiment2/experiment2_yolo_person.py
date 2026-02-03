import os
import clingo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ultralytics import YOLO

# ============================
# Configuration
# ============================

IMAGE_DIR = "images"
ASP_FILE = "asp/person_reasoning.lp"
CONF_THRESHOLD = 0.5

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
    return models

# ============================
# Visualization
# ============================

def visualize_results(image_path, boxes, asp_atoms):
    img = plt.imread(image_path)
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.imshow(img)
    ax.axis("off")

    # Decide semantic mode from ASP
    if "crowded_scene" in asp_atoms:
        color = "red"
        label = "Crowded Scene"
    elif "multiple_people_scene" in asp_atoms:
        color = "orange"
        label = "Multiple People"
    elif "single_person_scene" in asp_atoms:
        color = "green"
        label = "Single Person"
    else:
        color = "gray"
        label = "Unclassified"

    # Draw boxes
    for (x1, y1, x2, y2) in boxes:
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none"
        )
        ax.add_patch(rect)

    ax.set_title(f"ASP Interpretation: {label}", fontsize=14)
    plt.show()

# ============================
# Main Experiment Loop
# ============================

print("\n==================== STARTING EXPERIMENT 2: YOLO PERSON ====================\n")

for img_name in sorted(os.listdir(IMAGE_DIR)):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    print(f"Image: {img_name}")

    # --- YOLO detection ---
    results = model(img_path)[0]
    boxes = []
    confidences = []

    for box in results.boxes:
        if int(box.cls[0]) == 0:  # person
            conf = float(box.conf[0])
            if conf >= CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
                confidences.append(conf)

    print("YOLO confidences:", confidences)

    # --- Projection to ASP facts ---
    facts = []

    if len(boxes) >= 1:
        facts.append("person_detected.")
    if len(boxes) >= 2:
        facts.append("multiple_people.")
    if len(boxes) >= 4:
        facts.append("crowded.")

    print("ASP facts:", facts)

    # --- ASP reasoning ---
    models = run_clingo(facts)
    asp_atoms = models[0] if models else set()

    print("Answer sets:", models)

    # --- Visualization ---
    visualize_results(img_path, boxes, asp_atoms)

print("\n==================== FINAL INTERPRETATION ====================")
print("This experiment demonstrates:")
print("- YOLO performs low-level perception")
print("- ASP reasons over symbolic projections")
print("- Answer sets control semantic visualization")
print("- No neural retraining is required")
