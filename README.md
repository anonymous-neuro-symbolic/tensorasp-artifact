# Tensorized ASP: Neuro-Symbolic Reasoning with Geometric Evidence

This repository contains the experimental code accompanying the paper:

"Attention-Aware Geometric Answer Set Programming"

## Requirements

- Conda (recommended) or Python ≥ 3.10
- NVIDIA GPU (optional, CPU supported)
- clingo ≥ 5.6
- PyTorch
- sentence-transformers
- ultralytics (YOLOv8)

## Setup (Conda)

```bash
conda env create -f environment.yml
conda activate tensorasp
```

## Experiment 1: Text-based Involvement Reasoning

To test the experiment in the context of Occupational Therapy, go to the folder `/experiment1` and run the main Python file:

```bash
cd experiment1/
python main7.py
```

**Expected output:**
- Answer sets per epoch
- Learning curve
- Natural language interpretation
- Visualization window

## Experiment 2: Vision-based Person Reasoning (YOLO)

```bash
cd experiments/exp2_yolo_person
python experiment2_yolo_person.py
```

**Expected output:**
- YOLO detections
- ASP answer sets
- Visual overlays corresponding to symbolic conclusions
