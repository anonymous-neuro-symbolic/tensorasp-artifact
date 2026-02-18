# Attention-Aware Tensor-Annotated ASP (AAT-ASP)

This repository contains the experimental code accompanying the paper:

**“Attention-Aware Tensor-Annotated Answer Set Programming”**

---

# Attention-Aware Tensor-Annotated ASP (AAT-ASP)

This project implements **Attention-Aware Tensor-Annotated ASP (AAT-ASP)**, an extension of classical Answer Set Programming that enables reasoning over tensor-valued evidence produced by neural models.

## Theoretical Framework

AAT-ASP preserves classical stable model semantics while introducing tensor annotations and attention-based aggregation mechanisms.

### Core Components

1. **Tensor-annotated predicates**  
   Predicates associated with tensor-valued evidence (e.g., embeddings from transformers or CNNs).

2. **Annotated atoms**  
   Projection produces symbolic atoms together with tensor scores.

3. **Projection functions**  
   Mappings πₚ: ℝⁿ → {0,1} that convert tensor evidence into ASP atoms.

4. **Attention aggregation**  
   Tensor evidence may be aggregated across related atoms before projection.

5. **Stable model semantics**  
   The symbolic layer remains unchanged and follows classical ASP semantics.

6. **Meta-level learning**  
   Threshold parameters are updated between reasoning episodes without altering reduct computation.

---

## Requirements

- Conda (recommended) or Python ≥ 3.10
- NVIDIA GPU (optional, CPU supported)
- clingo ≥ 5.6
- PyTorch
- sentence-transformers
- ultralytics (YOLOv8)

---

## Setup (Conda)

```bash
conda env create -f environment.yml
conda activate tensorasp
```

---

## Experiment 1: Text-based Involvement Reasoning

Located in `/experiment1`.

```bash
cd experiment1/
python main7.py
```

### Output
- Answer sets per epoch
- Threshold updates
- Stable model convergence logs
- Natural language interpretation

This experiment processes interview transcripts (text input) and projects tensor embeddings into symbolic predicates for involvement reasoning.

---

## Experiment 2: Vision-based Person Reasoning (YOLO)

Located in `/experiment2`.

```bash
cd experiment2/
python experiment2_yolo_person.py
```

### Output
- YOLO detections
- Projected ASP facts
- Answer sets
- Visual overlays

This experiment separates perception (YOLO) from symbolic reasoning (ASP).

---

## Additional Benchmarks (Experiment 2)

Experiment 2 also includes additional evaluation settings:

### 1. Controlled Ablation Study
Evaluates robustness under confidence perturbations:
- Unary Projection (UP)
- Attention-Dependent Projection (ADP)

Metrics:
- Average distinct answer sets
- Stability rate under noise

### 2. Logical Consistency Benchmark
Evaluates whether projection design affects stable model existence under symbolic constraints:
- Measures frequency of inconsistent runs
- Compares projection strategies

These benchmarks demonstrate how projection mechanisms influence symbolic outcomes while preserving classical ASP semantics.

---

## Significance

- Maintains formal stable model semantics
- Separates tensor projection from reasoning
- Supports text and vision inputs
- Demonstrates projection-dependent symbolic effects

---

## Reproducibility Notes

- ASP programs are in `/asp/`
- Neural modules are modular and replaceable
- Environment file contains dependencies
- Deterministic for fixed random seeds

---

## Contact

For anonymized review purposes, no identifying metadata is included.
