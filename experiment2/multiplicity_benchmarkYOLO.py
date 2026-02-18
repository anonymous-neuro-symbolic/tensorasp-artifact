import clingo
import numpy as np

# Ensure this path is correct for your environment
ASP_FILE = "asp/multiplicity_benchmark_YOLO.lp" 
CONF_THRESHOLD = 0.5
NOISE_STD = 0.03
N_TRIALS = 20
BASE_CONF = [0.55, 0.54]

def run_with_signals(sig1, sig2):
    ctl = clingo.Control(["0"])
    ctl.load(ASP_FILE)
    ctl.ground([("base", [])])

    sym1 = clingo.Function("multiple_people_detected")
    sym2 = clingo.Function("single_person_detected")

    ctl.assign_external(sym1, sig1)
    ctl.assign_external(sym2, sig2)

    models = []

    def on_model(model):
        models.append({str(sym) for sym in model.symbols(shown=True)})

    ctl.solve(on_model=on_model)

    # Important: release externals after solving
    ctl.release_external(sym1)
    ctl.release_external(sym2)

    return models


# def run_with_signals(sig1, sig2):
#     # ["0"] tells Clingo to find ALL stable models
#     ctl = clingo.Control(["0"]) 
#     ctl.load(ASP_FILE)
#     ctl.ground([("base", [])])

#     # Convert signals to Clingo TruthValues
#     val1 = clingo.TruthValue.True_ if sig1 else clingo.TruthValue.False_
#     val2 = clingo.TruthValue.True_ if sig2 else clingo.TruthValue.False_
    
#     ctl.assign_external(clingo.Function("signal1"), val1)
#     ctl.assign_external(clingo.Function("signal2"), val2)

#     models = []
#     def on_model(model):
#         models.append({str(sym) for sym in model.symbols(shown=True)})

#     ctl.solve(on_model=on_model)
#     return models

def unary_projection(conf):
    return conf[0] >= CONF_THRESHOLD, conf[1] >= CONF_THRESHOLD

def attention_projection(conf):
    idx = np.argmax(conf)
    return (idx == 0), (idx == 1)

print("\n================ MULTIPLICITY BENCHMARK ================")
results = {}

for mode in ["UP", "ADP"]:
    print(f"\nProjection Mode: {mode}")
    multiplicities = []
    for t in range(N_TRIALS):
        # Generate noisy confidence
        noisy_conf = [c + np.random.normal(0, NOISE_STD) for c in BASE_CONF]
        
        if mode == "UP":
            s1, s2 = unary_projection(noisy_conf)
        else:
            s1, s2 = attention_projection(noisy_conf)

        found_models = run_with_signals(s1, s2)
        m_count = len(found_models)
        multiplicities.append(m_count)
        
        print(f"Trial {t+1:02d}: conf={np.round(noisy_conf,3)} signals=({int(s1)},{int(s2)}) models={m_count}")

    avg = np.mean(multiplicities)
    results[mode] = avg
    print(f"Average stable models: {avg}")

print("\n================ SUMMARY ================")
print(f"UP: Average stable models = {results['UP']}")
print(f"ADP: Average stable models = {results['ADP']}")