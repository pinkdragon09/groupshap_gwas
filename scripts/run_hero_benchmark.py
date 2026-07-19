
import os
import sys
import pandas as pd
from dataclasses import dataclass, field

# Add scripts/ to path
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from run_benchmark import BenchmarkConfig, _run_and_save

def run_targeted_scenario3():
    out_path = "outputs/benchmark/scenario3_interaction_recovery_v2.tsv"
    if os.path.exists(out_path):
        os.remove(out_path)
    
    # We use a higher number of samples and reps to get stable, "Hero-quality" results
    # Scenario 3 is pairwise epistasis
    for rho in [0.0, 0.2, 0.4, 0.6, 0.8]:
        cfg = BenchmarkConfig(
            scenario=3, 
            n=1000,   # Increased sample size
            rho=rho, 
            h2=0.5, 
            n_reps=50, # Sufficient for stable means
        )
        # Note: I'm relying on the existing _run_and_save which calls _run_one_rep.
        # _run_one_rep uses n_eval=10 and mc_samples=20 for grouped_shapley.
        # This might be enough for a clear signal in Scenario 3.
        _run_and_save(cfg, out_path, f"Scenario3 rho={rho}")

if __name__ == "__main__":
    run_targeted_scenario3()
