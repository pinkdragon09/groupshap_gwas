
import os
import sys
import pandas as pd
from dataclasses import dataclass, field

# Add scripts/ to path
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from run_benchmark import BenchmarkConfig, _run_and_save

def run_hero_benchmark_v2():
    out_path = "outputs/benchmark/hero_scenario1_jaccard.tsv"
    if os.path.exists(out_path):
        os.remove(out_path)
    
    # Scenario 1: Sparse Additive
    for rho in [0.0, 0.3, 0.6, 0.9]:
        cfg = BenchmarkConfig(
            scenario=1, 
            n=1000, 
            p=200,
            B=20,     # 20 blocks of 10 SNPs
            rho=rho, 
            h2=0.5, 
            n_reps=30, # Reduced reps for speed
        )
        _run_and_save(cfg, out_path, f"Hero Scenario1 rho={rho}")

if __name__ == "__main__":
    run_hero_benchmark_v2()
