"""
Sensitivity analysis for NF-kB pathway (Research Project proto)

Implements loading a BioRECIPE-formatted Excel file, building
discrete update rules, running stochastic asynchronous simulations,
conducting sensitivity analysis on a target node, plotting results,
and exporting scores to Excel.

Requirements: pandas, openpyxl, numpy, matplotlib
"""
from copy import deepcopy
import random
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_biorecipe_model(filepath: str) -> dict:
    """
    Read BioRECIPE .xlsx into a Python dict of elements.
    Each element stores its regulators and update rules.

    Expected columns: Element Name, Positive Regulators, Negative Regulators,
    Initial Value, Max Value
    Positive/Negative regulator columns are semicolon-separated strings.
    Returns: model dict keyed by element name with fields:
      - positives: list[str]
      - negatives: list[str]
      - initial: int (0/1)
      - max: int
    """
    df = pd.read_excel(filepath, engine="openpyxl")
    required = [
        "Element Name",
        "Positive Regulators",
        "Negative Regulators",
        "Initial Value",
        "Max Value",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    model = {}
    for _, row in df.iterrows():
        name = str(row["Element Name"]).strip()
        pos = (
            str(row["Positive Regulators"]).strip()
            if not pd.isna(row["Positive Regulators"]) else ""
        )
        neg = (
            str(row["Negative Regulators"]).strip()
            if not pd.isna(row["Negative Regulators"]) else ""
        )
        positives = [p.strip() for p in pos.split(";") if p.strip()]
        negatives = [n.strip() for n in neg.split(";") if n.strip()]
        initial = int(row["Initial Value"]) if not pd.isna(row["Initial Value"]) else 0
        maxv = int(row["Max Value"]) if not pd.isna(row["Max Value"]) else 1
        model[name] = {
            "positives": positives,
            "negatives": negatives,
            "initial": 1 if initial else 0,
            "max": maxv,
            # placeholders; will be filled by build_update_rules
            "update_fn": None,
            "locked": False,
            "locked_value": None,
        }
    return model


def build_update_function(element_name: str, model: dict) -> Callable[[Dict[str, int]], int]:
    """
    Build update function for a single element using discrete majority logic:
      - Count active positive regulators
      - Count active negative regulators
      - If pos > neg: return 1
      - If neg > pos: return 0
      - If equal: return current value

    The returned function signature: f(current_state: dict) -> next_value (0/1)
    """

    positives = model[element_name]["positives"]
    negatives = model[element_name]["negatives"]

    def update_fn(current_state: Dict[str, int]) -> int:
        # Respect locking
        if model[element_name].get("locked"):
            return int(model[element_name].get("locked_value", 0))

        pos_count = 0
        neg_count = 0
        for p in positives:
            if current_state.get(p, 0):
                pos_count += 1
        for n in negatives:
            if current_state.get(n, 0):
                neg_count += 1

        if pos_count > neg_count:
            return 1
        if neg_count > pos_count:
            return 0
        # equal -> stay the same
        return int(current_state.get(element_name, 0))

    return update_fn


def attach_update_functions(model: dict):
    for name in list(model.keys()):
        model[name]["update_fn"] = build_update_function(name, model)


def run_simulation(model: dict, steps: int = 100, runs: int = 50) -> List[List[Dict[str, int]]]:
    """
    Run stochastic asynchronous simulation.

    Each run:
      - Initialize state from element['initial']
      - For `steps` iterations, pick a random element, apply its update function,
        and record the full state vector after each step.

    Returns: list of runs, each run is list of state snapshots (dict)
    """
    attach_update_functions(model)
    elements = list(model.keys())
    all_runs = []
    for r in range(runs):
        state = {name: int(model[name]["initial"]) for name in elements}
        traj = [deepcopy(state)]
        for s in range(steps):
            el = random.choice(elements)
            fn = model[el]["update_fn"]
            new_val = fn(state)
            state[el] = int(new_val)
            traj.append(deepcopy(state))
        all_runs.append(traj)
    return all_runs


def _final_mean_of_target(sim_runs: List[List[Dict[str, int]]], target: str) -> float:
    finals = []
    for run in sim_runs:
        finals.append(run[-1].get(target, 0))
    return float(np.mean(finals))


def sensitivity_analysis(model: dict, target_element: str, steps: int = 100, runs: int = 50) -> dict:
    """
    For each element E in the network:
      1. Knock it OUT (force value = 0), run simulation
      2. Knock it IN  (force value = 1), run simulation
      3. Compare target_element's steady-state vs baseline
      4. Sensitivity score = max(|KO_mean - baseline_mean|, |KI_mean - baseline_mean|)

    Returns: dict of {element_name: sensitivity_score}
    """
    # baseline
    baseline_runs = run_simulation(deepcopy(model), steps=steps, runs=runs)
    baseline_mean = _final_mean_of_target(baseline_runs, target_element)

    scores = {}
    for element in model.keys():
        # knockout
        mko = deepcopy(model)
        mko[element]["locked"] = True
        mko[element]["locked_value"] = 0
        ko_runs = run_simulation(mko, steps=steps, runs=runs)
        ko_mean = _final_mean_of_target(ko_runs, target_element)

        # knockin
        mki = deepcopy(model)
        mki[element]["locked"] = True
        mki[element]["locked_value"] = 1
        ki_runs = run_simulation(mki, steps=steps, runs=runs)
        ki_mean = _final_mean_of_target(ki_runs, target_element)

        scores[element] = float(max(abs(ko_mean - baseline_mean), abs(ki_mean - baseline_mean)))

    return scores


def plot_sensitivity_ranking(scores: dict, title: str = "Sensitivity Ranking"):
    """
    Bar chart: x = element names, y = sensitivity score
    Highlight top 3 nodes in a different color.
    """
    if not scores:
        print("No scores to plot")
        return
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    names = [n for n, _ in items]
    vals = [v for _, v in items]

    colors = ["C0"] * len(names)
    for i in range(min(3, len(names))):
        colors[i] = "C3"

    plt.figure(figsize=(max(6, len(names) * 0.4), 4))
    bars = plt.bar(range(len(names)), vals, color=colors)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.ylabel("Sensitivity score")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def export_to_biorecipe_results(scores: dict, output_path: str):
    """
    Save results in a simple BioRECIPE-compatible Excel format.
    Columns: Element Name, Sensitivity Score
    """
    df = pd.DataFrame(
        [(k, v) for k, v in scores.items()], columns=["Element Name", "Sensitivity Score"]
    )
    df.to_excel(output_path, index=False, engine="openpyxl")


def generate_test_model_excel(path: str = "test_model.xlsx"):
    """
    Create the tiny fake pathway from the instructions as an Excel file.
    Element Name | Positive Regulators | Negative Regulators | Initial Value | Max Value
    """
    rows = [
        {"Element Name": "A", "Positive Regulators": "B;1", "Negative Regulators": "", "Initial Value": 1, "Max Value": 1},
        {"Element Name": "B", "Positive Regulators": "", "Negative Regulators": "A", "Initial Value": 0, "Max Value": 1},
        {"Element Name": "NFkB", "Positive Regulators": "A", "Negative Regulators": "B", "Initial Value": 0, "Max Value": 1},
    ]
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False, engine="openpyxl")
    print(f"Wrote test model to {path}")


if __name__ == "__main__":
    # Quick smoke-run: create test model and run sensitivity on NFkB
    try:
        generate_test_model_excel("test_model.xlsx")
        model = load_biorecipe_model("test_model.xlsx")
        scores = sensitivity_analysis(model, target_element="NFkB", steps=50, runs=20)
        print("Top sensitivity scores:", sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5])
        plot_sensitivity_ranking(scores, title="NF-kB Pathway Sensitivity (test)")
        export_to_biorecipe_results(scores, "results_test.xlsx")
    except Exception as e:
        print("Quick test failed:", e)
