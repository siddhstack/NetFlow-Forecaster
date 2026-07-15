# AES-oriented reframing for NetFlow-Forecaster

## Proposed contribution

This project is reframed as an autonomous, self-adaptive software architecture for long-term engineering time-series reliability rather than as a single-model forecasting study. The software loop continuously profiles drift, evaluates candidate configurations, and deploys the strongest reliable configuration without manual intervention.

## Highlights

- Programmatic self-improving loop mitigates ML model concept drift.
- Multi-Armed Bandit meta-policy automates configuration selection.
- Spike-weighted loss optimization captures rare engineering telemetry faults.

## Mathematical formulation

The candidate selection policy uses a UCB1 rule:

$$
\text{UCB1}(c) = \mu_c + \sqrt{\frac{2\log T}{n_c}}
$$

where $\mu_c$ is the historical quality score for candidate $c$, $n_c$ is the number of times it has been selected, and $T$ is the total number of tournament runs.

The spike-weighted training objective is expressed as:

$$
\mathcal{L}_{\text{total}} = \mathbf{e}^{T} (\mathbf{W}_{\text{feature}} \odot \mathbf{W}_{\text{spike}}) \mathbf{e}
$$

where $\mathbf{e}$ is the raw error vector, $\mathbf{W}_{\text{feature}}$ encodes feature importance, and $\mathbf{W}_{\text{spike}}$ scales the penalty when ground-truth spikes are present.

## Experimental plan

1. Static Pipeline: train a single model once and evaluate it over drifting telemetry windows.
2. Greedy Loop: disable exploration in the meta-policy and observe stagnation in local optima.
3. Full NetFlow Loop: enable UCB1 exploration and spike-weighted loss to maintain stable reliability.

## Software availability

Repository: https://github.com/your-org/netflow-forecaster

Reproducibility commands:

```bash
python ml/self_improve.py --data ml/telemetry.csv --output-dir runs/replicate
python ml/ablation_selection.py --data ml/telemetry.csv --output-dir runs/ablation
```
