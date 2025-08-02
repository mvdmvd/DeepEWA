# DeepEWA: Approximating Learning Dynamics in 2 × 2 Games with Neural Networks

[![Julia](https://img.shields.io/badge/julia-v1.11+-blue.svg)](https://julialang.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This project implements a neural network approach to predict convergence characteristics of Experience-Weighted Attraction (EWA) learning dynamics in 2×2 games, based on Pangallo et al. (2022). This project was the final assignment for the course ''Computational Game Theory'', taught by Prof. Davide Grossi, and was graded 9.7/10.

## Overview

Experience Weighted Attraction generalizes various learning algorithms in game theory:

**Attraction updating function**:

```math
Q_{i}^{a}(t) = \frac{(1-\alpha) \mathcal{N}(t-1) Q_{i}^{a}(t-1)}{\mathcal{N}(t)} + \frac{\left[ \delta + (1-\delta) \mathbb{I}(s_i^a,s_{-i}(t)) \right] \Pi_i(s_i^a, s_{-i}(t))}{\mathcal{N}(t)} 
```

**Action selection**:

```math
\mu_i(t)=\frac{e^{\beta Q_1 (t)}}{e^{\beta Q_1^a (t)} + e^{\beta Q_2^b (t)}}
```

Special cases include:
- **Best Response Dynamics** (α=1, β=∞, δ=1)
- **Fictitious Play** (α=0, β=∞, δ=1, κ=0) 
- **Reinforcement Learning** (δ=0)
- **Replicator Dynamics** (β→0, α=0, δ=1)
- **Logit Dynamics** (α=1, δ=1, κ=1)

We train a deep neural network to classify convergence outcomes of learning dynamics on 2×2 games into four categories:
1. **Limit cycles/chaos** 
2. **Mixed strategy fixed points**
3. **Pure strategy fixed points** 
4. **Pure Nash equilibria**

## Project Structure

```
DeepEWA/
├── ProbsEWA.jl           # Core EWA algorithm and convergence classification
├── DataGen.jl            # Data generation for neural network training (also included in notebook)
├── DeepEWA.ipynb         # Main Jupyter notebook with analysis
├── dependencies.jl       
├── data/                 # Training and test datasets
│   ├── train_data.csv
│   └── test_data.csv
├── images/               # Generated plots
├── notebooks EWA/        # Testing notebooks
├── notebooks NN/         # NN experiments
├── versions EWA/         # Older versions
├── literature/           # References
└── paper/                # Submitted assignment
```

## Installation

1. **Install Julia** (v1.11 or higher): [https://julialang.org/downloads/](https://julialang.org/downloads/)


### Quick start

Run the following notebook:
```bash
jupyter notebook DeepEWA.ipynb
```

For custom 2×2 games, modify the payoff matrices:
```julia
# Example coordination game: 
custom_game = [[7 2; 2 3], [7 2; 2 3]]
```

## EWA Parameters

- **α** (memory loss): [0, 1] 
- **κ** (discount rate): [0, 1]  
- **δ** (foregone payoffs): [0, 1] 
- **β** (temperature): [0, ∞] 

## Model Architecture
- **Input**: 12 features (4 EWA parameters + 8 payoff matrix entries)
- **Hidden layers**: 3 layers with 32 ReLU units each
- **Output**: 4-class softmax for convergence classification
- **Optimizer**: AdaGrad with learning rate adaptation
- **Training**: 2500 epochs with batch size 64

Example accuracy results:
- **Overall accuracy**: ~85-95% depending on game type

## References

- Pangallo, M., et al. (2024). "Learning dynamics prediction in game theory"
- Camerer, C., & Ho, T. H. (1999). "Experience-weighted attraction learning in normal form games"


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---


