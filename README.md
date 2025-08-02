# DeepEWA: Approximating Learning Dynamics in 2 × 2 Games with Neural Networks

[![Julia](https://img.shields.io/badge/julia-v1.11+-blue.svg)](https://julialang.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This project implements a neural network approach to predict convergence characteristics of Experience-Weighted Attraction (EWA) learning dynamics in 2×2 games, based on Pangallo et al. (2022).

## Overview

Experience Weighted Attraction generalizes various learning algorithms in game theory:
**Attraction updating function**:
$$
Q_{i}^{\mu}(t) = \frac{(1-\alpha) N(t-1) Q_{i}^{\mu}(t-1)}{N(t)} + \frac{\left[ \delta + (1-\delta) \mathbb{I}(s_i^\mu,s^{-\mu}(t)) \right] \Pi^\mu(s_i^\mu, s^{-\mu}(t))}{N(t)}
$$

**Action selection**:
$$
\sigma^{\mu}(t)=\frac{e^{\beta Q_1^R (t)}}{e^{\beta Q_1^R (t)} + e^{\beta Q_2^R (t)}}
$$

Special cases include:
- **Best Response Dynamics** (α=1, β=∞, δ=1)
- **Fictitious Play** (α=0, β=∞, δ=1, κ=0) 
- **Reinforcement Learning** (δ=0)
- **Replicator Dynamics** (β→0, α=0, δ=1)
- **Logit Dynamics** (α=1, δ=1, κ=1)

The project trains a deep neural network to classify convergence outcomes into four categories:
1. **Limit cycles/chaos** 
2. **Mixed strategy fixed points**
3. **Pure strategy fixed points** 
4. **Pure Nash equilibria**

## Project Structure

```
DeepEWA/
├── ProbsEWA.jl           # Core EWA algorithm and convergence classification
├── DataGen.jl            # Data generation for neural network training
├── DeepEWA.ipynb         # Main Jupyter notebook with analysis
├── dependencies.jl       # Package dependencies
├── data/                 # Training and test datasets
│   ├── train_data.csv
│   └── test_data.csv
├── images/               # Generated plots and visualizations
├── notebooks EWA/        # EWA algorithm testing notebooks
├── notebooks NN/         # Neural network experiments
├── versions EWA/         # Different EWA implementations
│   ├── DeepEWA_game.jl   # Main training script
│   ├── DeepEWA_testing.jl # Model evaluation
│   ├── EWA_funs&structs.jl # EWA functions and structures
│   ├── FastEWA.jl        # Optimized EWA implementation
│   └── models/           # Saved neural network models
├── literature/           # Academic papers and references
└── paper/                # Project documentation
```

## Installation

1. **Install Julia** (v1.11 or higher): [https://julialang.org/downloads/](https://julialang.org/downloads/)


### Quick start

Run the following notebook:
```bash
jupyter notebook DeepEWA.ipynb
```

To analyze custom 2×2 games, modify the payoff matrices:
```julia
# Example: Custom coordination game
custom_game = [[5 1; 1 4], [5 1; 1 4]]
```

## EWA Parameters

- **α** (memory loss): [0, 1] 
- **κ** (discount rate): [0, 1]  
- **δ** (foregone payoffs): [0, 1] 
- **β** (stochasticity): [0, ∞] 

## Model Architecture

The neural network uses:
- **Input**: 12 features (4 EWA parameters + 8 payoff matrix entries)
- **Hidden layers**: 3 layers with 32 ReLU units each
- **Output**: 4-class softmax for convergence classification
- **Optimizer**: AdaGrad with learning rate adaptation
- **Training**: 2500 epochs with batch size 64

## Results and Visualization

Example accuracy results:
- **Overall accuracy**: ~85-95% depending on game type

## References

- Pangallo, M., et al. (2024). "Learning dynamics prediction in game theory"
- Camerer, C., & Ho, T. H. (1999). "Experience-weighted attraction learning in normal form games"


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

**Keywords**: Game Theory, Machine Learning, Experience-Weighted Attraction, Neural Networks, Convergence Analysis, Julia

