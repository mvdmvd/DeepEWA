# DeepEWA

This project is based on Pangallo et al. (2021).

We train a NN to predict the convergence characteristics of learning dynamics on 2x2 games.

The code is written in [Julia](https://github.com/JuliaLang/julia)

ProbsEWA.jl contains the algorithm that classifies the convergence of a specific combination of learning dynamics and game.
This algorithm is used in DeepEWA_game.jl, which generates data for the training of a neural network for the predicting of convergence propetries of EWA.

Dependencies can be found in dependencies.jl

---
### Running training

```shell
# run the character segmentation and classification pipeline 
# on a folder containing .png files
PATH_TO_JULIA DeepEWA_game.jl
```

---

