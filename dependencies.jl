using Pkg

dependencies = [
    "Distributions",
    "GameTheory",
    "Flux", 
    ".fEWA",
    "Random", 
    "IterTools", 
    "ProgressMeter",
    "ProgressMeter",
    "DataFrames",
    "CSV",
    "Statistics",
    "Plots",
    "BSON"
]

Pkg.add(dependencies)