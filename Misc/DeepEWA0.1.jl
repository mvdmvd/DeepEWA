include("/Users/meesvandartel/Desktop/Coursework/CGT/DeepEWA/FastEWA.jl")
using Flux, .fEWA, Random, IterTools

coord = [[[5 1; 1 4], [5 1; 1 4]], fEWA.find_NE_mixed([[5 1; 1 4], [5 1; 1 4]])]


α_grid = rand(0.0:0.01:1.0, 3)
κ_grid = rand(0.0:0.01:1.0, 3)
δ_grid = rand(0.0:0.01:1.0, 3)
β_grid = exp.(rand(0.0:0.1:4.0, 3))

combinations = [[α, κ, δ, β] for (α, κ, δ, β) in IterTools.product(α_grid, κ_grid, δ_grid, β_grid)]

print(combinations)
#=

    params = fEWA.init_EWA(;α=α, κ=κ, δ=δ, β=β, game=coord)
    sₜ, σ, Qₜ, NE_found  = fEWA.Run_FastEWA(params)
    =#