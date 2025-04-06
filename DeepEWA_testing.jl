include("ProbsEWA.jl")
using Flux, .pEWA, Random, IterTools, ProgressMeter, DataFrames, CSV, Statistics, Plots, BSON

# Game setup
coord = [[[5 1; 1 4], [5 1; 1 4]], pEWA.find_NE_mixed([[5 1; 1 4], [5 1; 1 4]])]
dom = [[[1 3; 0 2],[1 0; 3 2]], pEWA.find_NE_mixed([[1 3; 0 2],[1 0; 3 2]])]
cyclic = [[[5 1; 1 4], [-5 1; 1 -4]], pEWA.find_NE_mixed([[5 1; 1 4], [-5 1; 1 -4]])]

# Choose game and number of samples
game = coord

# Model definition
deepEWA = Chain(
    Dense(12 => 32, relu),
    Dense(32 => 32, relu),
    Dense(32 => 32, relu),
    Dense(32 => 4)
)

# Save model
BSON.@load "model.bson" deepEWA

points = 10_000

# Visualization
custom_palette = [RGB(0.9, 0.1, 0.1), RGB(0.1, 0.7, 0.1), RGB(1.0, 1.0, 0.0), RGB(0.3, 0.8, 0.8)]

# Parameter sampling
α_grid, κ_grid, δ_grid = [rand(0.0:0.0001:1.0, points) for _ in 1:3]
β_grid = exp.(rand(-0.5:0.001:1.5, points))
mat1s = [rand(0:10, 2, 2) for _ in 1:points]
mat2s = [rand(0:10, 2, 2) for _ in 1:points]
combs = [[α_grid[i], κ_grid[i], δ_grid[i], β_grid[i], mat1s[i], mat2s[i]] for i in 1:points]

# Split data
x_test = combs
y_test = Vector{Int}(undef, points)

# Compute test labels
@showprogress for i in eachindex(x_test)
    α, κ, δ, β, m1, m2 = x_test[i]
    rand_game = [[m1, m2], pEWA.find_NE_mixed([m1, m2])]
    params = pEWA.init_pEWA(; α, κ, δ, β, game=rand_game)
    y_test[i] = pEWA.multicat_pEWA(params)
end

# Feature flattening
function flatten_features(data)
    hcat([Float32.(vcat(x[1:4], vec(x[5]), vec(x[6]))) for x in data]...)
end

x_test = flatten_features(x_test)

# Evaluation on new game
new_game = dom

#fix vars

# x_test[1, :] .= 0.0  # α = 1
x_test[2, :] .= 0.0  # κ = 1
x_test[3, :] .= 1.0  # δ = 1
#x_test[4, :] .= 0.0  # β = 0

y_test_new = Vector{Int}(undef, size(x_test, 2))

@showprogress for i in 1:size(x_test, 2)
    α, κ, δ, β = Float64.(x_test[1:4, i])
    params = pEWA.init_pEWA(; α, κ, δ, β, game=new_game)
    y_test_new[i] = pEWA.multicat_pEWA(params)
end

out_new = deepEWA(x_test)
probs_new = softmax(out_new, dims=1)
predicted_labels_new = Flux.onecold(probs_new, 1:4)

# Plot α (alpha) vs β (beta)
p_true_new = scatter(x_test[1, :], x_test[4, :], group=y_test_new, 
    title="True convergence", legend=true, markersize=3.5, 
    palette=custom_palette, xlabel="α (alpha)", ylabel="β (beta)", yscale=:ln)
p_pred_new = scatter(x_test[1, :], x_test[4, :], group=predicted_labels_new, 
    title="Predicted convergence", legend=false, markersize=3.5, 
    palette=custom_palette, xlabel="δ (alpha)", ylabel="β (beta)", yscale=:ln)

plot(p_true_new, p_pred_new, layout=(1, 2))
savefig("EWA_dom_alpha_vs_beta_k0_d1.png")


accuracy_new = mean(predicted_labels_new .== y_test_new)
println("Accuracy on new game (coord): ", round(accuracy_new * 100, digits=2), "%")
