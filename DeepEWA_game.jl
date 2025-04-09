include("ProbsEWA.jl")
using Flux, .pEWA, Random, IterTools, ProgressMeter, DataFrames, CSV, Statistics, Plots, BSON

# Game setup
coord = [[[5 1; 1 4], [5 1; 1 4]], pEWA.find_NE_mixed([[5 1; 1 4], [5 1; 1 4]])]
dom = [[[1 3; 0 2], [1 0; 3 2]], pEWA.find_NE_mixed([[1 3; 0 2], [1 0; 3 2]])]
cyclic = [[[5 1; 1 4], [-5 1; 1 -4]], pEWA.find_NE_mixed([[5 1; 1 4], [-5 1; 1 -4]])]

# Choose game and number of samples
game = dom
points = 10_000

# Parameter sampling
α_grid, κ_grid, δ_grid = [rand(0.0:0.0001:1.0, points) for _ in 1:3]
β_grid = exp.(rand(-0.5:0.001:1.5, points))
mat1s = [rand(0:10, 2, 2) for _ in 1:points]
mat2s = [rand(0:10, 2, 2) for _ in 1:points]
combs = [[α_grid[i], κ_grid[i], δ_grid[i], β_grid[i], mat1s[i], mat2s[i]] for i in 1:points]

# Split data
split_idx = Int(floor(0.6 * points))
x_train, x_test = combs[1:split_idx], combs[(split_idx+1):end]
y_train = Vector{Int}(undef, split_idx)
y_test = Vector{Int}(undef, points - split_idx)

# Compute training labels
@showprogress for i in eachindex(x_train)
    α, κ, δ, β, m1, m2 = x_train[i]
    rand_game = [[m1, m2], pEWA.find_NE_mixed([m1, m2])]
    params = pEWA.init_pEWA(; α, κ, δ, β, game=rand_game)
    y_train[i] = pEWA.multicat_pEWA(params)
end

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

x_train = flatten_features(x_train)
x_test = flatten_features(x_test)

# Model definition
deepEWA = Chain(
    Dense(12 => 32, relu),
    Dense(32 => 32, relu),
    Dense(32 => 32, relu),
    Dense(32 => 4)
)

# Training setup
target = Flux.onehotbatch(y_train, 1:4)
loader = Flux.DataLoader((x_train, target), batchsize=64, shuffle=true)
opt = Flux.setup(Flux.AdaGrad(), deepEWA)

# Training loop
losses = Float32[]
@showprogress for epoch in 1:2500
    for (x, y) in loader
        loss, grads = Flux.withgradient(deepEWA) do model
            Flux.logitcrossentropy(model(x), y)
        end
        Flux.update!(opt, deepEWA, grads[1])
        push!(losses, loss)
    end
end

# Predictions
out2 = deepEWA(x_test)
probs2 = softmax(out2, dims=1)
predicted_labels = Flux.onecold(probs2, 1:4)

# Save model
BSON.@save "model.bson" deepEWA

# Visualization
custom_palette = [RGB(0.9, 0.1, 0.1), RGB(0.1, 0.7, 0.1), RGB(1.0, 1.0, 0.0), RGB(0.3, 0.8, 0.8)]

p_true = scatter(x_test[1, :], x_test[4, :], group=y_test, title="True convergence", legend=true,
    markersize=3.5, palette=custom_palette, yscale=:ln)
p_pred = scatter(x_test[1, :], x_test[4, :], group=predicted_labels, title="Predicted convergence",
    legend=false, markersize=3.5, palette=custom_palette, yscale=:ln)

plot(p_true, p_pred, layout=(1, 2), size=(900, 330))
savefig("EWA.png")

accuracy = mean(predicted_labels .== y_test)
println("Accuracy: ", round(accuracy * 100, digits=2), "%")

