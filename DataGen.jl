using Flux, .pEWA, Random, IterTools, ProgressMeter, DataFrames, CSV, Statistics

# precompute some benchmark games
coord = [[[5 1; 1 4], [5 1; 1 4]], pEWA.find_NE_mixed([[5 1; 1 4], [5 1; 1 4]])]
dom = [[[5 0; 20 1], [5 0; 20 1]], pEWA.find_NE_mixed([[5 0; 20 1], [5 0; 20 1]])]
cyclic = [[[5 1; 1 4], [-5 1; 1 -4]], pEWA.find_NE_mixed([[5 1; 1 4], [-5 1; 1 -4]])]

# set game and amount of observations
game = coord
points = 100_000

# generate data
α_grid, κ_grid, δ_grid = [rand(0.0:0.0001:1.0, points) for _ in 1:3]
β_grid = exp.(rand(-0.5:0.001:1.5, points)) # take exp to get the lim to infty effect since β is unbounded above
combs = [[α_grid[i], κ_grid[i], δ_grid[i], β_grid[i]] for i in 1:points]

# make empty 50/50 train test split vectors
subset = Int64(floor(0.5 * length(combs)))
x_train, x_test = combs[1:subset], combs[(subset+1):end]
y_train, y_test = Vector{Int64}(undef, length(x_train)), Vector{Int64}(undef, length(x_test))

# populate vectors with convergence classifications
@showprogress for i in 1:length(x_train)
    comb = x_train[i]
    α, κ, δ, β = comb
    params = pEWA.init_pEWA(; α=α, κ=κ, δ=δ, β=β, game=game)
    cat = pEWA.multicat_pEWA(params)
    y_train[i] = cat
end

@showprogress for i in 1:length(x_test)
    comb = x_test[i]
    α, κ, δ, β = comb
    params = pEWA.init_pEWA(; α=α, κ=κ, δ=δ, β=β, game=game)
    cat = pEWA.multicat_pEWA(params)
    y_test[i] = cat
end

# formatting for Flux.jl NN
x_train = hcat([Float32.(x[1:4]) for x in x_train]...)
x_test = hcat([Float32.(x[1:4]) for x in x_test]...)