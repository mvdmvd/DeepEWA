module pEWA
using Distributions, GameTheory

function find_NE_mixed(payoff::Vector{Matrix{Int64}})
    g = NormalFormGame([Player(payoff[1]), Player(payoff[2])]) # GameTheory.jl game formatting
    found = support_enumeration(g) # solve for NE with GameTheory.jl
    NE = [collect(ne) for ne in found] # convert to regular vector
    return NE
end

function init_pEWA(; # initialisation function
    s₀=[0, 0],                   # empty actions vector
    μ₀=[], # μᵢ represents the probability of playing a₁
    Q₀=[[0.0001, 0.0001], [0.0001, 0.0001]], # prior attractions (technically a parameter)
    N₀=0.0,                      # prior history (also a parameter)
    α=1.0,                       # memory loss
    κ=0.0,                       # discount rate
    δ=1.0,                       # degree of foregone payoffs consideration
    β=Inf64,                     # thermodynamic beta, controls stochasticity (β→lim ∞ = deterministic policy, β→lim ∞ = fully random policy)
    game=dom_game)
    return s₀, μ₀, Q₀, N₀, α, κ, δ, β, game # default parameterisation: best response dynamics for a domination game (pure NE always found). 
end

function pEWA_step!( # step function first selects actions based off attractions, then updates attractions based off actions
    sₜ::Vector{Int64}, μ::Vector{Any}, Qₜ::Vector{Vector{Float64}}, Nₜ::Float64,
    α::Float64, κ::Float64, δ::Float64, β::Float64, payoff::Vector{Matrix{Int64}})

    σ = [[0.0, 0.0], [0.0, 0.0]]
    # new actions
    @inbounds for i ∈ 1:2 # iterate over players
        Q₁, Q₂ = Qₜ[i...] # get current attractions
        local s1, s2 = β * Q₁, β * Q₂
        local m = s1 > s2 ? s1 : s2 # find max
        local exp1, exp2 = exp(s1 - m), exp(s2 - m)
        local p₁ = exp1 / (exp1 + exp2) # softmax, subtracting the max for numerical stability
        sₜ[i] = rand(Bernoulli(p₁)) == 1 ? 1 : 2 # draw action
        σ[i] = [p₁, 1 - p₁] # store mixed strategy
    end
    push!(μ, σ)

    # update attractions
    local Nₙ = (1 - α) * (1 - κ) * Nₜ + 1 # update history
    @inbounds for i ∈ 1:2
        j = 3 - i # j is other player
        for aᵢ ∈ 1:2 # iterate over action space
            I = (aᵢ == sₜ[i]) ? 1.0 : 0.0 # indicator function for action was played
            Πₐ = payoff[i][aᵢ, sₜ[j]] # payoff for action
            local oldQ = Qₜ[i][aᵢ]
            Qₜ[i][aᵢ] = ((1 - α) * Nₜ * oldQ + (δ + (1 - δ) * I) * Πₐ) / Nₙ # EWA updating
        end
    end
    return sₜ, μ, Qₜ, Nₜ
end

function multicat_pEWA(parameters::Tuple{Vector{Int64},Vector{Any},Vector{Vector{Float64}},Float64,Float64,Float64,Float64,Float64,Vector{Vector}};
    T=5000)
    s₀, μ₀, Q₀, N₀, α, κ, δ, β, game = parameters
    payoff, NE = game

    # convergence criterion is whether the expectation of the histories of play are an NE
    local cat = 1 # 1 = cycles/chaos, 2 = mixed FP, 3 = pure FP, 4= pure NE
    sₜ, μ, Qₜ, Nₜ = pEWA_step!(s₀, μ₀, Q₀, N₀, α, κ, δ, β, payoff) # the first EWA step is based on the priors
    @inbounds for t in 1:T # after T iterations, classifies as cycles/chaos
        sₜ, μ, Qₜ, Nₜ = pEWA_step!(sₜ, μ, Qₜ, Nₜ, α, κ, δ, β, payoff)
        if t > 6 && all(isapprox(μ[end-3:end-1], μ[end-2:end], atol=0.003)) # tolerance level ϵ₁
            if any(isapprox(μ[end], ne, atol=0.001) for ne in NE) # tolerance level ϵ₂
                cat = 4
            else
                any(x -> isapprox(x, 1.0, atol=0.003), μ[end][1]) ? cat = 3 : cat = 2 # tolerance level ϵ₃
            end
            break
        end
    end
    return cat
end
end