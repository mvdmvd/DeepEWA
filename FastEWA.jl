module fEWA

using Distributions, GameTheory

# returns a vector of all pure and mixed NE for a 2x2 normal form in the format Vector{Matrix{Int64}}
function find_NE_mixed(payoff::Vector{Matrix{Int64}})
    g = NormalFormGame([Player(payoff[1]), Player(payoff[2])]) # GameTheory.jl game formatting
    found = support_enumeration(g) # solve for NE with GameTheory.jl
    NE = [collect(ne) for ne in found] # convert to regular vector
    return NE
end

#= # This version finds only the pure NE, it's faster but utility depends on usecase
function find_NE_pure(payoff)
    g = NormalFormGame([Player(payoff[1]), Player(payoff[2])])
    found = pure_nash(g)
    NE = [collect(ne) for ne in found]
    return NE
end
=#

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # Initialisation
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
dom = [[5 0; 20 1], [5 0; 20 1]] # standard domination game
dom_game = [dom, fEWA.find_NE_mixed(dom)]

function init_EWA(; # initialisation function
    s₀=[0, 0],                   # empty actions vector
    μ₀=[0.0, 0.0],               # μᵢ represents the probability of playing a₁ based on the history of player i
    Q₀=[[0.0, 0.0], [0.0, 0.0]], # prior attractions (technically a parameter)
    N₀=0.0,                      # prior history (also a parameter)
    α=1.0,                       # memory loss
    κ=0.0,                       # discount rate
    δ=1.0,                       # degree of foregone payoffs consideration
    β=Inf64,                     # thermodynamic beta, controls stochasticity (β→lim ∞ = deterministic policy, β→lim ∞ = fully random policy)
    game=dom_game)
    return s₀, μ₀, Q₀, N₀, α, κ, δ, β, game # default parameterisation: best response dynamics for a domination game (pure NE always found). 
end

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # Simulation
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

function EWA_step!( # step function first selects actions based off attractions, then updates attractions based off actions
    sₜ::Vector{Int64}, μ::Vector{Float64}, Qₜ::Vector{Vector{Float64}}, Nₜ::Float64,
    α::Float64, κ::Float64, δ::Float64, β::Float64, payoff::Vector{Matrix{Int64}})

    # new actions
    @inbounds for i ∈ 1:2 # iterate over players
        Q₁, Q₂ = Qₜ[i...] # get current attractions
        if isinf(β) # check if deterministic policy, if so, do fast action selection
            if Q₁ > Q₂
                sₜ[i] = 1
                continue
            elseif Q₁ < Q₂
                sₜ[i] = 2
                continue
            else # tie breaker is now 50/50 assignment (note: this is stochastic, however we follow the EWA equations)
                sₜ[i] = (rand(Bernoulli(0.5)) == 1 ? 1 : 2) # draw action
            end
        else # stochastic policy
            local s1, s2 = β * Q₁, β * Q₂
            local m = s1 > s2 ? s1 : s2 # find max
            local exp1, exp2 = exp(s1 - m), exp(s2 - m)
            local p₁ = exp1 / (exp1 + exp2) # softmax, subtracting the max for numerical stability
            sₜ[i] = (rand(Bernoulli(p₁)) == 1 ? 1 : 2)
        end
        μ[i] += sₜ[i] == 1 ? 1.0 : 0.0 # numerator for bayesian updating
    end

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

function Run_FastEWA(parameters::Tuple{Vector{Int64},Vector{Float64},Vector{Vector{Float64}},Float64,Float64,Float64,Float64,Float64,Vector{Vector}};
    T=10000)
    s₀, μ₀, Q₀, N₀, α, κ, δ, β, game = parameters
    payoff, NE = game
    conv = T
    # convergence criterion is whether the expectation of the histories of play are an NE
    local NE_found = false # set to true if converged = NE 
    sₜ, μ, Qₜ, Nₜ = EWA_step!(s₀, μ₀, Q₀, N₀, α, κ, δ, β, payoff) # the first EWA step is based on the priors
    @inbounds for t in 1:T # T=1000 is assumed to be close enough to ∞, test this for more rigor (low T is required for speed).
        sₜ, μ, Qₜ, Nₜ = EWA_step!(sₜ, μ, Qₜ, Nₜ, α, κ, δ, β, payoff)
        local μ₁, μ₂ = μ ./ (t + 1) # bayesian updating of a₁ probabilities 
        local σ = [[μ₁, 1 - μ₁], [μ₂, 1 - μ₂]] # compute mathematical expectation of strategies

        if any(isapprox(σ, ne, atol=0.005) for ne in NE)
            NE_found = true # break if converged expectation is a NE
            # future: check for multiple NE, FP, limit cycles, chaos
            conv = t
            break
        end
    end
    μ₁, μ₂ = μ ./ (conv)
    σ = [[μ₁, 1 - μ₁], [μ₂, 1 - μ₂]]
    return sₜ, σ, Qₜ, NE_found
end

function multicat_FastEWA(parameters::Tuple{Vector{Int64},Vector{Float64},Vector{Vector{Float64}},Float64,Float64,Float64,Float64,Float64,Vector{Vector}};
    T=10000)
    s₀, μ₀, Q₀, N₀, α, κ, δ, β, game = parameters
    payoff, NE = game

    # convergence criterion is whether the expectation of the histories of play are an NE
    local cycles, pure_NE, mixed_NE, FP = true, false, false, false # set to true if converged = NE 
    conv = T
    sₜ, μ, Qₜ, Nₜ = EWA_step!(s₀, μ₀, Q₀, N₀, α, κ, δ, β, payoff) # the first EWA step is based on the priors
    @inbounds for t in 1:T # T=1000 is assumed to be close enough to ∞, test this for more rigor (low T is required for speed).
        old_μ = μ
        old_μ₁, old_μ₂ = old_μ ./ t
        old_σ = [[old_μ₁, 1 - old_μ₁], [old_μ₂, 1 - old_μ₂]]
        sₜ, μ, Qₜ, Nₜ = EWA_step!(sₜ, μ, Qₜ, Nₜ, α, κ, δ, β, payoff)
        μ₁, μ₂ = μ ./ (t + 1) # bayesian updating of a₁ probabilities 
        σ = [[μ₁, 1 - μ₁], [μ₂, 1 - μ₂]] # compute mathematical expectation of strategies
        if t > 1000 && any(isapprox(old_σ, σ, atol=0.001))
            cycles = false # break if converged expectation is a NE
            # future: check for multiple NE, FP, limit cycles, chaos
            if any(isapprox(σ, ne, atol=0.01) for ne in NE)
                if any(x -> isapprox(x, 1.0, atol=0.01), σ[1])
                    pure_NE = true
                else
                    mixed_NE = true
                end
            else
                FP = true
            end
            conv = t
            break
        end

    end
    μ₁, μ₂ = μ ./ (conv) # bayesian updating of a₁ probabilities 
    σ = [[μ₁, 1 - μ₁], [μ₂, 1 - μ₂]]
    return sₜ, σ, Qₜ, cycles, pure_NE, mixed_NE, FP
end

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
############################################################################################################################################

end




