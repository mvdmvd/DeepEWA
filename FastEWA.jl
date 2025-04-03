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
    T=1000)
    s₀, μ₀, Q₀, N₀, α, κ, δ, β, game = parameters
    payoff, NE = game

    # convergence criterion is whether the expectation of the histories of play are an NE
    local NE_found = false # set to true if converged = NE 
    sₜ, μ, Qₜ, Nₜ = EWA_step!(s₀, μ₀, Q₀, N₀, α, κ, δ, β, payoff) # the first EWA step is based on the priors
    @inbounds for t in 1:T # T=1000 is assumed to be close enough to ∞, test this for more rigor (low T is required for speed)
        sₜ, μ, Qₜ, Nₜ = EWA_step!(sₜ, μ, Qₜ, Nₜ, α, κ, δ, β, payoff)
        local μ₁, μ₂ = μ ./ (t + 1) # bayesian updating of a₁ probabilities 
        local σ = [[μ₁, 1 - μ₁], [μ₂, 1 - μ₂]] # Mathematical expectation of strategies

        if any(isapprox(σ, ne, atol=0.1) for ne in NE)
            NE_found = true # check if this expectation is an NE, break early if so
            break           # no convergence or convergence to FP, or mixed FP that is not NE is treated equally
        end                 # a good next step is to classify into 1. limit cycles / chaos, 2. single pure NE, 3. Mult. pure NE, 4. single FP, 5. mMult. FP
    end                     # however this is tricky to do fast. for multiple solutions need to run the same parameters n times
    # for limit cycles or chaos, check if expected strategies converge, if not classify as true
    μ₁, μ₂ = μ ./ (T + 1)
    σ = [[μ₁, 1 - μ₁], [μ₂, 1 - μ₂]]
    return sₜ, σ, Qₜ, NE_found
end

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
############################################################################################################################################
end




