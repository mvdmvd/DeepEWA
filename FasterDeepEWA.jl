module fEWA

using Distributions, NNlib, GameTheory # NNlib for numerically stable softmax


# some example games


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # GameTheory solutions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

function find_NE_mixed(payoff)
    g = NormalFormGame([Player(payoff[1]), Player(payoff[2])])
    found = support_enumeration(g)
    NE = [collect(ne) for ne in found]
    return NE
end

function find_NE_pure(payoff)
    g = NormalFormGame([Player(payoff[1]), Player(payoff[2])])
    found = pure_nash(g)
    NE = [collect(ne) for ne in found]
    return NE
end

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # Initialisation
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
dom = [[5 0; 20 1], [5 0; 20 1]]
dom_nash = fEWA.find_NE_pure(dom)
dom_game = [dom, dom_nash]

function init_EWA(;
    s₀=[0, 0],
    Q₀=[[0.0, 0.0], [0.0, 0.0]], # Prior attractions
    N₀=0.0,
    α=1.0,
    κ=0.0,
    δ=1.0,
    β=Inf64,
    game=dom_game
)
    return s₀, Q₀, N₀, α, κ, δ, β, game # default params is best response dynamics
end

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # Simulation
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

function EWA_step!(
    sₜ::Vector{Int64}, Qₜ::Vector{Vector{Float64}}, Nₜ::Float64,
    α::Float64, κ::Float64, δ::Float64, β::Float64, payoff::Vector{Matrix{Int64}})

    # new actions
    @inbounds for i ∈ 1:2 # for both players
        Q₁, Q₂ = Qₜ[i...] # get attractions vector for player i
        if isinf(β) # if deterministic policy:
            if Q₁ > Q₂
                sₜ[i] = 1
                continue
            elseif Q₁ < Q₂
                sₜ[i] = 2
                continue
            else
                p₁ = 0.5 # tie breaker, I guess this is still stochastic, maybe choose deterministically?
            end
        else
            local s1 = β * Q₁
            local s2 = β * Q₂
            local m = s1 > s2 ? s1 : s2
            local exp1 = exp(s1 - m)
            local exp2 = exp(s2 - m)
            local p₁ = exp1 / (exp1 + exp2)
        end
        sₜ[i] = (rand(Bernoulli(p₁)) == 1 ? 1 : 2) # draw an action from the mixed strategy vector. 2-Binomial(μ)= 1 with P=μ, 2 else.
    end

    local Nₙ = (1 - α) * (1 - κ) * Nₜ + 1 # incremented history

    @inbounds for i ∈ 1:2 # for both players 
        j = 3 - i # j is opponent
        for aᵢ ∈ 1:2 # update the attractions
            I = (aᵢ == sₜ[i]) ? 1.0 : 0.0 # indicator function for if action was played
            Πₐ = payoff[i][aᵢ, sₜ[j]] # find action's payoff
            local oldQ = Qₜ[i][aᵢ]
            Qₜ[i][aᵢ] = ((1 - α) * Nₜ * oldQ + (δ + (1 - δ) * I) * Πₐ) / Nₙ # EWA updating rule
        end
    end
    return sₜ, Qₜ, Nₜ
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # Run simulation
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

function run_EWA_pure(parameters; T=1000000)
    s₀, Q₀, N₀, α, κ, δ, β, game = parameters
    payoff, NE = game

    aₜ = [0.0]
    conv, window, lim_chaos, NE_found = 0, 0, false, false
    sₜ, Qₜ, Nₜ = EWA_step!(s₀, Q₀, N₀, α, κ, δ, β, payoff) # first iter from priors
    @inbounds for t in 1:T
        old_a = aₜ
        sₜ, Qₜ, Nₜ = EWA_step!(sₜ, Qₜ, Nₜ, α, κ, δ, β, payoff)
        aₜ = sₜ
        window += (old_a == aₜ) ? 1 : (window > 0 ? -window : 0)
        if window == 100
            conv = t
            break
        elseif t == T
            lim_chaos = true
            break
        end
    end

    NE_found = sₜ ∈ NE ? true : NE_found

    return sₜ, Qₜ, lim_chaos, NE_found
end

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
############################################################################################################################################

end




