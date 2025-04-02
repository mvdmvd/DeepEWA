module EWA

using Distributions, NNlib, GameTheory # NNlib for numerically stable softmax


# some example games
coord_payoff = [[5 1; 1 4], [5 1; 1 4]] #player 1, player 2 payoff matrixes respectively
pris_payoff = [[5 20; 0 1], [5 20; 0 1]]

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


function init_EWA(;
    Q₀=[[0.0, 0.0], [0.0, 0.0]], # Prior attractions
    κ=0.0,
    α=1.0,
    N₀=0.0,
    δ=1.0,
    payoff=pris_payoff,
    β=Inf64)
    return Q₀, κ, α, N₀, δ, payoff, β # default params is best response dynamics
end

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # Simulation steps
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
function mixed_strategy_draw(β::Float64, Qₙ::Vector{Vector{Float64}})
    # This function computes the period t (mixed) strategy vector based off current attractions.
    s = Int64[0, 0] #placeholder
    σₜ = [Float64[], Float64[]]

    for i in 1:2 # for both players
        Q = Qₙ[i] # get attractions vector for player i
        if isinf(β) # if deterministic policy:
            if Q[1] > Q[2]
                p = [1.0, 0.0]
            elseif Q[1] < Q[2]
                p = [0.0, 1.0]
            else
                p = [0.5, 0.5] # tie breaker, I guess this is still stochastic, maybe choose deterministically?
            end
        else
            #Stochastic policy
            p = exp.(logsoftmax(β * Q)) # log softmax for numerical stability (for high β values)
        end
        a = 2 - rand(Bernoulli(p[1])) # draw an action from the mixed strategy vector. 2-Binomial(μ)= 1 with P=μ, 2 else.
        s[i] = a # next action for player i
        σₜ[i] = [p[1], 1 - p[1]] # mixed strat vector for player i
    end
    return s, σₜ
end

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
function update_EWA(
    s::Vector{Int64},
    N::Float64,
    Q::Vector{Vector{Float64}},
    κ::Float64, α::Float64, δ::Float64, payoff::Vector{Matrix{Int64}})

    Qₙ = [Float64[0.0, 0.0], Float64[0.0, 0.0]] # updated Q vector placeholder
    Nₙ = (1 - α) * (1 - κ) * N + 1 # incremented history

    for i in 1:2 # for both players 
        j = 3 - i # j is opponent
        sᵢ = s[i] # action player i played
        for a in 1:2 # update the attractions
            I = (a == sᵢ) ? 1 : 0 # indicator function for if action was played
            Π = payoff[i][a, s[j]] # find action's payoff
            Qᵤ = ((1 - α) * N * Q[i][a] + (δ + (1 - δ) * I) * Π) / Nₙ # EWA updating rule
            Qₙ[i][a] = Qᵤ # update attraction of each action
        end
    end
    return Qₙ, Nₙ
end


function EWA_step!(Q, N, κ, α, δ, payoff, β)
    s, σ = mixed_strategy_draw(β, Q) # compute new mixed strat and draw action
    Qₙ, Nₙ = update_EWA(s, N, Q, κ, α, δ, payoff) # compute new attractions based off payoffs
    return Qₙ, Nₙ, s, σ
end

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # Run simulation
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
function run_EWA_mixed(parameters; T=1000000000, ϵ=1e-4)
    Q₀, κ, α, N₀, δ, payoff, β = parameters
    println("Prior Q: $Q₀, prior N: $N₀")

    count_1 = 0
    count_2 = 0

    Qₜ, Nₜ, sₜ, σₜ = EWA_step!(Q₀, N₀, κ, α, δ, payoff, β) # first iter from priors

    conv = 0
    window = 0
    NE_found = false
    for t in 1:T
        if (t ≤ 10 && t % 2 == 0) || (t ≤ 100 && t % 50 == 0)
            println("Iter $t: σ=$σₜ, s=$sₜ, Q=$Qₜ, N=$Nₜ")
        end

        Qₜ, Nₜ, sₜ, σₜ = EWA_step!(Qₜ, Nₜ, κ, α, δ, payoff, β)

        old_1 = count_1 / (t - 1)
        old_2 = count_2 / (t - 1)

        count_1 += sₜ[1] == 1 ? 1 : 0
        count_2 += sₜ[2] == 1 ? 1 : 0

        new_1 = count_1 / t
        new_2 = count_2 / t

        if abs(old_1 - new_1) < ϵ && abs(old_2 - new_2) < ϵ
            window += 1
        end

        if t > 100 && window > 100
            println("Converged at iter: $t")
            break
        end
        conv = t
    end
    freq_1 = count_1 / conv
    freq_2 = count_2 / conv
    conv_sol = [[freq_1, 1 - freq_1], [freq_2, 1 - freq_2]]
    NE = find_NE_mixed(payoff)

    for ne in NE
        for i in eachindex(ne)
            for j in 1:2
                if all(all.(abs.(conv_sol[j] .- ne[i]) .< 0.015))
                    NE_found = true
                end
            end
        end
    end

    return Qₜ, Nₜ, sₜ, σₜ, conv_sol, NE_found, NE
end





function run_EWA_pure(parameters; T=1000000, ϵ=1e-4)
    Q₀, κ, α, N₀, δ, payoff, β = parameters
    println("Prior Q: $Q₀, prior N: $N₀")

    a₁ = 0
    a₂ = 0

    Qₜ, Nₜ, sₜ, σₜ = EWA_step!(Q₀, N₀, κ, α, δ, payoff, β) # first iter from priors

    conv = 0
    window = 0
    lim_chaos = false
    NE_found = false
    for t in 1:T
        if (t ≤ 10 && t % 2 == 0) || (t ≤ 100 && t % 50 == 0)
            println("Iter $t: σ=$σₜ, s=$sₜ, Q=$Qₜ, N=$Nₜ")
        end

        old_a₁ = a₁
        old_a₂ = a₂

        Qₜ, Nₜ, sₜ, σₜ = EWA_step!(Qₜ, Nₜ, κ, α, δ, payoff, β)

        a₁ = sₜ[1]
        a₂ = sₜ[2]

        if old_a₁ == a₁ && old_a₂ == a₂
            window += 1
        end

        if t > 100 && window > 100
            println("Converged at iter: $t")
            break
        end
        conv = t
    end
    lim_chaos = conv ≥ T ? true : lim_chaos
    Nash = find_NE_pure(payoff)
    NE_found = sₜ ∈ Nash ? true : NE_found

    return Qₜ, Nₜ, sₜ, σₜ, lim_chaos, NE_found, Nash
end

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
############################################################################################################################################

end




