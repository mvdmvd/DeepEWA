include("/Users/meesvandartel/Desktop/Coursework/CGT/DeepEWA/DeepEWA_funs&structs.jl")
using Flux, Statistics, ProgressMeter, .EWA, Base.Iterators, ProgressMeter, Random
using Base.Threads

coord = [[5 1; 1 4], [5 1; 1 4]]
dom = [[5 0; 20 1], [5 0; 20 1]]
cyclic = [[5 1; 1 4], [-5 1; 1 -4]]

payoffs = [coord, dom, cyclic]
payoff_names = ["Coordination", "Dominance", "Cyclic"]


LinGrid1 = 0.0:0.1:1.0
LogGridβ = vcat(0.0, collect(logrange(1e-6, 999.0, (length(LinGrid1) - 1))))
shots = 1

total_iterations = length(LinGrid1)^3 * length(LogGridβ) * length(payoffs)

data = Vector{Any}(undef, total_iterations)



total_iterations = length(LinGrid1)^3 * length(LogGridβ) * length(payoffs)

n_alpha = length(LinGrid1)
n_delta = length(LinGrid1)
n_kappa = length(LinGrid1)
n_beta = length(LogGridβ)
n_payoffs = length(payoffs)
parameter_space = CartesianIndices((n_payoffs, n_alpha, n_delta, n_kappa, n_beta))



@showprogress for i in 1:total_iterations
    if rand() < 0.7
        continue
    end


    idx_tuple = parameter_space[i]
    payoff_idx, alpha_idx, delta_idx, kappa_idx, beta_idx = Tuple(idx_tuple)
    current_payoff = payoffs[payoff_idx]
    current_payoff_name = payoff_names[payoff_idx]
    α = LinGrid1[alpha_idx]
    δ = LinGrid1[delta_idx]
    κ = LinGrid1[kappa_idx]
    β = LogGridβ[beta_idx]

    L = 0
    NEs = Set()
    FPs = Set()
    lim_chaos_found_in_shots = false
    for t in 1:shots
        params = EWA.init_EWA(
            Q₀=[[0.0, 0.0], [0.0, 0.0]], N₀=0.0, # priors
            α=α, δ=δ, κ=κ, β=β, # params
            payoff=current_payoff)
        _, _, sₜ, _, lim_chaos, NE_found, NE = EWA.run_EWA_pure(params; print=false)
        if lim_chaos == true
            L = 1
            lim_chaos_found_in_shots = true
            break
        elseif NE_found == true
            push!(NEs, sₜ)
        else
            push!(FPs, sₜ)
        end
    end  # L: 1 = chaos, 2 = unique NE, 3 = multiple NE, 4 = unique FP, 5 = multiple FP
    if !lim_chaos_found_in_shots
        if length(NEs) == 1
            L = 1
        elseif length(NEs) > 1
            L = 2
        elseif length(FPs) == 1
            L = 3
        elseif length(FPs) > 1
            L = 4
        else
            L = 5
        end
    end
    data[i] = (PayoffName=current_payoff_name, Alpha=α, Delta=δ, Kappa=κ, Beta=β, ResultCode=L)
end

print(data)