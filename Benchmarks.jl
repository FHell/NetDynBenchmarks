using DifferentialEquations
using Sundials # for CVODE_BDF
using ODEInterfaceDiffEq # for rodas
using LSODA # for lsoda
using Plots
using LightGraphs
using NetworkDynamics
using DiffEqDevTools # for TestSolution


function kuramoto_edge!(e,v_s,v_d,p,t)
    e[1] = sin(v_s[1] - v_d[1]) * p.T_inv * t
    nothing
end

struct kuramoto_parameters
    ω
    T_inv
end

function kuramoto_vertex!(dv, v, e_s, e_d, p, t)
    # Note that e_s and e_d might be empty, the code needs to be able to deal
    # with this situation.
    dv .= p.ω
    for e in e_s
        dv .-= e[1]
    end
    for e in e_d
        dv .+= e[1] # .-= e[2]
    end
    nothing
end

g = barabasi_albert(100,5)

odevertex = ODEVertex(f! = kuramoto_vertex!, dim = 1)
staticedge = StaticEdge(f! = kuramoto_edge!, dim = 1)

vertexes = [odevertex for v in vertices(g)]
edgices = [staticedge for e in edges(g)]

parameters = [kuramoto_parameters(10. * randn(), 0.) for v in vertices(g)]
append!(parameters, [kuramoto_parameters(0.,  1. /100.) for v in edges(g)])

kuramoto_network! = network_dynamics(vertexes,edgices,g)

x0 = randn(nv(g))
dx = similar(x0)

kuramoto_network!(dx, x0, parameters, 0.)

prob = ODEProblem(kuramoto_network!, x0, (0.,1000.), parameters)

sol_lp = solve(prob,CVODE_BDF(),abstol=1/10^9,reltol=1/10^9)
plot(sol_lp)

function order_parameter(sol)
    [abs(sum((exp.(im .* sol[:,i]))))/length(sol[:,i]) for i in 1:length(sol[1,:])]
end
plot(sol_lp.t ./100., order_parameter(sol_lp))

sol = solve(prob,Rodas4(autodiff=false),abstol=1/10^10,reltol=1/10^10)
test_sol = TestSolution(sol)


plot(sol)

abstols = 1. /10 .^(5:8)
reltols = 1. /10 .^(1:4);
setups = [Dict(:alg=>Rosenbrock23(autodiff=false)),
          Dict(:alg=>Rodas3(autodiff=false)),
          Dict(:alg=>TRBDF2(autodiff=false)),
          Dict(:alg=>CVODE_BDF())] #,
        #  Dict(:alg=>rodas()),
        #  Dict(:alg=>radau()),
        #  Dict(:alg=>lsoda())]
wp = WorkPrecisionSet(prob,abstols,reltols,setups;verbose=false,
                      save_everystep=false,appxsol=test_sol,maxiters=Int(1e5))
plot(wp)
