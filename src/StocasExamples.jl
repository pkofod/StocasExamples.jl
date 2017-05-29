module StocasExamples

using Stocas

# An example from Aguirregabiria Mira 2010
function am2010(;Nmarket = 5, β = 0.95)
    ### Aguirregabiria & Mira
    ## State 1: Endogenous incumbancy state
    X_entry = EntryState(;dense=false)

    ## State 2: Exogenous market state
    # Construct a market state with typical probability matrix as below
    # (shown for Nmarket = 5)
    # M = [0.8 0.2 0.0 0.0 0.0;
    #      0.2 0.6 0.2 0.0 0.0;
    #      0.0 0.2 0.6 0.2 0.0;
    #      0.0 0.0 0.2 0.6 0.2;
    #      0.0 0.0 0.0 0.2 0.8;]

    X2 = 1:Nmarket
    M = zeros(Nmarket, Nmarket)
    p = [0.2, 0.6, 0.2]
    # First corner
    M[1, 1:2] = [sum(p[1:2]), p[3]]
    # Last corner
    M[end, end-1:end] = [p[1], sum(p[2:3])]
    # Second to second last rows
    M[2:end-1, :] = vcat([vcat(zeros(i-2), p, zeros(Nmarket-1-i))' for i = 2:Nmarket-1]...)

    # Combine states
    S = States(X_entry, CommonState(X2, M))

    ## Utility specification
    # Z matrices for
    # exit choice:
    Z1 = zeros(Nmarket*2, 3)
    # entry choice:
    Z2 = [-ones(Nmarket) log.(X2) -ones(Nmarket);
          -ones(Nmarket) log.(X2)  zeros(Nmarket)]
    Z = (Z1, Z2)

    truepar = [-1.9;1.;2.] # fixed costs, ..., entry costs
    U = LinearUtility(Z, β, truepar)

    return U, S
end

# As AM but with a Tauchen discretized Gaussian market state
function am_tauchen(;β=0.95, # discount factor,
                     N = 500)
    # Aguirregabiria & Magesan 2016
    truepar = [-1.9;1.;2.] # fixed costs, ..., entry costs

    nX2 = N
    X2 = 1:nX2
    F2 = Stocas.tauchen(0.1,0.01,2, nX2)

    S = States(EntryState(),
               CommonState(X2, F2))

    Z = (zeros(nX2*2, 3), # exit
        [-ones(nX2) log.(X2) -ones(nX2); # entry
         -ones(nX2) log.(X2)  zeros(nX2)])

    U = LinearUtility(Z, β, truepar)

    return U, S
end

# A simple market entry model
function dixit(;N = 5, β = 0.99)
    # State 1
    X1 = 0:1
    F1 = [[1. 0.; 1. 0.], [0. 1.; 0. 1.]]
    # State 2
    nX2 = N
    X2 = 1:nX2
    F2 = 1./(1+abs.(ones(length(X2),1)*X2'-X2*ones(1, length(X2))))
    F2 = F2./sum(F2,1)'
    # States
    S = States(State("incumbancy", X1, F1), CommonState("market", X2, F2))

    # Utility variables
    Z1 = zeros(nX2*2, 3)
    Z2 = [ones(nX2) X2 -ones(nX2);
          ones(nX2) X2 zeros(nX2)]
    U = LinearUtility((Z1, Z2), β, [-.1;.2;1])

    return U, S
end

# An example from a presentation by Günter Hitsch
function hitsch(;N = 20)
    n = [0, 2, 5]
    next_i(i, k) = max.(0, min.(I, i+n[k]-1))
    F_X = ["low", "high"]
    F_P = [0.16 0.84; 0.16 0.84]
    SP = CommonState(F_X, F_P)

    I = N
    K = length(n)
    F_i = [spzeros(1+I, 1+I) for k = 1:K]
    for i = 0:I, j = 1:K
            F_i[j][i+1, next_i(i, j)+1]=1
    end
    Si = State(0:I, F_i);

    S = States(SP, Si)

    delta, alpha = 4., 4.

    Z1 = [[zeros(1); ones(I)] zeros(I+1) -next_i(0:I, 1);
          [zeros(1); ones(I)] zeros(I+1) -next_i(0:I, 1)]
    Z2 = [ones(2*(I+1)) -[1.2*ones(I+1);2*ones(I+1)] -kron(ones(2), next_i(0:I, 2))]
    Z3 = [ones(2*(I+1)) -[3.0*ones(I+1);5*ones(I+1)] -kron(ones(2), next_i(0:I, 3))]
    U = LinearUtility(("Buy 0", "Buy 2", "Buy 5"), (Z1, Z2, Z3), 0.998, [delta; alpha; 0.05])
    return U, S
end

# Rust 1987
function rust(;N = 175, β = 0.95, sparse = false)
    # State space
    nX = N
    n = nX
    o_maximum = 450
    X = linspace(0, o_maximum, nX)
    p = [0.0937; 0.4475; 0.4459; 0.0127; 0.0002];

    F1 = zeros(nX, nX)
    offset = -1
    for i = 1:nX, j = 1:length(p)
        i+offset+j > nX && continue
        F1[i,i+offset+j] = i+offset+j == nX ? sum(p[j:end]) : p[j]
    end

    # We can handle sparse matrices
    F = [F1, F1[ones(Int64, nX), :]]
    sparse == true && map!(sparse, F)
    S = State(:mileage, X, F)

    Z1 = [zeros(nX) -0.001*X]
    Z2 = [-ones(nX) zeros(nX)]

    U = LinearUtility(("replace", "repair"),(Z1, Z2), β, [11.;2.5])

    return U, S
end
# A somewhat contrived day effect model
function seven(;N = 5, β = 0.95)
	# Aguirregabiria & Mira
	# 2 Model
	truepar = [-1.9;1.;2.] # fixed costs, ..., entry costs
	# State 1: Exogenous state
	nX1 = 2
	X1 = [0.;1.]
	F1 = [sparse([1. 0.; 1. 0.]), sparse([0. 1.; 0. 1.])]

	nX2 = N
	X2 = 1:nX2
	M = [zeros(nX2-1) eye(nX2-1, nX2-1); [1.0 zeros(1,nX2-1)]]

	S = States(State(X1, F1),
	           CommonState(X2, M))

	Z = (zeros(nX2*2, 3), # exit
	             [-ones(nX2) log.(X2) -ones(nX2); # entry
				  -ones(nX2) log.(X2)  zeros(nX2)])

	U = LinearUtility(Z, β, truepar)

    return U, S
end
end # module
