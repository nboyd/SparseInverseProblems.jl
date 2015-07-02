export ForwardModel, Loss, phi, localDescent, solveFiniteDimProblem, lmo, loss
abstract ForwardModel
abstract Loss
# Computes the forward operator.
# Inputs:
#   parameters:: p by k matrix. You might want a generic list instead if the parameter space is more complex.
#   weights:: p-dimensional vector of weights.
#
#   output:: d-dimensional vector.
#   $o = \sum_{i=1}^k w_i \psi(\theta_i).$
phi(model :: ForwardModel, parameters :: Matrix{Float64}, weights :: Vector{Float64}) =
   error("phi not implemented for model $(typeof(model)).")

# Perform some local optimization of
#   $(theta_1, \ldots, \theta_k) \mapsto \ell \left(\sum_{i=1}^k w_i \psf(\theta_i) \right).$
# Hopefully uses differentiability of $\psi.$
#
# Inputs:
#   loss :: A loss function.∆˚†
#   weights :: k-dimensional vector of (fixed) weights.
#   y :: d-dimensional target vector.
#
localDescent(model :: ForwardModel, loss :: Loss, thetas :: Matrix{Float64}, weights :: Vector{Float64}, y :: Vector{Float64})  =
   error("localDescent not implemented for model $(typeof(model)) and loss $(typeof(loss)).")

# Solve the (convex) optimization problem:
# $$ \min_{0 \le w \in \mathbb{R}^k} \ell \left(\sum_i w_i \psi(\theta_i) - y \right) .$$
solveFiniteDimProblem(model :: ForwardModel, loss :: Loss, parameters :: Matrix{Float64}, y :: Vector{Float64}, tau :: Float64)  =
   error("phi not implemented for model $(typeof(model)) and loss $(typeof(loss)).")

# Get the next atom.
# i.e. Find $ \arg\max_{\theta \in \Theta} | \langle\psi(\theta), v \rangle |.$
#
# If the model requires weights to be nonnegative, we instead need to find
# $ \arg\min_{\theta \in \Theta}  \langle\psi(\theta), v \rangle .$
#
# Returns the tuple (theta, score), where score is $-|\langle \psi(\theta), v \rangle$
lmo(model :: ForwardModel, v :: Vector{Float64})  =
   error("lmo not implemented for model $(typeof(model)).")
#Return the tuple (v, g) where $v = \ell(r)$ and $g = \nabla \ell(r).$
loss(loss :: Loss, r :: Vector{Float64}) = error("loss not implemented for loss $(typeof(loss)).")
