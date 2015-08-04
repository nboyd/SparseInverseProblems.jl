export BoxConstrainedDifferentiableModel
using NLopt, SparseInverseProblems.Util
# A simple forward model with box constrained parameters.
# Assumes differentiablity of the forward operator.
#
# For concrete examples see examples/smi or examples/sysid.
#
abstract BoxConstrainedDifferentiableModel <: ForwardModel

#compute the measurement model
psi(model :: BoxConstrainedDifferentiableModel, theta :: Vector{Float64}) =
  error("psi not implemented for model $(typeof(model)).")

#compute the jacobian of the forward model
dpsi(model :: BoxConstrainedDifferentiableModel, theta :: Vector{Float64}) =
    error("dpsi not implemented for model $(typeof(model)).")

# Initial starting point for continuous optimization for the FW step.
# Should return a good guess for $\arg\min_\theta \langle \psi(theta), v \rangle.$
# Often computed using a grid.
getStartingPoint(model :: BoxConstrainedDifferentiableModel, v :: Vector{Float64}) =
  error("getStartingPoint not implemented for model $(typeof(model)).")

# Box constraints on the parameters.
# Returns a tuple of two vectors : lower bounds and upper bounds.
parameterBounds(model :: BoxConstrainedDifferentiableModel) =
  error("parameterBounds not implemented for model $(typeof(model)).")

#default implementation of phi in terms of psi --- you should probably overwride this!
phi(model :: BoxConstrainedDifferentiableModel, parameters :: Matrix{Float64}, weights :: Vector{Float64}) =
    sum([weights[i]*psi(model, vec(parameters[:,i])) for i = 1:size(parameters,2)])

# Computes $\nabla_{\theta_1,\ldots,\theta_k} \sum_i \langle w_i \psi(\theta_i), v \rangle.$
# Returns a matrix with the same shape as thetas (which is p by k).
# You should probably overwride this!
computeGradient(model :: BoxConstrainedDifferentiableModel, weights :: Vector{Float64},
  thetas :: Matrix{Float64}, v :: Vector{Float64}) =
  hcat([weights[i]*vec(dpsi(model, vec(thetas[:,i]))'*v) for i = 1:size(thetas,2)]...)

# Sets the parameters for the continuous optimizer.
# Can be overwridden.
function initializeOptimizer!(model :: BoxConstrainedDifferentiableModel, opt :: Opt)
  ftol_abs!(opt, 1e-6)
  xtol_rel!(opt, 0.0)
  maxeval!(opt, 200)
end

# Default implementation. Uses NLopt to do continuous optimization.
function lmo(model :: BoxConstrainedDifferentiableModel, v :: Vector{Float64})
  lb,ub = parameterBounds(model)
  initial_x = getStartingPoint(model, v)
  p = length(lb)

  function f_and_g!(point :: Vector{Float64}, gradient_storage :: Vector{Float64})
    output = phi(model,reshape(point,p,1), [1.0])
    ip = dot(output,v)
    s = sign(ip)
    gradient_storage[:] = -s*computeGradient(model, [1.0],reshape(point,length(point),1), v)
    return -s*ip
  end

  opt = Opt(:LD_MMA, p)
  initializeOptimizer!(model, opt)
  min_objective!(opt, f_and_g!)
  lower_bounds!(opt, lb)
  upper_bounds!(opt, ub)
  (optf,optx,ret) = optimize(opt, initial_x)
  return (optx,optf)
end

immutable SupportUpdateProblem
  nPoints :: Int64
  p :: Int64
  s :: BoxConstrainedDifferentiableModel
  y :: Vector{Float64}
  w :: Vector{Float64}
  lossFn :: Loss
end

# Default implementation. Uses NLopt to do continuous optimization.
function localDescent(s :: BoxConstrainedDifferentiableModel, lossFn :: Loss, thetas ::Matrix{Float64}, w :: Vector{Float64}, y :: Vector{Float64})
  lb,ub = parameterBounds(s)
  nPoints = size(thetas,2)
  p = size(thetas,1)
  su = SupportUpdateProblem(nPoints,p,s,y,w,lossFn)
  f_and_g!(x,g) = localDescent_f_and_g!(x,g,su)
  opt = Opt(NLopt.LD_MMA, length(thetas))
  initializeOptimizer!(s, opt)
  min_objective!(opt, f_and_g!)
  lower_bounds!(opt, vec(repmat(lb,1,nPoints)))
  upper_bounds!(opt, vec(repmat(ub,1,nPoints)))
  (optf,optx,ret) = optimize(opt, vec(thetas))
  return reshape(optx,p,nPoints)
end

function localDescent_f_and_g!(points :: Vector{Float64}, gradient_storage :: Vector{Float64}, s :: SupportUpdateProblem)
  points = reshape(points,s.p,s.nPoints)
  output = phi(s.s,points,s.w)
  residual = output - s.y
  l,v_star = loss(s.lossFn,residual)
  gradient_storage[:] = computeGradient(s.s, s.w, points, v_star)
  return l
end

# Only for squared loss
solveFiniteDimProblem(model :: BoxConstrainedDifferentiableModel, loss :: LSLoss, thetas :: Matrix{Float64}, y :: Vector{Float64}, tau :: Float64) = nnlasso(hcat([psi(model, vec(thetas[:,i])) for i = 1:size(thetas,2)]...),y,tau)
