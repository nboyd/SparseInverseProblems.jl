using SparseInverseProblems
import SparseInverseProblems: lmo, phi, solveFiniteDimProblem, localDescent,
getStartingPoint, parameterBounds, computeGradient
using SparseInverseProblems.Util
const min_theta = 0.001
const max_theta = pi
const min_r = 1E-3
const max_r = 0.999
const max_x = 1.0
const max_b = 1.0
immutable LinearSysID <: BoxConstrainedDifferentiableModel
  u :: Vector{Float64}  #u is the input time series
  grid
  normalization :: Function # (r,theta) -> n(r,theta)
  dndr :: Function # (r,theta) -> dn/dr
  dndt :: Function # (r,theta) -> dn/dt
  LinearSysID(u, gr,gt,n,dndr,dndt) = new(u, grid(gr,gt,u),n,dndr,dndt)
end

function getStartingPoint(model :: LinearSysID, v :: Vector{Float64})
  (gridR, gridT, l_x, l_b)  = model.grid

  scores = zeros(length(gridR), length(gridT))
  for i = 1:length(gridR)
    for j = 1:length(gridT)
      r,theta = gridR[i], gridT[j]
      nf = model.normalization(r,theta)
      scores[i,j] = nf*(max_x*sumabs(v'*l_x[i,j]) + max_b*sumabs(v'*l_b[i,j]))
    end
  end
  ind = indmax(scores)
  (rind,tind) = ind2sub((length(gridR), length(gridT)),ind)
  best_x = -max_x * sign(v'*l_x[ind])
  best_b = -max_b * sign(v'*l_b[ind])
  gridP = [gridR[rind]; gridT[tind]; vec(best_x); vec(best_b)]

  return gridP
end

parameterBounds(model :: LinearSysID) =
  ([min_r, min_theta,-max_x,-max_x, -max_b, -max_b],
  [max_r, max_theta,max_x,max_x, max_b, max_b])

function computeGradient(model :: LinearSysID, weights :: Vector{Float64},
  thetas :: Matrix{Float64}, v :: Vector{Float64})
  gradient = zeros(thetas)
  for i = 1:size(thetas,2)
    gradient[:,i] = weights[i]*jacobian(model, vec(thetas[:,i]))* v
  end
  return gradient
end

function phi(sim :: LinearSysID,parameters :: Matrix{Float64}, weights :: Vector{Float64})
  return runSim(parameters,sim.u,sim.normalization)*weights
end

function runSim(atoms,u,n)
  nAtoms = size(atoms,2)
  results = zeros(length(u),nAtoms)
  for i = 1:nAtoms
    w = vec(atoms[:,i])
    r,t,x,b = unpackParameters(w)
    l_x,l_b = getLinearOps(r,t,u)
    results[:,i] = (l_x*x + l_b*b)*n(r,t)
  end
  return results
end

function getLinearOps(r,theta,u; l_x :: Matrix{Float64} = zeros(length(u),2),l_b :: Matrix{Float64} = zeros(length(u),2) )
  n = length(u)
  rt = r.^(1:n)
  ttheta = (1:n)*theta
  ct = cos(ttheta)
  st = sin(ttheta)
  @inbounds for t = 1:n
    l_x[t,1] = rt[t]*ct[t]
    l_x[t,2] = -rt[t]*st[t]
     @inbounds for tau = 1:t-2 #u[0] = 0
      flipped_t = t-1-tau
      c = u[tau]*rt[flipped_t]
      l_b[t,1] += c*ct[flipped_t]
      l_b[t,2] -= c*st[flipped_t]
    end
  end
  @inbounds for t = 2:n
    l_b[t,1] += u[t-1] # flipped_t = 0
  end
  return (l_x, l_b)
end

function jacobian(s :: LinearSysID, w :: Vector{Float64}; j_r :: Vector{Float64} = zeros(length(s.u)), j_theta :: Vector{Float64} = zeros(length(s.u)), l_x :: Matrix{Float64} = zeros(length(s.u),2), l_b :: Matrix{Float64} = zeros(length(s.u),2))
  r,theta,x,b = unpackParameters(w)
  n = length(s.u)
  u :: Vector{Float64} = s.u
  rt = r.^(1:n)
  ttheta = (1:n)*theta
  ct = cos(ttheta)
  st = sin(ttheta)

  @inbounds for t = 1:n
    l_x[t,1] = rt[t]*ct[t]
    l_x[t,2] = -rt[t]*st[t]
    j_theta[t] -= t*rt[t]*st[t]*x[1]
    j_theta[t] -= t*rt[t]*ct[t]*x[2]
     for tau = 1:t-2 #u[0] = 0
      flipped_t = t-1-tau
      c = u[tau]*rt[flipped_t]
      l_b[t,1] += c*ct[flipped_t]
      l_b[t,2] -= c*st[flipped_t]
      j_theta[t] -= u[tau]*flipped_t*rt[flipped_t]*st[flipped_t]*b[1]
      j_theta[t] -= u[tau]*flipped_t*rt[flipped_t]*ct[flipped_t]*b[2]
    end
     for tau = 1:t-3 #u[0] = 0
      flipped_t = t-1-tau
      j_r[t] += u[tau]*flipped_t*rt[flipped_t-1]*ct[flipped_t]*b[1]
      j_r[t] -= u[tau]*flipped_t*rt[flipped_t-1]*st[flipped_t]*b[2]
    end
  end
  @inbounds for t = 2:n
    j_r[t] += t*rt[t-1]*ct[t]*x[1]
    j_r[t] -= t*rt[t-1]*st[t]*x[2]
    l_b[t,1] += u[t-1] # flipped_t = 0
  end
  #normalization!
  nf = s.normalization(r,theta)
  o = l_x*x + l_b*b
  j_r = j_r*nf + o*s.dndr(r,theta)
  j_theta = j_theta*nf + o*s.dndt(r,theta)
  return [j_r j_theta l_x*nf l_b*nf]'
end

function grid(nr,nt,u)
  gridR = max_r * sqrt(linspace(min_r,1.0,nr))
  gridT = linspace(min_theta,max_theta,nt)
  l_x = Array(Matrix{Float64},nr,nt)
  l_b = Array(Matrix{Float64},nr,nt)
  for i = 1:nr, j = 1:nt
    r = gridR[i]
    theta = gridT[j]
    (l_x[i,j],l_b[i,j]) = getLinearOps(r,theta,u)
  end
  return (gridR, gridT, l_x,l_b)
end

unpackParameters(p) = (p[1],p[2],p[3:4],p[5:6]) #r, t, x_0, B

function solveFiniteDimProblem(sim :: LinearSysID, lossFn :: LSLoss, thetas :: Matrix{Float64}, y :: Vector{Float64}, tau :: Float64)
  A = runSim(thetas,sim.u,sim.normalization)
  return lasso_kernel(A'*A,A'*y,tau)
end
