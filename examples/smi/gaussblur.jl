using SparseInverseProblems
import SparseInverseProblems: lmo, phi, solveFiniteDimProblem, localDescent,
getStartingPoint, parameterBounds, computeGradient
using SparseInverseProblems.Util

immutable GaussBlur2D <: BoxConstrainedDifferentiableModel
  sigma :: Float64
  n_pixels :: Int64
  grid :: Vector{Float64} #this is small now.
  grid_f
  u :: Vector{Float64}
  l :: Vector{Float64}
  GaussBlur2D(s,np,ng) = new(sqrt(s),np, linspace(0.0,1.0,ng),
    computeFs(linspace(0.0,1.0,ng),np,sqrt(s)),
    zeros(np), zeros(np))
end

function getStartingPoint(model :: GaussBlur2D, v :: Vector{Float64})
  v = reshape(v, model.n_pixels, model.n_pixels)
  ng = length(model.grid)
  grid_objective_values = vec(model.grid_f'*v*model.grid_f)
  best_point_lin_idx = indmin(grid_objective_values)
  best_point_idx = ind2sub((ng,ng), best_point_lin_idx)
  best_grid_score = grid_objective_values[best_point_lin_idx]
  return [model.grid[best_point_idx[2]];model.grid[best_point_idx[1]]]
end

parameterBounds(model :: GaussBlur2D) = ([0.0,0.0],[1.0,1.0])

function computeGradient(model :: GaussBlur2D, weights :: Vector{Float64},
  thetas :: Matrix{Float64}, v :: Vector{Float64})
  v = reshape(v, model.n_pixels, model.n_pixels)

  gradient = zeros(thetas)
  #allocate temporary variables...
  f_x = zeros(model.n_pixels)
  f_y = zeros(f_x)
  fpy = zeros(f_x)
  fpx = zeros(f_x)
  v_x = zeros(model.n_pixels)
  v_y = zeros(model.n_pixels)
  v_yp = zeros(model.n_pixels)

  #compute gradient
  for i = 1:size(thetas,2)
    point = vec(thetas[:,i])
    computeFG(model, point[1], f_x, fpx)
    computeFG(model, point[2], f_y, fpy)
    v_x = A_mul_B!(v_x,v,f_x)
    v_y = At_mul_B!(v_y,v,f_y)
    v_yp = At_mul_B!(v_yp,v,fpy)
    function_value = dot(f_y, v_x)
    g_x = dot(v_y, fpx)
    g_y = dot(v_x, fpy)

    gradient[:,i] = weights[i]*[g_x; g_y]
  end
  return gradient
end

##not optimized.
function computeFs(x,n_pixels, sigma; result :: Matrix{Float64}  = zeros(n_pixels,length(x)))
  ng = length(x)
  z = zeros(n_pixels)
  u = zeros(n_pixels)
  l = zeros(n_pixels)
  for i = 1:ng
    computeF!(z,sigma,u,l,x[i])
    for j = 1:n_pixels
      @inbounds result[j,i] = z[j]
    end
  end
  return result
end
function computeFGs!(s :: GaussBlur2D,x,f,g)
  n_pixels = s.n_pixels
  ng = length(x)
  fz = zeros(n_pixels)
  gz = zeros(n_pixels)
  for i = 1:ng
    computeFG(s,x[i],fz,gz)
    @inbounds @simd for j= 1:length(fz)
      f[j,i] = fz[j]
      g[j,i] = gz[j]
    end
  end
end
function computeF!(f,sigma, u,l, x :: Float64)
  n_pixels = length(u)
  inc = 1.0/(sigma*n_pixels)
  xovers = x/sigma
  const prefactor = sigma * sqrt(pi) / 2.0
  @fastmath @inbounds @simd for i = 1:n_pixels
    u[i] = i*inc - xovers
    l[i] = (i-1)*inc - xovers
  end
  @fastmath @inbounds @simd for i = 1:n_pixels
    f[i] = prefactor*(erf(u[i]) - erf(l[i]))
  end
  return f
end

function computeFG(s :: GaussBlur2D, x :: Float64, f :: Vector{Float64}  = zeros(s.n_pixels), g :: Vector{Float64}  = zeros(s.n_pixels))
  n_pixels = s.n_pixels
  u = s.u
  l = s.l
  inc = 1.0/(s.sigma*n_pixels)
  xovers = x/s.sigma
  const prefactor = s.sigma * sqrt(pi) / 2.0

  @fastmath @inbounds @simd for i = 1:n_pixels
    u[i] = i*inc - xovers#(i - x)/s.sigma
    l[i] = (i-1)*inc - xovers
  end

  #might wanna just bite the bullet and allocate here...
  @fastmath @inbounds @simd  for i = 1:n_pixels
    f[i] = prefactor*(erf(u[i]) - erf(l[i]))
  end

  @fastmath @inbounds @simd for i = 1:n_pixels
    exp_u = exp(-u[i]*u[i])
    exp_l = exp(-l[i]*l[i])
    g[i] = exp_l - exp_u
  end
end

function phi(s :: GaussBlur2D, parameters :: Matrix{Float64},weights :: Vector{Float64})
  n_pixels = s.n_pixels
  if size(parameters,2) == 0
    return zeros(n_pixels*n_pixels)
  end
  v_x = computeFs(vec(parameters[1,:]),n_pixels,s.sigma)
  v_y = computeFs(vec(parameters[2,:]),n_pixels,s.sigma)
  scale!(v_x, weights)
  return vec(v_y*v_x')
end

function solveFiniteDimProblem(model :: GaussBlur2D, loss :: Loss, thetas :: Matrix{Float64}, y :: Vector{Float64}, tau :: Float64)
  nPoints = size(thetas,2)
  y = reshape(y,model.n_pixels,model.n_pixels)
  t_vecs = Any[zeros(model.n_pixels,nPoints) for i = 1:4]
  computeFGs!(model,vec(thetas[1,:]),t_vecs[1:2]...)
  computeFGs!(model,vec(thetas[2,:]),t_vecs[3:4]...)
  K_x = (t_vecs[1]'*t_vecs[1])
  K_y = (t_vecs[3]'*t_vecs[3])
  K = K_x.*K_y
  AtY = vec(sum((t_vecs[3]'*y).*(t_vecs[1]'),2))
  return nnlasso_kernel(K,AtY, tau)
end
