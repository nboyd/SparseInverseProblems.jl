using SparseInverseProblems
import SparseInverseProblems: lmo, phi, solveFiniteDimProblem, localDescent
using SparseInverseProblems.Util

immutable MatrixCompletion <: ForwardModel
  m :: Int64
  k :: Int64
  nnz :: Int64
  nz_i :: Vector{Int64}
  nz_j :: Vector{Int64}
end

function solveFiniteDimProblem(model :: MatrixCompletion, loss :: Loss, parameters :: Matrix{Float64}, y :: Vector{Float64}, tau :: Float64)
  A = runForwardSimulator(model, parameters)
  return nnlasso_kernel(A'*A,A'y,tau)
end

function localDescent(sim :: MatrixCompletion, loss :: LSLoss, thetas :: Matrix{Float64}, w :: Vector{Float64}, y :: Vector{Float64})
  rank = size(thetas,2)
  m,k = sim.m,sim.k
  u = thetas[1:m,:]
  v = thetas[(m+1):end,:]
  #M = uv'
  # min ||Phi(u*v') -y ||^2
  # ||u_i|| \le sqrt(w[i])
  # ||v_i|| \le sqrt(w[i])
  #GD w/ backtracking linesearch on the sphere.
  alpha = 0.05
  beta = 0.5
  c_1 = 1E-4
    for iter = 1:1
      (f,g_u,g_v) = fg_c(u,v,y,w,sim) #50% of the time is spent here..
      #project gradient onto tangent plane
      h_u = g_u - scale(u,vec(sum(g_u.*u,1)))
      h_v = g_v - scale(v,vec(sum(g_v.*v,1)))
      #normalize
      norm_u = vec(sqrt(sumabs2(h_u,1)))
      norm_v = vec(sqrt(sumabs2(h_v,1)))
      scale!(h_u,1.0./norm_u)
      scale!(h_v,1.0./ norm_v)
      #BTLS
      while(true)
        if alpha < 1E-6
          break;
        end
        sw = sum(w)
        ca = cos(alpha)
        sa = sin(alpha)
        u_n = scale(u,ca) - scale(h_u,sa)
        v_n = scale(v,ca) - scale(h_v,sa)
        f_n = f_only_c(u_n,v_n,y,w,sim)
        #check for sufficient decrease
        if f_n <= f + c_1*dot(vec(u_n - u),vec(g_u)) + c_1*dot(vec(v_n -v),vec(g_v))
          break;
        end
        alpha = beta*alpha
      end
      if (alpha < 1E-6)
        println("small stepsize, breaking!")
        break;
      end
      u = u*cos(alpha) - h_u*sin(alpha)
      v = v*cos(alpha) - h_v*sin(alpha)
    end
  return [u;v]
end

function runForwardSimulator(s :: MatrixCompletion, parameters)
  u = parameters[1:s.m, :]
  v = parameters[s.m+1:end, :]
  rank = size(parameters,2)
  result = zeros(s.nnz,rank)
  # out_v = zeros(s.nnz)
  n = length(s.nz_i)
  for idx = 1:n
    @fastmath @inbounds @simd  for r = 1:rank
      result[idx,r] = u[s.nz_i[idx],r]*v[s.nz_j[idx],r]
    end
  end
  return result
end

function phi(s :: MatrixCompletion, parameters :: Matrix{Float64}, w :: Vector{Float64})
  u = parameters[1:s.m, :]
  v = parameters[s.m+1:end, :]
  rank = size(parameters,2)
  result = zeros(s.nnz)
  n = length(s.nz_i)
  for idx = 1:n
    @fastmath @inbounds  @simd for r = 1:rank
       result[idx] += w[r]*u[s.nz_i[idx],r]*v[s.nz_j[idx],r]
    end
  end
  return result
end

function lmo(s :: MatrixCompletion,vector :: Vector{Float64})
  M = sparse(s.nz_i,s.nz_j, vector)
  (u,sv,v,t1,t2,t3,t4) = svds(M; nsv = 5, tol = 1E-7)
  p = vec([-u[:,1];v[:,1]])
  o = runForwardSimulator(s,p)
  return (p, dot(vec(vector),vec(o)) )
end

function fg_c(u,v,y,w,simulator)
  rank = length(w)
  u=u'
  v=v'
  u_aa = Array{Array{Float64}}(simulator.m)
  v_aa = Array{Array{Float64}}(simulator.k)
  for i = 1:simulator.m
    u_aa[i] = copy(vec(u[:,i]))
  end
  for i = 1:simulator.k
    v_aa[i] = copy(vec(v[:,i]))
  end
  u = u_aa
  v = v_aa
  up = deepcopy(u)
  vp = deepcopy(v)
  f_val = ccall((:fg,"c/libmc.so.1.0"),Float64,(Ptr{Ptr{Float64}},Ptr{Ptr{Float64}},Ptr{Ptr{Float64}},Ptr{Ptr{Float64}},Ptr{Int64},Ptr{Int64},Ptr{Float64},Ptr{Float64},Int64,Int64),up,vp,u,v,sim.nz_i,sim.nz_j,w,y,length(sim.nz_j),rank)
  u = zeros(rank,simulator.m)
  v = zeros(rank,simulator.k)
  for i = 1:simulator.m
    u[:,i] = up[i]
  end
  for i = 1:simulator.k
    v[:,i] = vp[i]
  end
  return f_val, u',v'
end

function f_only_c(u, v,y,w ::Vector{Float64},simulator)
  rank = length(w)
  u=u'
  v=v'
  u_aa = Array{Array{Float64}}(simulator.m)
  v_aa = Array{Array{Float64}}(simulator.k)
  for i = 1:simulator.m
    u_aa[i] = copy(vec(u[:,i]))
  end
  for i = 1:simulator.k
    v_aa[i] = copy(vec(v[:,i]))
  end
  u = u_aa
  v = v_aa
  return ccall((:f_only,"c/libmc.so.1.0"),Float64,(Ptr{Ptr{Float64}},Ptr{Ptr{Float64}},Ptr{Int64},Ptr{Int64},Ptr{Float64},Ptr{Float64},Int64,Int64),u',v',sim.nz_i,sim.nz_j,w,y,length(sim.nz_j),rank)
end
