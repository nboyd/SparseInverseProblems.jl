export nnlasso_kernel, lasso_kernel, nnlasso

function nnlasso(A,y,tau; x :: Vector{Float64} = tau*rand(size(A,2))/(size(A,2)*2), l = ones(size(A,2)+1)/(2*size(A,2)))
  K = full(A'*A);
  b = full(A'*y);
  return nnlasso_kernel(K,b,tau)
end

function nnlasso_kernel(K,b,tau; x :: Vector{Float64} = tau*rand(length(b))/(length(b)*2), l = ones(length(b)+1)/(2*length(b)))
  K = full(K)
  b = full(b)
  n = size(K,2)
  if (n == 0)
    return zeros(0)
  end

  function f_g_h(x)
    0.5*dot(x,K*x) - dot(x,b) ,K*x -b,K
  end
  p = OptimizationProblem(n,n+1,f_g_h,[-eye(n); ones(1,n)], [zeros(n);tau])
  x,l = primalDualSolve(p :: OptimizationProblem, x, l, 10.0, 1e-12 , 1e-12 ,1000, 0.1 , 0.4)
  x[x.<1e-9] = 0.0
  return x
end

function lasso_kernel(K,b,tau; x :: Vector{Float64} = tau*rand(length(b))/(length(b)*2), l = ones(2*length(b)+1)/(2*length(b)))
  K = full(K)
  b = full(b)
  n = size(K,2)
  if ( n == 0)
    return zeros(0)
  end
  function f_g_h(x)
    x = x[1:n]
    f,g,h = 0.5*dot(x,K*x) - dot(x,b) ,K*x -b,K
    f,[g; zeros(n)], [h zeros(n,n); zeros(n,n) zeros(n,n)]
  end
  consrt1 = [eye(n) -eye(n)]
  consrt2 = [-eye(n) -eye(n)]
  p = OptimizationProblem(2*n,2*n + 1,f_g_h,[consrt1;consrt2; [zeros(1,n) ones(1,n)]], [zeros(2*n);tau])
  x,l = primalDualSolve(p :: OptimizationProblem, [x; 1.1*abs(x)], l, 10.0, 1e-10 , 1e-10 ,1000, 0.1 , 0.4)
  x = x[1:n]
  x[abs(x).<1e-9] = 0.0
  return x
end
