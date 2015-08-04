using SparseInverseProblems
import SparseInverseProblems: getStartingPoint, parameterBounds, psi, dpsi

immutable SimpleExample <: BoxConstrainedDifferentiableModel
  evaluation_points :: Vector{Float64}
  grid_points :: Vector{Float64}
  grid :: Matrix{Float64}
  SimpleExample(p,grid) = new(p,grid,psf(grid',p))
end

psf(theta, points) = exp(-(points .- theta).^2/2.0)
deriv_psf(theta, points) = exp(-(points .- theta).^2/2.0).*(points .- theta)

psi(s :: SimpleExample, parameters :: Vector{Float64}) = psf(parameters,s.evaluation_points)

dpsi(s :: SimpleExample, parameters :: Vector{Float64}) = reshape(deriv_psf(parameters,s.evaluation_points),length(s.evaluation_points),1)

getStartingPoint(model :: SimpleExample, v :: Vector{Float64}) = [model.grid_points[indmin(model.grid'*v)]]

parameterBounds(model :: SimpleExample) = ([-Inf], [Inf])
