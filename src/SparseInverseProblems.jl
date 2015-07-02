module SparseInverseProblems
include("abstractTypes.jl")
include("BoxConstrainedDifferentiableModel.jl")
include("ADCG.jl")
include("lsLoss.jl")
module Util
include("util/ip.jl")
include("util/ip_lasso.jl")
end
end
