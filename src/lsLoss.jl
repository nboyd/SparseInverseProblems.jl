export LSLoss, loss
## Represents a loss function for a linear inverse problem
# of the form loss(r) = ||r||_2
immutable LSLoss <: Loss end
#evaluates and returns the gradient of the loss w.r.t r
loss(l :: LSLoss,r :: Vector{Float64}) = norm(r,2), r/norm(r,2)
