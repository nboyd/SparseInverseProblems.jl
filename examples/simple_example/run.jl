include("simple_example.jl")
using Gadfly, Compose
evaluation_points = -5.0:0.25:5.0
model = SimpleExample(evaluation_points, -10.0:0.5:10.0)
means = [-2.0,1.0,3.5]
k = length(means)
weights = ones(k)+randn(k)*0.1
weights[1] = 0.2
target = max(phi(model,means',weights) + randn(length(evaluation_points))*0.1,0.0);

function callback(old_thetas,thetas, weights,output,old_obj_val)
  #evalute current OV
  new_obj_val,t = loss(LSLoss(), output - target)
  println("gap = $(old_obj_val - new_obj_val)")
  if old_obj_val - new_obj_val < 1E-1
    return true
  end
  return false
end

(means_est,weights_est) = ADCG(model, LSLoss(), target,10000.0;  callback=callback)

#draw the observations alone
draw(SVG("observations.svg", 5inch, 2inch), plot(x=evaluation_points,y=target))

#draw the observations with the locations and weights of the true blurs
anno = Guide.annotation(
       compose(context(), circle([means,means], [zeros(k),weights], [2mm]), fill(nothing),
       stroke("blue")))
draw(SVG("truth.svg", 5inch, 2inch), plot(x=evaluation_points,y=target,anno))


#draw the estimation along with the predicted means/observations
anno = Guide.annotation(
       compose(context(), circle([means_est',means_est'], [zeros(k),weights_est], [2mm]), fill(nothing),
       stroke("red")))
output_est = phi(model, means_est, weights_est)
draw(SVG("est.svg", 5inch, 2inch), plot(x=evaluation_points, y=output_est,anno))
