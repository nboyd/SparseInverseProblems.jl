include("matrixcompletion.jl")

if !isdir("data")
  print("Downloading data...")
  run(`./get-data.sh`)
  run(`mkdir output`)
  println("done.")
end

if !isfile("c/libmc.so.1.0")
  print("Compiling c library...")
  run(`./make-lib.sh`)
  println("done.")
end

train = readdlm("data/jellyfish/datasets/netflix-dataset/netflix_train_set.dat",' ', Int64; dims=(100198805, 3))
test = readdlm("data/jellyfish/datasets/netflix-dataset/netflix_probe_set.dat",' ', Int64; dims=(281702, 3))

println(size(train))
println("training on $(size(train,1)) examples, testing on $(size(test,1)).")

m,k = maximum(train,1)[1],maximum(train,1)[2]

test_sim = MatrixCompletion(m,k,size(test,1),test[:,1],test[:,2])
sim = MatrixCompletion(m,k,size(train,1),train[:,1], train[:,2])

n_train = size(train,1)

target = copy(float(train[:,3]))
train_mean = 3.35
target.-= train_mean

n_test = size(test,1)
test = vec(test[:,3])

const tau = 100000.0

start_time = time()
history = zeros(4,0)
function evalTestRMSE(old_points, points, weights,output, ov)
  test_out = runForwardSimulator(test_sim, points)*weights .+ train_mean
  test_out_clip = copy(test_out)
  test_out_clip[test_out_clip .>= 5.0] = 5.0
  test_out_clip[test_out_clip .<= 1.0] = 1.0
  println(sum(weights)," : ",tau)
  t_rmse = sqrt(sumabs2(test_out_clip - test)/n_test)
  println("clipped test RMSE: ", t_rmse)
  println("objective value on: ", ov)
  println("rank: ", length(weights))
  println("time: $(time() - start_time)")
  global history = [history [ov, t_rmse, length(weights), time() - start_time]]
  return false
end

#run with improvement
(t,w) = ADCG(sim, LSLoss(), target , tau;  callback = evalTestRMSE,max_iters = 20, max_cd_iters = 200)
writecsv("output/fwami.csv",history')

#ablations
#fwa-m
history = zeros(4,0)
(t,w) = ADCG(sim, LSLoss(), target , tau;  callback = evalTestRMSE,max_iters = 20, max_cd_iters = 0)
writecsv("output/fwam.csv",history')

#gf
history = zeros(4,0)
(t,w) = ADCG(sim, LSLoss(), target , tau;  callback = evalTestRMSE,max_iters = 20, max_cd_iters = 1)
writecsv("output/gf.csv",history')
