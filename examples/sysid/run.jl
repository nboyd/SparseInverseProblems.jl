include("sysid.jl")

if !isdir("data")
  print("Downloading data...")
  run(`./get-data.sh`)
  run(`mkdir output`)
  println("done.")
end

function runExperiment(filename,tau)
  const nTrain = 300
  const nTest = 700
  const nTotal = nTrain+nTest
  dat = readdlm("data/$filename.dat")
  const u  = vec(dat[:,1])[1:nTotal]
  const y  = vec(dat[:,2])[1:nTotal]

  const uTrain = u[1:nTrain]
  const yTrain = y[1:nTrain]
  const yTest = y[(nTrain+1):end]

  function normalization(r,theta)
    1.0-r*r
  end

  function dndr(r,theta)
    -2*r
  end

  function dndt(r,theta)
    0.0
  end

  sim = LinearSysID(uTrain,100,50,normalization,dndr,dndt)

  target = yTrain

  sim_t = LinearSysID(u,2,2,normalization,dndr,dndt)
  function fullResponse(t,w,u)
    response = phi(sim_t,t,w)
    return response[(nTrain+1):end]
  end
  global scores = zeros(0)

  function printRMSE(old_thetas, t,w,k,o)
    est = phi(sim,t,w)
    rmse = norm(vec(target) - vec(est))
    fR = fullResponse(t,w,u)
    testRMSE = norm(vec(yTest) - vec(fR))
    mean_test = mean(yTest)
    println()
    println(size(t,2), " atoms.")
    response = phi(sim_t,t,w)
    score = 100*(1.0 - testRMSE/norm(yTest.-mean_test))
    global scores
    scores = [scores; score]
    println("Score: ", score)
    println("Tau: ", sum(abs(w)))
    println("thetas: ", t)
    println("weights: ", w)
    println(" ")
    response = phi(sim,t,w)
    return false
  end

  #fwami
  global scores = zeros(0)
  result = ADCG(sim, LSLoss(), target, tau; max_iters = 15, callback = printRMSE)
  writecsv("output/$(filename)_fwami.csv",[["iter" "score"]; [1:length(scores) scores]])

  #fwa-m
  global scores = zeros(0)
  result = ADCG(sim, LSLoss(), target, tau; max_cd_iters=0, max_iters = 15, callback = printRMSE)
  writecsv("output/$(filename)_fwam.csv",[["iter" "score"]; [1:length(scores) scores]])

  #gradient flow
  global scores = zeros(0)
  result = ADCG(sim, LSLoss(), target, tau; max_cd_iters=1, max_iters = 15, callback = printRMSE)
  writecsv("output/$(filename)_gf.csv",[["iter" "score"]; [1:length(scores) scores]])
end

runExperiment("dryer",20.0)
runExperiment("robot_arm",47.0)
