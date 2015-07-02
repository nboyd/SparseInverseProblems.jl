include("gaussblur.jl")

if !isdir("data")
  print("Downloading data...")
  run(`./get-data.sh`)
  run(`mkdir output`)
  println("done.")
end

using Images
const noise_mean = 0.0021172377176794793
const sigmasq = let
  lambda = 723.0
  NA = 1.4
  FWHM = lambda/(2*NA)
  sigma = FWHM/(2*log(2.0))
  (sigma/(64*100.0))^2
end

const gb_sim = GaussBlur2D(sigmasq,64,500)

function readImages(imageDir,nImages,n_pixels_x)
  imA = Array(Float64,n_pixels_x,n_pixels_x,nImages)
  for q = 1:nImages
    imageId = @sprintf("%05D",q)
    img = imread("$imageDir/$imageId.tif")
    imA[:,:,q] = float(data(img))
  end
  imA
end

function runFW(sim,imageArray; n_cd_iters :: Int64 = 200)
  nImages = size(imageArray,3)
  results = Array(Array{Float64},nImages)
  for imageIdx = 1:nImages
    img = imageArray[:,:,imageIdx]
    target = vec(img).-noise_mean
    function callback(old_thetas,thetas, weights,output,old_obj_val)
      #evalute current OV
      new_obj_val,t = loss(LSLoss(), output - target)
      if old_obj_val - new_obj_val < 7E-5
        return true
      end
      return false
    end
    (points,weights) = ADCG(gb_sim, LSLoss(), target, 200000.0; max_iters=20, callback=callback, max_cd_iters=n_cd_iters)
    results[imageIdx] = points
    print(".")
  end
  results
end

function writeCSV(filename,results)
  header = "Ground-truth,frame,xnano,ynano,znano,intensity\n"
  nImages = size(results,1)
  csvfile = open(filename,"w")
  write(csvfile,header)
  for frame = 1:nImages
    localizations = results[frame]
    nSources = size(localizations,2)
    for i = 1:nSources
      xPos = localizations[1,i]*6400.0
      yPos = localizations[2,i]*6400.0
      write(csvfile,"$i, $frame, $yPos,$xPos, 0.0,0.0\n")
    end
  end
  close(csvfile)
end

nImages = 100;

imageArray = readImages("data/sequence",nImages,64)


print("Running FWA-MI")
results = runFW(gb_sim,imageArray)
println(" done.")
writeCSV("output/fwami.csv",results)
##ablations...
print("Running FWA-M")
results = runFW(gb_sim,imageArray; n_cd_iters=0)
println(" done.")
writeCSV("output/fwam.csv",results)
print("Running GF")
results = runFW(gb_sim,imageArray; n_cd_iters=1)
println(" done.")
writeCSV("output/gf.csv",results)
