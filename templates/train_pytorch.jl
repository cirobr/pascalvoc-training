"""
Author: cirobr@GitHub
Date: 12-Nov-2026

Template for training vision models with PyTorch augmentation.

The solution fails.

The Python portion dispatches many parallel threads in the data loader
when parallel=true that are not properly handled by PythonCall. In result,
a segmentation failure occurs.

Code still needs debug for parallel=false.
"""

@info "*** Project $( basename(@__FILE__) ) started ***"
cd(@__DIR__)

### arguments
envpath     = "../"
cudadevice  = 0
epochs      = 500
train_batch = 4
accum_steps = 4
valid_batch = train_batch * accum_steps
debugflag   = true


### libs
using Pkg
Pkg.activate(expanduser(envpath))

using CUDA
using Flux, cuDNN
import Flux: gpu, cpu
dev = cpu
if CUDA.has_cuda_gpu()
      dev = gpu
      CUDA.device!(cudadevice)
      @info "gpu OK"
else
      @warn "gpu disabled: using cpu."
end

using TinyMachines
using Images
using DataAugmentation
using OffsetArrays
using DataFrames
using MLUtils
using ParameterSchedulers
import ParameterSchedulers: Sequence, CosAnneal
using Random
using CSV
using JLD2
using TensorBoardLogger
using FLoops
using Printf
using ProgressBars
import Statistics: mean, minimum, maximum, norm, std, median

# private libs
using LibFluxML
using PreprocessingImages; p=PreprocessingImages
using LibDataAug
import LibDataAug: apply_data_augmentation
using PascalVocTools; const pv=PascalVocTools
using LibCUDA

# python calls
homedir = ENV["HOME"] * "/"   # homedir = expanduser("~/") does not work
ENV["JULIA_PYTHONCALL_EXE"] = homedir * ".pyenv/versions/torch/bin/python"
ENV["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"  # common Linux path
using PythonCall
# using DLPack

torch              = pyimport("torch")
# torchvision        = pyimport("torchvision")
# models             = pyimport("torchvision.models.segmentation")
# feature_extraction = pyimport("torchvision.models.feature_extraction")
v2 = pyimport("torchvision.transforms.v2")
np = pyimport("numpy")
device = CUDA.has_cuda_gpu() ? torch.device("cuda") : torch.device("cpu")


# dataset constants
const imagesize = (500,500)   # original size
const framesize = (256,256)   # cropped size
const classnrs  = pv.class_numbers   # 0:20
const C_out = length(classnrs)
C_in = 3
# C_in = 1
colorflag =  C_in == 3 ? true : false

# imagenet constants
const imagenet_means = Float32[0.485, 0.456, 0.406]
const imagenet_stds  = Float32[0.229, 0.224, 0.225]


# custom functions
function get_image(path::String; colorflag::Bool=true)
      img = Images.load(expanduser(path))
      return colorflag ? RGB{N0f8}.(img) : Gray{N0f8}.(img)
end

# Pascal VOC mask PNG to integers conversion
to_ids(mask) = mask.index

function get_mask(path::String)
      return Images.load(expanduser(path)) |> mask->to_ids(mask) .|> Int16
end

function normalize_tensor(x::AbstractArray; means::AbstractVector, stds::AbstractVector)
      n = size(means)[1]
      @assert n ∈ [1,3]
      @assert n == size(stds)[1]

      if n == 3
            means = reshape(means, (1,1,3))
            stds  = reshape(stds,  (1,1,3))
      end
      
      return (x .- means) ./ (stds)
end

function convert_image_to_tensor(img)
      return p.image2Float32(img) # HWC
end

function convert_mask_to_tensor(mask)
      return Flux.onehotbatch(mask, (classnrs..., 255), 255) |> # CHW
            x->permutedims(x, (2,3,1)) |>                       # HWC
            x->x[:,:,1:end-1] .|>                               # remove 255/void class
            Bool
end

np = pyimport("numpy")
tv_tensors = pyimport("torchvision.tv_tensors")

function apply_pytorch_augmentation(img, mask, transform)
    # Image → (1, C, H, W) UInt8
    chw = channelview(img)
    chw8 = reinterpret(UInt8, chw)
    nchw = reshape(chw8, (1, size(chw)...))
    nchw_np = np.asarray(nchw)
    img_t = torch.from_numpy(nchw_np)
    img_tv = tv_tensors.Image(img_t)

    # Mask → (1, H, W) long, wrapped as tv_tensors.Mask
    mask_np = np.asarray(Int.(mask))
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(torch.long)
    mask_tv = tv_tensors.Mask(mask_t)

    # Apply transform (now both are transformed)
    img_out, mask_out = transform(img_tv, mask_tv)

    # Back to Julia
    img_chw = pyconvert(Array, img_out.numpy().squeeze(0))
    img_jl  = reinterpret(RGB{N0f8}, img_chw) |>
              x -> dropdims(x, dims=1) .|> RGB{N0f8}

    mask_jl = pyconvert(Array, mask_out.numpy().squeeze(0)) .|> Int16

    return img_jl, mask_jl
end

# images augmentation
# Training: geometric + photometric only
train_transform = v2.Compose([
    v2.RandomResizedCrop(
        size = pyconvert(Tuple, framesize),
        scale = (0.4, 2.0),
        ratio = (0.8, 1.25),
        antialias = true
    ),
    v2.RandomHorizontalFlip(p = 0.5),
    v2.RandomAffine(
        degrees = 15,
        translate = (0.1, 0.1),
        scale = (0.9, 1.1),
        shear = 10,
        fill = 0
    ),
    v2.RandomPerspective(distortion_scale = 0.2, p = 0.5),
    v2.ColorJitter(
        brightness = 0.4,
        contrast = 0.4,
        saturation = 0.4,
        hue = 0.15
    ),
    v2.RandomGrayscale(p = 0.1),
    v2.RandomPosterize(bits = 6, p = 0.1),
    v2.RandomSolarize(threshold = 0.5, p = 0.1),
    v2.RandomAdjustSharpness(sharpness_factor = 2, p = 0.3),
    v2.GaussianBlur(kernel_size = (5, 9), sigma = (0.1, 2.0)),
])

# Validation: deterministic center crop only
valid_transform = v2.Compose([
    v2.CenterCrop(size = pyconvert(Tuple, framesize)),
])
@info "environment OK"

# X = rand(RGB{N0f8},256,256)
# y = rand(classnrs, 256,256)
# Xout,yout = apply_pytorch_augmentation(X,y,valid_transform)
# @show size(Xout), typeof(Xout)
# @show size(yout), typeof(yout)
# @assert false

### folders
outputfolder  = basename(@__FILE__)[1:end-3] * "/"

datasetfolder = "../dataset/"

modelsfolder  = "./models/" * outputfolder
if isdir(modelsfolder)   rm(modelsfolder, force=true, recursive=true)   end
mkpath(expanduser(modelsfolder))

tblogsfolder  = "./tblogs/" * outputfolder
if isdir(tblogsfolder)   rm(tblogsfolder, force=true, recursive=true)   end
mkpath(expanduser(tblogsfolder))
@info "folders OK"


### data frames
df = CSV.read(expanduser(datasetfolder) * "dftrain.csv", DataFrame)
dftest = CSV.read(expanduser(datasetfolder) * "dftest.csv", DataFrame)

# split df into train and valid
Random.seed!(1234)
N = size(df, 1)
train_idx, val_idx = MLUtils.splitobs(1:N, at=0.7, shuffle=true)
dftrain = df[train_idx, :]
dfvalid = df[val_idx, :]
# dftrain = first(dftrain, 100)
# dfvalid = first(dfvalid, 50)
# @warn "train/valid split: $(size(dftrain,1)) train, $(size(dfvalid,1)) valid"

# debug mode
if debugflag
      dftrain = first(dftrain, 3)
      dfvalid = first(dfvalid, 2)
      train_batch = 1
      accum_steps = 1
      valid_batch = 1
      epochs  = 2
end
# end debug mode

Ntrain = size(dftrain, 1)
Nvalid = size(dfvalid, 1)
@info "dataset OK"

# calculate normalization parameters and class frequencies
@info "calculating parameters ..."
Xs = Array{Float32,4}(undef, (framesize...,C_in,Ntrain))
# img = get_image(expanduser(dftrain.X[1]); colorflag=colorflag)   # dummy image
ys = Array{Bool,4}(undef, (framesize...,C_out,Ntrain))

FLoops.@floop for i in 1:Ntrain
      img  = get_image(expanduser(dftrain.X[i]); colorflag=colorflag)
      mask = get_mask(expanduser(dftrain.y[i]))

      img, mask = apply_pytorch_augmentation(img, mask, valid_transform)

      Xs[:,:,:,i] = convert_image_to_tensor(img)
      ys[:,:,:,i] = convert_mask_to_tensor(mask)
end

# normalization statistics
means, stds = p.get_normalization_params(Xs)
means = means |> x->trunc.(x,digits=3) .|> Float32
stds  = stds  |> x->trunc.(x,digits=3) .|> Float32
# means = imagenet_means
# stds  = imagenet_stds

@show means
@show stds
@info "normalization parameters OK"

# class frequencies and weights
cs = LibFluxML.compute_class_counts(ys)
cs = cs[2:end]   # ignore background class
median_weights, inverse_weights, inverse_squared_weights, inverse_class_weights =
      LibFluxML.compute_class_weights(cs)

train_weights = median_weights .|> Float32 |> dev

Xs = nothing
ys = nothing
@info "loss weights OK"


# data loading
struct PascalVocDataset
    df::DataFrame
    transform          # Python transform
end

Flux.numobs(d::PascalVocDataset) = nrow(d.df)

function Flux.getobs(d::PascalVocDataset, i::Int)
    row = d.df[i, :]

    img  = get_image(expanduser(row.X); colorflag=colorflag)
    mask = get_mask(expanduser(row.y))

    # Python augmentation (geometric + photometric)
    img_aug, mask_aug = apply_pytorch_augmentation(img, mask, d.transform)

    # Julia tensor conversion + normalization (on-the-fly stats)
    X = convert_image_to_tensor(img_aug) |> x -> normalize_tensor(x; means=means, stds=stds)
    y = convert_mask_to_tensor(mask_aug)

    return (X, y)
end

function Flux.getobs(d::PascalVocDataset, idx::AbstractVector{<:Integer})
    [Flux.getobs(d, i) for i in idx]
end

train_dataset = PascalVocDataset(dftrain, train_transform)
valid_dataset = PascalVocDataset(dfvalid, valid_transform)

# data loaders
train_loader = Flux.DataLoader(train_dataset,
                        batchsize=train_batch,
                        buffer=true,
                        collate=true,
                        parallel=false,   # safe for PythonCall but slow
                        shuffle=true,
)

valid_loader = Flux.DataLoader(valid_dataset,
                        batchsize=valid_batch,
                        buffer=true,
                        collate=true,
                        parallel=false,   # safe for PythonCall but slow
                        shuffle=false,
)
@info "data loaders OK"


### model
modelcpu = TinyMachines.UNet(C_in, C_out; activation=leakyrelu)
# fpfn = "../2.baseline/models/dfl_2e-3_part2/bestmodel.jld2"
# LibFluxML.loadModelState!(fpfn, modelcpu)
# @info "model loaded from $fpfn"
model = modelcpu |> dev
@info "model OK"


# model checkpoint
function testfn(model, X, y)   # loss function for testing purposes only
      yhat = model(X)
      return Flux.mse(yhat, y)
end

(X,y) = first(train_loader)
X = X |> dev
y = y |> dev
loss = testfn(model, X, y)   # arguments (model, X, y) - follows Flux.train! >= 0.13.9
@assert !isnan(loss) || error("model checkpoint failure")
@info "model checkpoint OK"


# loss functions
lossfn(yhat, y) = LibFluxML.dfl(yhat, y;
                        logits=true,
                        include_background=false,
                        exclude_voids=true,
                        gamma=2.0,
                        focal_weight=1.0,
                        dice_weight =1.0,
                        weights=train_weights,
                        reduction=:mean,
                        device=dev,
)
function trainfn(model, X, y)
      yhat = model(X)
      return lossfn(yhat, y)
end

validfn(yhat, y) = LibFluxML.iou_loss(yhat, y;
                                    logits=true,
                                    include_background=false,
                                    exclude_voids=true,
                                    reduction=:mean,
                                    device=dev,
)
@info "loss functions OK"


# optimizer & scheduler
lr  = 5e-5
lr2 = 1e-4
lr3 = 1e-5
β   = (0.9, 0.999)
λ   = 5e-5   # or 1e-4
# cn = 1.0    # clip norm
# cg = 1.0    # clip grad

opt = OptimiserChain(
      Flux.AccumGrad(accum_steps),
      # Flux.ClipNorm(cn),
      # Flux.ClipGrad(cg),
      Flux.AdamW(lr, β, λ),
      # Flux.Adam(η, β),
)
# opt_mp = Optimisers.MixedPrecision(Float16, opt)
optimizerState = Flux.setup(opt, model)
# Flux.freeze!(optimizerState.backbone_head)

T1 = 8
T2 = 100
T3 = 500 - (T1+T2)
raising_part  = Triangle(l0=lr, l1=lr2, period=2*T1) # 1/2 period triangle
cosine_part   = CosAnneal(l0=lr2, l1=lr3, period=T2, restart=false)
constant_part = lr3
scheduler = Sequence(raising_part => T1, cosine_part => T2, constant_part => T3)
@info "optimizer OK"


# one-number metrics
acc_score(yhat,y) = LibFluxML.acc_score(yhat, y; logits=true, exclude_voids=true, device=dev)
f1_score(yhat,y)  = LibFluxML.f1_score(yhat, y; logits=true, exclude_voids=true, device=dev)
iou_score(yhat,y) = LibFluxML.iou_score(yhat, y; logits=true, exclude_voids=true, device=dev)
mean_iou(yhat,y)  = LibFluxML.per_class_iou(yhat, y; logits=true, exclude_voids=true, reduction=:mean,
                              device=dev)
validCE(yhat, y)  = LibFluxML.ce_loss(yhat, y; logits=true, include_background=false, exclude_voids=true,
                              weights=train_weights, reduction=:mean,
                              device=dev)
metrics = [
      acc_score,
      f1_score,
      iou_score,
      mean_iou,
      validCE,
]

# per-class metrics
per_class_iou_score(yhat,y) = LibFluxML.per_class_iou(yhat, y; logits=true, exclude_voids=true, device=dev)
@info "metrics OK"


# tensorboard logger
logger = TBLogger(expanduser(tblogsfolder))
@info "tensorboard logger OK"


# warmup training
@info "warmup training..."
loader = first(train_loader, 1) .|> dev
Flux.train!(model, loader, optimizerState) do m, x, y
      trainfn(m, x, y)
end
loader = nothing
@info "warmup OK"


###########################################
### training
###########################################
@info "start training ..."
println()

# model checkpoint & early stopping
model_monitor = LibFluxML.EarlyStopper()     # used for model saving only
model_monitor.number_since_best = 10*epochs  # not used
model_monitor.patience = 10*epochs           # not used

stop_monitor = LibFluxML.EarlyStopper()
stop_monitor.number_since_best = 20   # to bypass noise peaks
stop_monitor.patience = 5

# training loop
LibCUDA.cleangpu()
reset!(logger)

validlosses = []
# for epoch in 1:epochs
for (eta, epoch) in zip(scheduler, 1:epochs)
      LibCUDA.garbage_collection()
      # LibCUDA.cleangpu()

      @printf "*** Epoch %d/%d ***\n" epoch epochs
      Flux.adjust!(optimizerState, eta)
      @printf "Learning rate: %.3e \n\n" eta

      # train epoch
      trainloss = 
            trainEpoch!(trainfn, model, train_loader, optimizerState; device=dev)
      @assert !isnan(trainloss) || error("training loss is NaN")
      @printf "Training loss: %.4f \n\n" trainloss

      # evaluate epoch
      validloss, validmetrics = 
            evaluateEpoch(validfn, model, valid_loader, metrics; device=dev)
      @assert !isnan(validloss) || error("validation loss is NaN")
      @printf "Validation loss: %.4f \n" validloss

      push!(validlosses, validloss)
      avg_valid_loss = LibFluxML.last_n_average(validlosses, 3)
      @assert !isnan(avg_valid_loss) || error("average validation loss is NaN")
      @printf "Avg validation loss: %.4f \n" avg_valid_loss

      for metric in zip(metrics, validmetrics)
            metric_name  = nameof(metric[1])
            metric_value = metric[2]
            @printf "   %s: %.4f \n" metric_name metric_value
      end
      println()

      # per-class IoU
      losses = LibFluxML.evaluatePerClassMetric(model, valid_loader, per_class_iou_score; device=dev)
      println("Per-class IoU:")
      for (i, loss) in enumerate(losses)
            @printf "   IoU Class %d: %.4f \n" classnrs[i] loss
      end
      println()

      # log data
      Base.with_logger(logger) do
            @info "Loss/Training" train_loss=trainloss
            @info "Loss/Validation" valid_loss=validloss log_step_increment=0
            @info "Loss/Avg_Validation" avg_valid_loss=avg_valid_loss log_step_increment=0

            for (metric, value) in zip(metrics, validmetrics)   # log metrics
                  metric_name = nameof(metric)
                  @info "Metric/Validation/$metric_name" metric=value log_step_increment=0
            end
            
            for (i, loss) in enumerate(losses)   # log per-class IoU
                  class_name = "Class $(classnrs[i])"
                  @info "Metric/Validation/IoU/$class_name" metric=loss log_step_increment=0
            end
      end

      # model checkpoint & early stopping
      push!(model_monitor.validation_losses, validloss)
      _ = early_stop!(model_monitor)

      push!(stop_monitor.validation_losses, avg_valid_loss)
      earlystop = early_stop!(stop_monitor)

      if model_monitor.improved_validation
            LibFluxML.saveModelState(expanduser(modelsfolder) * "model.jld2", model)
      end
      if earlystop   break   end
end # epochs

close(logger)


# rename the best model state
mv(expanduser(modelsfolder) * "model.jld2", expanduser(modelsfolder) * "bestmodel.jld2", force=true)
@info "training OK"


# cleanup
modelcpu = nothing
model    = nothing
LibCUDA.cleangpu()
@info "*** Project $( basename(@__FILE__) ) finished ***"
println()