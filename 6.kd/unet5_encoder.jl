"""
Author: cirobr@GitHub
Date: 23-Oct-2026

Template for training vision models.
"""

@info "*** Project $( basename(@__FILE__) ) started ***"
cd(@__DIR__)

### arguments
envpath       = "../"
cudadevice    = 0
epochs        = 500
minibatchsize = 8
accum_steps   = 2
debugflag     = true


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

using Metalhead
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
import LibFluxML: acc_score, f1_score, iou_score, per_class_iou   # metrics
import PreprocessingImages: get_normalization_params
using LibDataAug
import LibDataAug: apply_data_augmentation
using PascalVocTools; const pv=PascalVocTools
include("../library/channel_augmentation_functions.jl")

using LibCUDA
LibCUDA.cleangpu()


# dataset constants
const imagesize = (500,500)   # original size
const framesize = (256,256)   # resized size
const classnrs  = pv.class_numbers   # 0:20
const C_out = 1024
C_in = 3
# C_in = 1
colorflag =  C_in == 3 ? true : false


# teacher model
# size(fs[1]) = (64, 64, 64, 1)
# size(fs[2]) = (64, 64, 256, 1)
# size(fs[3]) = (32, 32, 512, 1)
# unet encoder -> size(fs[4]) = (16, 16, 1024, 1)
# size(fs[5]) = (8, 8, 2048, 1)
teachermodel = Metalhead.backbone(ResNet(50; pretrain=true)) |> dev

# custom functions
function get_image(path::String; colorflag::Bool=true)
      img = Images.load(expanduser(path))
      return colorflag ? RGB{N0f8}.(img) : Gray{N0f8}.(img)
end

# Pascal VOC mask PNG to integers conversion
# to_ids(mask) = mask.index

function get_mask(x::AbstractArray)
      # return Images.load(expanduser(path)) |> mask->to_ids(mask) .|> Int16
      return Flux.activations(teachermodel, dev(x))[4] |> cpu
end


# images augmentation
const short_sides = 320:500
const crop_size = framesize[1]

random_crop = RandomCentricCrop(
      short_sides = short_sides,
      crop = crop_size,
)
class_crop = ClassCentricCrop(
      short_sides = short_sides,
      crop = crop_size,
)

intensity_trainpipe =
      Maybe(AdjustBrightness(0.15), 0.8) |>
      Maybe(AdjustContrast(0.15), 0.8)

geometric_trainpipe =
      Maybe(FlipX{2}(), 0.5) |>
      Maybe(Rotate(15), 0.5) |>
      OneOf([CenterResizeCrop(framesize), random_crop, class_crop], [0.1, 0.1, 0.8])
      # CenterCrop(framesize)

intensity_validpipe = Identity()

geometric_validpipe = CenterCrop(framesize)
@info "environment OK"


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
      minibatchsize = 1
      accum_steps = 1
      epochs  = 2
end
# end debug mode

Ntrain = size(dftrain, 1)
Nvalid = size(dfvalid, 1)
@info "dataset OK"


# calculate normalization parameters and class frequencies
@info "calculating parameters ..."
Xs = Array{Float32,4}(undef, (framesize...,C_in,Ntrain))
# ys = Array{Bool,4}(undef, (framesize...,C_out,Ntrain))

FLoops.@floop for i in 1:Ntrain
      img  = get_image(expanduser(dftrain.X[i]); colorflag=colorflag)
      # mask = get_mask(expanduser(dftrain.y[i]))
      
      img = apply_data_augmentation(img;
                        intensity_tfm = intensity_validpipe,
                        geometric_tfm = geometric_validpipe,
                        image_to_tensor_tfm  = ImageToTensor(),
                        # target_to_tensor_tfm = OneHot(),
      )
      Xs[:,:,:,i] = img
      # ys[:,:,:,i] = mask[:,:,1:C_out] .|> Bool
end

# normalization statistics
means, stds = get_normalization_params(Xs)
means = means |> x->trunc.(x,digits=3) .|> Float32
stds  = stds  |> x->trunc.(x,digits=3) .|> Float32

# imagenet statistics
# means = Float32[0.485, 0.456, 0.406]
# stds  = Float32[0.229, 0.224, 0.225]

@show means
@show stds
@info "normalization parameters OK"

# class frequencies and weights
# cs = LibFluxML.compute_class_counts(ys)
# cs = cs[2:end]   # ignore background class
# median_weights, inverse_weights, inverse_squared_weights, inverse_class_weights =
#       LibFluxML.compute_class_weights(cs)

# train_weights = median_weights .|> Float32 |> dev

Xs = nothing
ys = nothing
@info "loss weights OK"


# data loading
struct PascalVocDataset
    df::DataFrame
    intensity_tfm
    geometric_tfm
    image_to_tensor_tfm
#     target_to_tensor_tfm
end

# dataset interface
# σnoise = 0.02
image_to_tensor_train = ImageToTensor() |> Normalize(means, stds) #|> GaussianNoise(σnoise)
image_to_tensor_valid = ImageToTensor() |> Normalize(means, stds)
# mask_to_tensor = OneHot()

Flux.numobs(d::PascalVocDataset) = nrow(d.df)

function Flux.getobs(d::PascalVocDataset, i::Int)
      row = d.df[i, :]
      
      img  = get_image(expanduser(row.X); colorflag=colorflag)
      # mask = get_mask(expanduser(row.y))
    
      img = apply_data_augmentation(img;
                        intensity_tfm = d.intensity_tfm,
                        geometric_tfm = d.geometric_tfm,
                        image_to_tensor_tfm = d.image_to_tensor_tfm,
      )

      X = img
      y = get_mask( insertdims(img, dims=4) ) |> x->dropdims(x, dims=4)
      # y = mask[:,:,1:C_out] .|> Bool
      return (X, y)
end

# Batch version
function Flux.getobs(d::PascalVocDataset, idx::AbstractVector{<:Integer})
    [Flux.getobs(d, i) for i in idx]
end

train_dataset = PascalVocDataset(dftrain,
                              intensity_trainpipe,
                              geometric_trainpipe,
                              image_to_tensor_train,
                              # mask_to_tensor,
)

valid_dataset = PascalVocDataset(dfvalid,
                              intensity_validpipe,
                              geometric_validpipe,
                              image_to_tensor_valid,
                              # mask_to_tensor,
)

# data loaders
train_loader = Flux.DataLoader(train_dataset,
                        batchsize=minibatchsize,
                        buffer=true,
                        collate=true,
                        parallel=true,
                        shuffle=true,
)

valid_loader = Flux.DataLoader(valid_dataset,
                        batchsize=minibatchsize * accum_steps,
                        buffer=true,
                        collate=true,
                        parallel=true,
                        shuffle=false,
)
@info "data loaders OK"


### model
function evaluate_model(model, X, y)
      yhat = model(X)
      return (yhat, y)
end

modelcpu = TinyMachines.UNet(C_in, C_out).encoder
# fpfn = "../2.baseline/models/dfl_2e-3_part2/bestmodel.jld2"
# LibFluxML.loadModelState!(fpfn, modelcpu)
# @info "model loaded from $fpfn"
model = modelcpu |> dev
@info "model OK"


# model checkpoint
function lossfn(model, X, y)   # loss function for testing purposes only
      yhat, y = evaluate_model(model, X, y)
      return Flux.mse(yhat, y)
end

(X,y) = first(train_loader)
X = X |> dev
y = y |> dev
loss = lossfn(model, X, y)   # the model is the first argument (follows Flux.train! >= 0.13.9)
@assert !isnan(loss) || error("model checkpoint failure")
@info "model checkpoint OK"


# loss functions
"""
NOTE: T² factor applies to log-based losses such as kldivergence and cross-entropy, but not to mse.
The mse loss is already scaled by the temperature factor in the softmax function.
"""
T = 1   ### Distillation temperature: T=1 for encoder, T=3 or 4 for logits.

function trainLossFunction(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return (T*T) * LibFluxML.softloss(yhat, y, Flux.kldivergence; T=T, dims=3)
end

function validLossFunction(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return LibFluxML.softloss(yhat, y, Flux.kldivergence; T=1, dims=3)
end
@info "loss functions OK"


# optimizer & scheduler
lr  = 1e-4
lr2 = 2e-4
lr3 = 5e-5
β  = (0.9, 0.999)
λ  = 5e-5           # or 1e-4
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
# Flux.freeze!(optimizerState.encoder)

T1 = 8
T2 = 100
T3 = 500 - (T1+T2)
raising_part  = Triangle(l0=lr, l1=lr2, period=2*T1) # 1/2 period triangle
cosine_part   = CosAnneal(l0=lr2, l1=lr3, period=T2, restart=false)
constant_part = lr3
scheduler = Sequence(raising_part => T1, cosine_part => T2, constant_part => T3)
@info "optimizer OK"


# one-number metrics
function mae(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return Flux.mae(yhat, y)
end

function mse(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return Flux.mse(yhat, y)
end

function kldiv(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return Flux.kldivergence(yhat, y)
end

function soft_cos_loss(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return LibFluxML.softloss(yhat, y, LibFluxML.cosine_loss; T=1, dims=3)
end

function soft_mae(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return LibFluxML.softloss(yhat, y, Flux.mae; T=1, dims=3)
end

function soft_mse(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return LibFluxML.softloss(yhat, y, Flux.mse; T=1, dims=3)
end

function soft_kldiv(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return LibFluxML.softloss(yhat, y, Flux.kldivergence; T=1, dims=3)
end

kldiv_mse_ratio(model,X,y)      = kldiv(model,X,y) / mse(model,X,y)
soft_kldiv_mse_ratio(model,X,y) = soft_kldiv(model,X,y) / soft_mse(model,X,y)

const ssim_conv = LibFluxML.make_ssim_conv(C_out; kernel_size=11, σ=1.5f0) |> dev
function ssim(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return LibFluxML.ssim(yhat,y; SSIM_CONV=ssim_conv)
end

metrics = [
      mae,
      mse,
      # kldiv,
      soft_cos_loss,
      soft_mae,
      soft_mse,
      soft_kldiv,
      # kldiv_mse_ratio,
      soft_kldiv_mse_ratio,
      ssim,
]
@info "metrics OK"


# tensorboard logger
logger = TBLogger(expanduser(tblogsfolder))
@info "tensorboard logger OK"


# warmup training
@info "warmup training..."
loader = first(train_loader, 1) .|> dev
Flux.train!(model, loader, optimizerState) do m, x, y
      trainLossFunction(m, x, y)
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
            trainEpoch!(trainLossFunction, model, train_loader, optimizerState; device=dev)
      @assert !isnan(trainloss) || error("training loss is NaN")
      @printf "Training loss: %.4f \n\n" trainloss

      # evaluate epoch
      validloss, validmetrics = 
            evaluateEpoch(validLossFunction, model, valid_loader, metrics; device=dev)
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
      # losses = LibFluxML.evaluatePerClassMetric(model, valid_loader, per_class_iou_score)
      # println("Per-class IoU:")
      # for (i, loss) in enumerate(losses)
      #       @printf "   IoU Class %d: %.4f \n" classnrs[i] loss
      # end
      # println()

      # log data
      Base.with_logger(logger) do
            @info "Loss/Training" train_loss=trainloss
            @info "Loss/Validation" valid_loss=validloss log_step_increment=0
            @info "Loss/Avg_Validation" avg_valid_loss=avg_valid_loss log_step_increment=0

            for (metric, value) in zip(metrics, validmetrics)   # log metrics
                  metric_name = nameof(metric)
                  @info "Metric/Validation/$metric_name" metric=value log_step_increment=0
            end
            
            # for (i, loss) in enumerate(losses)   # log per-class IoU
            #       class_name = "Class $(classnrs[i])"
            #       @info "Metric/Validation/IoU/$class_name" metric=loss log_step_increment=0
            # end
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
teachermodel = nothing
modelcpu = nothing
model    = nothing
LibCUDA.cleangpu()
@info "*** Project $( basename(@__FILE__) ) finished ***"
println()
