"""
Author: cirobr@GitHub
Date: 18-Sep-2026

Template for training vision models.
"""

@info "*** Project $( basename(@__FILE__) ) started ***"
cd(@__DIR__)

### arguments
# envpath       = "../"
# cudadevice    = 0
# epochs        = 1
# minibatchsize = 1
# accum_steps   = 1
# debugflag     = true


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
using ProgressBars; const pb=ProgressBars
import Statistics: mean, minimum, maximum, norm, std, median

# private libs
using LibFluxML
import LibFluxML: acc_score, f1_score, iou_score, per_class_iou   # metrics
using PascalVocTools; const pv=PascalVocTools

using LibCUDA
LibCUDA.cleangpu()


# dataset constants
const imagesize = (500,500)   # original size
const framesize = (256,256)   # resized size
const classnrs  = pv.class_numbers[1:end-1]   # 0:20
const C = length(classnrs)


# custom functions
function get_image(path)
      return Images.load(expanduser(path)) .|> RGB{N0f8}        # original size
end

function get_mask(path)
      return Images.load(expanduser(path)) |> mask -> mask.index .|> Int16
end

function convert_image2tensor(img::AbstractMatrix{RGB{N0f8}})
      return Images.channelview(img) |>       # CHW
            x -> permutedims(x, (2,3,1)) |>   # HWC
            x -> (x .- means) ./ (stds) .|>   # per channel normalization
            Float32
end

function convert_mask2tensor(mask::AbstractMatrix{Int16})
      return LibFluxML.onehot_fast(mask, classnrs; ignore_index=255)
            # y -> Flux.label_smoothing(y, 0.05, dims=1) .|>
            # Float32
end

function get_normalization_params(x::AbstractArray{RGB{N0f8}})
      xf = Images.channelview(x)   # CHWN
      dims = (2,3,4)
      μs = mean(xf, dims=dims) |> x -> dropdims(x, dims=dims) .|> Float32
      σs = std(xf, dims=dims)  |> x -> dropdims(x, dims=dims) .|> Float32
      return μs, σs
end


# augmentation pipeline
intensity_trainpipe = Identity()
geometric_trainpipe = CenterResizeCrop(framesize)
intensity_validpipe = Identity()
geometric_validpipe = CenterResizeCrop(framesize)


function data_augmentation(
      img::AbstractArray{RGB{N0f8}},
      mask::AbstractArray{Int16};
      intensity_tfm,
      geometric_tfm
)
      # wrap
      img_wrap  = Image(img)
      mask_wrap = MaskMulti((mask .+ 1), 1:256)   # 0:255 .+ 1

      # augment
      img_wrap = apply(intensity_tfm, img_wrap)   # intensity (img only)
      img_wrap, mask_wrap = apply(geometric_tfm, (img_wrap, mask_wrap))   # geometric (img, mask)

      # unwrap
      img_unwrap  = img_wrap.data .|> RGB{N0f8}
      mask_unwrap = (mask_wrap.data .- 1) .|> Int16

      # remove index offsets
      img_unwrap  = OffsetArrays.no_offset_view(img_unwrap)
      mask_unwrap = OffsetArrays.no_offset_view(mask_unwrap)
      
      return img_unwrap, mask_unwrap
end
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


# get normalization parameters
Xs = Array{RGB{N0f8}}(undef, (framesize...,Ntrain))
ys = zeros(Int16, imagesize)
FLoops.@floop for i in 1:Ntrain
      img = get_image(expanduser(dftrain.X[i]))
      mask = ys
      img, mask = data_augmentation(img, mask;
                        intensity_tfm=intensity_validpipe,
                        geometric_tfm=geometric_validpipe
      )
      Xs[:,:,i] = img
end
means, stds = get_normalization_params(Xs)
Xs = nothing
ys = nothing

means = means .|> Float32 |> x->trunc.(x, digits=6)
stds  = stds  .|> Float32 |> x->trunc.(x, digits=6)
@show means
@show stds
means = reshape(means, (1,1,3))
stds  = reshape(stds,  (1,1,3))
@info "normalization parameters OK"


### (optional) calculate loss weights from class frequencies
xs = zeros(RGB{N0f8}, imagesize)
ys = Array{Bool,4}(undef, (framesize...,C,Ntrain))

FLoops.@floop for i in 1:Ntrain
      img  = xs   # dummy image
      mask = get_mask(expanduser(dftrain.y[i]))

      img, mask = data_augmentation(img, mask;
                                    intensity_tfm = intensity_validpipe,
                                    geometric_tfm = geometric_validpipe,
      )

      mask = LibFluxML.onehot_fast(mask, classnrs; ignore_index=255) |>
            y -> reshape(y, size(y)..., 1) .|> Bool
      ys[:,:,:,i] = mask
end

cs = LibFluxML.compute_class_counts(ys)
xs = nothing
ys = nothing
# @assert false

cs = cs[2:end]   # ignore background class
median_weights, inverse_weights, inverse_squared_weights, inverse_class_weights =
      LibFluxML.compute_class_weights(cs)

train_weights = median_weights .|> Float32 |> dev
@info "loss weights OK"


# data loading
struct CityscapesDataset
    df::DataFrame
    intensity_tfm
    geometric_tfm
end

# Interface
Flux.numobs(d::CityscapesDataset) = nrow(d.df)

function Flux.getobs(d::CityscapesDataset, i::Int)
    row = d.df[i, :]
    
    img  = get_image(expanduser(row.X))
    mask = get_mask(expanduser(row.y))
    
    img, mask = data_augmentation(img, mask;
                                  intensity_tfm = d.intensity_tfm,
                                  geometric_tfm = d.geometric_tfm)
    
    X = convert_image2tensor(img)
    y = convert_mask2tensor(mask)
    
    return (X, y)
end

# Batch version
function Flux.getobs(d::CityscapesDataset, idx::AbstractVector{<:Integer})
    [Flux.getobs(d, i) for i in idx]
end

# Training dataset (with heavy augmentation)
train_dataset = CityscapesDataset(dftrain, 
                              intensity_trainpipe, 
                              geometric_trainpipe
)

# Validation dataset (light/no augmentation)
valid_dataset = CityscapesDataset(dfvalid, 
                              intensity_validpipe,
                              geometric_validpipe
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

modelcpu = UNet(3, C; activation=leakyrelu)
# fpfn = "models/unet5_4/bestmodel.jld2"
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
function trainLossFunction(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return LibFluxML.dice_loss(yhat, y;
                  logits=true,
                  include_background=false,
                  exclude_voids=true,
                  reduction=:sum,
                  weights=train_weights,
                  device=dev,
      )
end

function validLossFunction(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return LibFluxML.iou_loss(yhat, y;
                  logits=true,
                  include_background=false,
                  exclude_voids=true,
                  reduction=:sum,
                  device=dev,
      )
end
@info "loss functions OK"


# optimizer & scheduler
η       = 1e-3
# final_η = 5e-5
β  = (0.9, 0.999)
# λ  = 1e-5
# cn = 1.0    # clip norm
# cg = 1.0    # clip grad

opt = OptimiserChain(
      Flux.AccumGrad(accum_steps),
      # Flux.ClipNorm(cn),
      # Flux.ClipGrad(cg),
      # Flux.AdamW(η, β, λ),
      Flux.Adam(η, β),
)
# opt_mp = Optimisers.MixedPrecision(Float16, opt)
optimizerState = Flux.setup(opt, model)
# Flux.freeze!(optimizerState.encoder)

# T1 = 200
# T2 = 500 - T1
# cosine_part   = CosAnneal(l0=η, l1=final_η, period=T1, restart=false)
# constant_part = final_η
# scheduler = Sequence(cosine_part => T1, constant_part => T2)
@info "optimizer OK"


# one-number metrics
function acc_score(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return acc_score(yhat, y; logits=true, exclude_voids=true, device=dev)
end

function f1_score(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return f1_score(yhat, y; logits=true, exclude_voids=true, device=dev)
end

function iou_score(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return iou_score(yhat, y; logits=true, exclude_voids=true, device=dev)
end

function mean_iou(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return per_class_iou(yhat, y; logits=true, exclude_voids=true, reduction=:mean, device=dev)
end

metrics = [
      acc_score,
      f1_score,
      iou_score,
      mean_iou,
]

# per-class metrics
function per_class_iou_score(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return per_class_iou(yhat, y; logits=true, exclude_voids=true, device=dev)
end
@info "metrics OK"


# tensorboard logger
logger = TBLogger(expanduser(tblogsfolder))
@info "tensorboard logger OK"


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
stop_monitor.number_since_best = 15
stop_monitor.patience = 5

# training loop
LibCUDA.cleangpu()
reset!(logger)

validlosses = []
for epoch in 1:epochs
# for (eta, epoch) in zip(scheduler, 1:epochs)
      LibCUDA.garbage_collection()

      @printf "*** Epoch %d/%d ***\n" epoch epochs
      # Flux.adjust!(optimizerState, eta)
      # @printf "Learning rate: %.3e \n\n" eta

      # train epoch
      trainloss = trainEpoch!(trainLossFunction, model, train_loader, optimizerState;
                        device=dev
      )
      @assert !isnan(trainloss) || error("training loss is NaN")
      @printf "Training loss: %.4f \n\n" trainloss

      # evaluate epoch
      validloss, validmetrics = evaluateEpoch(validLossFunction, model, valid_loader, metrics)
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
      losses = LibFluxML.evaluatePerClassMetric(model, valid_loader, per_class_iou_score)
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
