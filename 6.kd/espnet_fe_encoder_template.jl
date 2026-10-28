"""
Author: cirobr@GitHub
Date: 03-Aug-2026

Template for front-end knowledge distillation of vision models.
"""

@info "*** Project $( basename(@__FILE__) ) started ***"
cd(@__DIR__)

### arguments
envpath       = "../"
cudadevice    = 1
epochs        = 500
minibatchsize = 2
accum_steps   = 4
debugflag     = false


### libs
using Pkg
Pkg.activate(expanduser(envpath))

using CUDA, cuDNN
using Flux
import Flux: gpu, cpu
dev = cpu
if CUDA.has_cuda_gpu()
      dev = gpu
      CUDA.device!(cudadevice)
      @info "gpu OK"
else
      @warn "gpu disabled: using cpu."
end

import Flux: kaiming_normal
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
# import LibFluxML:
#       gdfl, iou_loss,   # losses
#       acc_score, f1_score, iou_score, per_class_iou   # metrics
using PreprocessingImages; const p=PreprocessingImages
using CityscapesTools; const cst=CityscapesTools
include("functions.jl")

# python calls
homedir = ENV["HOME"] * "/"   # homedir = expanduser("~/") does not work
ENV["JULIA_PYTHONCALL_EXE"] = homedir * "pyenvs/torch/bin/python"
ENV["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"  # common Linux path
using PythonCall
# using DLPack

torch              = pyimport("torch")
torchvision        = pyimport("torchvision")
models             = pyimport("torchvision.models.segmentation")
feature_extraction = pyimport("torchvision.models.feature_extraction")
device = CUDA.has_cuda_gpu() ? torch.device("cuda") : torch.device("cpu")


# clean memory
using LibCUDA
LibCUDA.cleangpu()


# dataset constants
const imagesize  = (1024, 2048)   # original size
const scalefixed = (512, 1024)
const framesize  = (512, 512)
const classnrs = 0:18             # 19 classes without background, 255 is ignore class
const C = length(classnrs)
const class_to_idx = Dict(c => i for (i, c) in enumerate(classnrs))  # for onehot_fast


# custom functions
function get_image(path)
      return Images.load(expanduser(path)) .|> RGB{N0f8}        # original size
end

function get_mask(path)
      return Images.load(expanduser(path)) .|> Gray{N0f8} |>    # integer masks encoded as grayscale images
            x -> round.(x .* 255) .|> Int16 |>   # convert gray to integer classes
            x -> encode_labels(x) .|>   # encode labels to 0:18 + 255 (ignore)
            Int16
end

function convert_image2tensor(img::AbstractMatrix{RGB{N0f8}})
      return Images.channelview(img) |>       # CHW
            x -> permutedims(x, (2,3,1)) |>   # HWC
            x -> (x .- means) ./ (stds) .|>   # per channel normalization
            Float32   # Float32 standard case
end

### pytorch teacher model
# weights = models.DeepLabV3_ResNet101_Weights.DEFAULT
teachermodel = models.deeplabv3_resnet101(num_classes=C, aux_loss=true)

# load pretrained weights
model_path = "../teacher-deeplabv3/models/trainsum1/bestmodel.pth"
@assert isfile(model_path) || error("teacher file with pre-trained weights does not exist")
teachermodel.load_state_dict(torch.load(model_path, map_location="cpu"))

# choose the required feature layers
return_nodes = Dict(
      # "backbone.layer1" => "layer1",   # first encoder
      # "backbone.layer2" => "layer2",
      "backbone.layer3" => "layer3",
      # "backbone.layer4" => "layer4",   # last encoder
)

# create the extractor (works on the full pytorch model)
teachermodel = feature_extraction.create_feature_extractor(
      teachermodel,
      return_nodes = return_nodes
)

# ensure teachermodel is in inference mode
teachermodel.to(device)
teachermodel.eval()
###

function get_teacher_features(img::AbstractMatrix{RGB{N0f8}}, model=teachermodel)
      img_float = p.image2Float32(img) |>
                  x -> (x .- means) ./ (stds) .|>   # per channel normalization
                  Float32   # Float32 standard case

      feat = torch_predict_hwcn(
            img_float,
            model;
            get_features = output -> output["layer3"]
      )

      return dropdims(feat, dims=4)
end

function get_normalization_params(x::AbstractArray{RGB{N0f8}})
      xf = Images.channelview(x)   # CHWN
      dims = (2,3,4)
      μs = mean(xf, dims=dims) |> x -> dropdims(x, dims=dims) .|> Float32
      σs = std(xf, dims=dims)  |> x -> dropdims(x, dims=dims) .|> Float32
      return μs, σs
end

function compute_class_frequencies(y::AbstractArray)
      @assert ndims(y) == 4      # HWCN, one-hot encoded
      C = size(y, 3)
      @assert C > 1              # at least two classes
      dims = (1,2,4)
      fs = sum(y, dims=dims)     # sum over H,W,N
      return reshape(fs, C)
end


# augmentation pipeline
intensity_trainpipe =
      AdjustBrightness(0.5) |>
      AdjustContrast(0.5)
      
geometric_trainpipe = 
      Maybe(FlipX{2}()) |>
      Zoom() |>               # 1.0 - 1.2
      WarpAffine(0.2) |>      # random translation, shear and rotation
      RandomCrop(framesize)   # or RandomCrop(framesize)

intensity_validpipe = Identity()

geometric_validpipe = 
      ScaleFixed(scalefixed) |>
      CenterCrop(framesize)


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
dftrain = CSV.read(expanduser(datasetfolder) * "dftrain.csv", DataFrame)
dfvalid = CSV.read(expanduser(datasetfolder) * "dfval.csv", DataFrame)
dftrain = first(dftrain, 100)
dfvalid = first(dfvalid, 50)

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
# Xs = Array{RGB{N0f8}}(undef, (framesize...,Ntrain))
# FLoops.@floop for i in 1:Ntrain
#       Xs[:,:,i] = get_image(expanduser(dftrain.X[i])) |>
#             x -> p.center_pad(x, framesize)
# end

# means, stds = get_normalization_params(Xs)
# means = trunc.(means, digits=6)
# stds  = trunc.(stds, digits=6)
# Xs = nothing
# @show means
# @show stds

# cityscapes normalization parameters
means = [0.286895, 0.325133, 0.283891] .|> Float32   # pre-calculated as above
stds  = [0.051856, 0.036667, 0.051856] .|> Float32   # pre-calculated as above
means = reshape(means, (1,1,3))
stds  = reshape(stds, (1,1,3))
@info "normalization parameters OK"


### (optional) calculate loss weights from class frequencies
# xs = zeros(RGB{N0f8}, imagesize)
# ys = Array{Bool,4}(undef, (framesize...,C,Ntrain))

# FLoops.@floop for i in 1:Ntrain
#       img  = xs   # dummy image
#       mask = get_mask(expanduser(dftrain.y[i]))

#       img, mask = data_augmentation(img, mask;
#                                     intensity_tfm = intensity_validpipe,
#                                     geometric_tfm = geometric_validpipe,
#                                     # intensity_tfm = Identity(),
#                                     # geometric_tfm = Identity(),
#       )

#       mask = onehot_fast(mask, 255; dtype=Bool) |>
#             y -> reshape(y, size(y)..., 1) .|> Bool
#       ys[:,:,:,i] = mask
# end
# fs = compute_class_frequencies(ys)
# xs = nothing
# ys = nothing

# pre-calculated class frequencies for cityscapes trainset/validpipe
fs = [295707939, 20769252, 132722258, 2596889, 3494046, 7561022, 1327463,
      3564123, 115942140, 4522781, 44523823, 8667009, 1174485,
      36144488, 1743709, 1800587, 1585633, 553505, 2332488]

median_weights, inverse_weights, inverse_squared_weights, inverse_class_weights =
      LibFluxML.compute_class_weights(fs)

train_weights = inverse_class_weights .|> Float32 |> dev
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
      y = get_teacher_features(img, teachermodel)

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
                        parallel=false,   # parallel is not thread-safe with PythonCall
                        shuffle=true,
)

valid_loader = Flux.DataLoader(valid_dataset,
                        batchsize=minibatchsize * accum_steps,
                        buffer=true,
                        collate=true,
                        parallel=false,   # parallel is not thread-safe with PythonCall
                        shuffle=false,
)
@info "data loaders OK"


### model
# modelcpu = ESPNet(3, C; activation="prelu").encoder

# a) strategy for trainable alignment
modelcpu = Chain(
      ESPNet(3, C; activation="prelu").encoder,
      Conv((1,1), 256=>1024, stride=1, pad=0, bias=true, init=kaiming_normal()),
)

# b) exclusive strategy for non-trainable alignment
function align_features(student_feat, teacher_feat)   # (yhat, y)
      if size(student_feat)[1:2] != size(teacher_feat)[1:2]
            # PREFERRED, but div(0) error: Resize student features to teacher spatial size
            # student_feat = Flux.AdaptiveMeanPool(size(teacher_feat)[1:2])(student_feat)

            # Resize teacher features to student spatial size
            # teacher_feat = Flux.AdaptiveMeanPool(size(student_feat)[1:2])(teacher_feat)
      end
      return student_feat, teacher_feat
end |> dev

# c) pre-loaded weights
# fpfn = "models/unet5_4/bestmodel.jld2"
# LibFluxML.loadModelState!(fpfn, modelcpu)
# @info "model loaded from $fpfn"
model = modelcpu |> dev
@info "model OK"


# model checkpoint
function evaluate_model(model, X, y)
      yhat = model(X)
      yhat, y = align_features(yhat, y)
      return (yhat, y)
end

function lossfn(model, X, y)   # model -> student
      yhat, y = evaluate_model(model, X, y)
      return Flux.mse(yhat, y)   # loss function for testing purposes only
end |> dev

(X,y) = first(train_loader)
X = X |> dev
y = y |> dev
loss = lossfn(model, X, y)   # the model is the first argument (follows Flux.train! >= 0.13.9)
@assert !isnan(loss) || error("model checkpoint failure")
@info "model checkpoint OK"


# loss functions
T = 1   ### Distillation temperature ###
lossfn(yhat, y) = Flux.kldivergence(yhat, y)

function trainLossFunction(model, X, y)
      yhat, y = evaluate_model(model, X, y)
      return LibFluxML.softloss(yhat, y, lossfn; T=T, dims=3)
end |> dev

function validLossFunction(model, X, y)
      yhat, y = evaluate_model(model, X, y)
      return LibFluxML.softloss(yhat, y, lossfn; T=1, dims=3)
end |> dev
@info "loss functions OK"


# optimizer & scheduler
η       = 5e-4
final_η = 5e-5
β  = (0.9, 0.999)
λ  = 1e-4
cn = 0.9    # clip norm
cg = 1.0    # clip grad

opt = OptimiserChain(
      Flux.AccumGrad(accum_steps),
      Flux.ClipNorm(cn),
      # Flux.ClipGrad(cg),
      Flux.AdamW(η, β, λ),
)
# opt_mp = Optimisers.MixedPrecision(Float16, opt)
optimizerState = Flux.setup(opt, model)
# Flux.freeze!(optimizerState.encoder)

T1 = 200
T2 = 500 - T1
cosine_part   = CosAnneal(l0=η, l1=final_η, period=T1, restart=false)
constant_part = final_η
scheduler = Sequence(cosine_part => T1, constant_part => T2)
@info "optimizer OK"


# one-number metrics
function mae(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return Flux.mae(yhat, y)
end |> dev

function mse(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return Flux.mse(yhat, y)
end |> dev

function kldiv(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return Flux.kldivergence(yhat, y)
end |> dev

function soft_cos_loss(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return LibFluxML.softloss(yhat, y, LibFluxML.cosine_loss; T=1, dims=3)
end |> dev

function soft_mse(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return LibFluxML.softloss(yhat, y, Flux.mse; T=1, dims=3)
end |> dev

function soft_kldiv(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return LibFluxML.softloss(yhat, y, Flux.kldivergence; T=1, dims=3)
end |> dev

kldiv_mse_ratio(model,X,y)      = kldiv(model,X,y) / mse(model,X,y)
soft_kldiv_mse_ratio(model,X,y) = soft_kldiv(model,X,y) / soft_mse(model,X,y)

const ssim_conv = LibFluxML.make_ssim_conv(1024; kernel_size=11, σ=1.5f0) |> dev
function ssim(model,X,y)
      yhat, y = evaluate_model(model, X, y)
      return LibFluxML.ssim(yhat,y; SSIM_CONV=ssim_conv)
end |> dev

metrics = [
      mae,
      mse,
      # kldiv,
      soft_cos_loss,
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
stop_monitor.number_since_best = 10
stop_monitor.patience = 5

# training loop
LibCUDA.cleangpu()
reset!(logger)

# for epoch in 1:epochs
validlosses = []
for (eta, epoch) in zip(scheduler, 1:epochs)
      LibCUDA.garbage_collection()

      @printf "*** Epoch %d/%d ***\n" epoch epochs
      Flux.adjust!(optimizerState, eta)
      @printf "Learning rate: %.3e \n\n" eta

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
      avg_valid_loss = last_n_average(validlosses, 3)
      @assert !isnan(avg_valid_loss) || error("average validation loss is NaN")
      @printf "Avg validation loss: %.4f \n" avg_valid_loss

      for metric in zip(metrics, validmetrics)
            metric_name = nameof(metric[1])
            metric_value = round(metric[2], digits=4)
            @printf "   %s: %.4f \n" metric_name metric_value
      end
      println()

      # # per-class IoU
      # losses = LibFluxML.evaluatePerClassMetric(model, valid_loader, per_class_iou_score)
      # println("Per-class IoU:")
      # for (i, loss) in enumerate(losses)
      #       @printf "   IoU Class %d: %.4f \n" classnrs[i] loss
      # end
      # println()

      # model parameters
      ws = Flux.destructure(model)[1] |> cpu

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

            @info "Weights" weights=ws log_step_increment=0   # weights histogram
      end

      # model checkpoint & early stopping
      push!(model_monitor.train_losses, trainloss)
      push!(model_monitor.valid_losses, validloss)
      _ = early_stop!(model_monitor)

      push!(stop_monitor.train_losses, trainloss)
      push!(stop_monitor.valid_losses, avg_valid_loss)
      earlystop = early_stop!(stop_monitor)

      if model_monitor.improved_validation
            LibFluxML.saveModelState(expanduser(modelsfolder) * "model.jld2", model)
      end
      if earlystop   break   end
end # epochs

close(logger)


# rename the best model state
mv(expanduser(modelsfolder) * "model.jld2", expanduser(modelsfolder) * "best_fe_model.jld2", force=true)
@info "training OK"


# cleanup
modelcpu = nothing
model    = nothing
LibCUDA.cleangpu()
@info "*** Project $( basename(@__FILE__) ) finished ***"
println()