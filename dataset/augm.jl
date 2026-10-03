"""
Author: cirobr@GitHub
Date: 18-Sep-2026

Template for training vision models.
"""

@info "*** Project $( basename(@__FILE__) ) started ***"
cd(@__DIR__)

### arguments
envpath       = "../"
cudadevice    = 0
epochs        = 1
minibatchsize = 1
accum_steps   = 1
debugflag     = false


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
import LibFluxML:
      gdfl, iou_loss,   # losses
      acc_score, f1_score, iou_score, per_class_iou   # metrics
using PreprocessingImages; const p=PreprocessingImages
using PascalVocTools; const pv=PascalVocTools

using LibCUDA
LibCUDA.cleangpu()


# dataset constants
const imagesize  = (500,500)   # original size
const framesize  = (512, 512)
const classnrs = 0:20   # 0:20 + 255 (void)
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
      CenterCrop(framesize)   # or RandomCrop(framesize)

intensity_validpipe = Identity()

geometric_validpipe = 
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

datasetfolder = "./"

# modelsfolder  = "./models/" * outputfolder
# if isdir(modelsfolder)   rm(modelsfolder, force=true, recursive=true)   end
# mkpath(expanduser(modelsfolder))

# tblogsfolder  = "./tblogs/" * outputfolder
# if isdir(tblogsfolder)   rm(tblogsfolder, force=true, recursive=true)   end
# mkpath(expanduser(tblogsfolder))
# @info "folders OK"


### data frames
df = CSV.read(expanduser(datasetfolder) * "dftrain.csv", DataFrame)
dftest   = CSV.read(expanduser(datasetfolder) * "dftest.csv", DataFrame)

# split df into train and valid
Random.seed!(1234)
N = size(df, 1)
train_idx, val_idx = MLUtils.splitobs(1:N, at=0.7, shuffle=true)
dftrain = df[train_idx, :]
dfvalid = df[val_idx, :]
# dftrain = first(dftrain, 100)
# dfvalid = first(dfvalid, 50)

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
fs = compute_class_frequencies(ys)
xs = nothing
ys = nothing
# @assert false

median_weights, inverse_weights, inverse_squared_weights, inverse_class_weights =
      LibFluxML.compute_class_weights(fs)

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

# check augmented images
k = rand(1:Ntrain)
X, y = Flux.getobs(train_dataset, k)
X = (X .* stds) .+ means
img = p.array2image(X)