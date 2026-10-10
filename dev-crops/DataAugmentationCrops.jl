"""
Customized projective crops.

The caller supplies an image and a class-id matrix.

The window is `Crop(sz, FromRandom())`. Its `randstate` is a fractional
offset, shared by the photo (`Image`) and the mask (`MaskMulti`). The mask is
not wrapped as `Image`: a projective warp of an `Image` interpolates and mixes
class ids. `MaskMulti` is warped with nearest neighbor.
"""

using DataAugmentation
import DataAugmentation: apply, getrandstate, itemdata, OneOf

const IGNORE = 0xff

"""
    crop_offsets(len, crop, center)

Fractional offset consumed by `Crop(sz, FromRandom())`.

`offsetcropbounds` places the window at
`floor(firstindex + (len - crop + 1) * offset)`.
"""
function crop_offsets(len::Integer, crop::Integer, center::Integer)
    slack = len - crop + 1
    slack <= 1 && return 0.0
    top = clamp(center - crop ÷ 2, 1, slack)
    return (top - 1) / slack
end

function crop_offsets(labels::AbstractMatrix{<:Integer}, crop::NTuple{2,Integer}, cy::Integer, cx::Integer)
    return (
        crop_offsets(size(labels, 1), crop[1], cy),
        crop_offsets(size(labels, 2), crop[2], cx),
    )
end


"""
    class_center(labels; ignore=0xff, background=0) -> (cy, cx) or nothing

`labels` is the class-id matrix. Draw a class uniformly, excluding
background and void, then draw a pixel of that class.
"""
function class_center(labels::AbstractMatrix{<:Integer}; ignore::Integer = IGNORE, background::Integer = 0)
    present = Int[]
    for c in unique(labels)
        (c == background || c == ignore) && continue
        push!(present, Int(c))
    end
    isempty(present) && return nothing
    cls = present[rand(1:length(present))]
    idx = findall(==(cls), labels)
    p = idx[rand(1:length(idx))]
    return p[1], p[2]
end


"""
    class_center_state(labels, crop; kwargs...) -> NTuple{2,Float64}

`randstate` for `Crop(crop, FromRandom())`. Falls back to a uniform offset
when the label matrix has no foreground.
"""
function class_center_state(labels::AbstractMatrix{<:Integer}, crop::NTuple{2,Integer}; kwargs...)
    hit = class_center(labels; kwargs...)
    hit === nothing && return (rand(), rand())
    return crop_offsets(labels, crop, hit[1], hit[2])
end
class_center_state(labels, crop::Integer; kwargs...) = class_center_state(labels, (crop, crop); kwargs...)



struct ClassCentricCrop <: Transform
    sides::Vector{Int}
    crop::NTuple{2,Int}
    ignore::Int
    background::Int
end

struct RandomCentricCrop <: Transform
    sides::Vector{Int}
    crop::NTuple{2,Int}
end

"""
    ClassCentricCrop(; short_sides=320:500, crop=256)

Scale, then a class-center `Crop`. The mask item already holds class ids.
No foreground falls back to a uniform offset.
"""
function ClassCentricCrop(;
        short_sides = 320:500,
        crop::Integer = 256,
        ignore::Integer = IGNORE,
        background::Integer = 0,
)
    return ClassCentricCrop(
        collect(Int, short_sides),
        (Int(crop), Int(crop)),
        Int(ignore),
        Int(background),
    )
end

"""
    RandomCentricCrop(; short_sides=320:500, crop=256)

Scale, then a uniform `Crop(sz, FromRandom())`.
"""
function RandomCentricCrop(; short_sides = 320:500, crop::Integer = 256)
    return RandomCentricCrop(collect(Int, short_sides), (Int(crop), Int(crop)))
end

function _scaled(items, sides)
    side = sides[rand(1:length(sides))]
    scale = ScaleKeepAspect((side, side))
    return apply(scale, items[1]), apply(scale, items[2])
end

function _crop(items, crop, offsets)
    window = Crop(crop, DataAugmentation.FromRandom())
    cropped = (
        apply(window, items[1]; randstate = offsets),
        apply(window, items[2]; randstate = offsets),
    )
    return apply(PinOrigin(), cropped[1]), apply(PinOrigin(), cropped[2])
end

function apply(tfm::ClassCentricCrop, items::Tuple{Image,MaskMulti}; randstate = nothing)
    scaled = _scaled(items, tfm.sides)
    labels = itemdata(scaled[2]) .- 1
    offsets = class_center_state(labels, tfm.crop; ignore = tfm.ignore, background = tfm.background)
    return _crop(scaled, tfm.crop, offsets)
end

function apply(tfm::RandomCentricCrop, items::Tuple{Image,MaskMulti}; randstate = nothing)
    return _crop(_scaled(items, tfm.sides), tfm.crop, (rand(), rand()))
end

function apply(tfm::Union{ClassCentricCrop,RandomCentricCrop}, item::Item; randstate = nothing)
    error("apply the crop to an (Image, MaskMulti) pair")
end

# Sequence passes the pair to each step. Maybe is an OneOf, and the default
# tuple method would then apply the chosen crop to the image alone.
function apply(tfm::OneOf, items::Tuple{Image,MaskMulti}; randstate = getrandstate(tfm))
    i, inner = randstate
    return apply(tfm.tfms[i], items; randstate = inner)
end
