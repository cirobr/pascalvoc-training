"""
Projective crops with a dataset-supplied mask conversion.

Pascal VOC and Cityscapes both store the mask as an image, but the class ids
are not in the same place. The crop does not know that. The caller passes
`ids`, and only that function changes between datasets.

    pascal(mask) = mask.index
    cityscapes(mask) = mask.index          # or whatever that loader exposes
    already_ids(mask) = mask

The window is `Crop(sz, FromRandom())`. Its `randstate` is a fractional
offset, shared by the photo (`Image`) and the mask (`MaskMulti`). The mask is
not wrapped as `Image`: a projective warp of an `Image` interpolates and mixes
class ids. `MaskMulti` is warped with nearest neighbor.
"""

using DataAugmentation
import DataAugmentation: apply, getrandstate, itemdata

const VOC_IGNORE = 0xff

"""
    mask_ids(mask, ids)

`ids` converts the dataset mask image to a class-id matrix.
"""
mask_ids(mask, ids::Function) = ids(mask)


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

function crop_offsets(ids::AbstractMatrix{<:Integer}, crop::NTuple{2,Integer}, cy::Integer, cx::Integer)
    return (
        crop_offsets(size(ids, 1), crop[1], cy),
        crop_offsets(size(ids, 2), crop[2], cx),
    )
end


"""
    class_center(ids; ignore=0xff, background=0) -> (cy, cx) or nothing

`ids` is already the class-id matrix. Draw a class uniformly, excluding
background and void, then draw a pixel of that class.
"""
function class_center(ids::AbstractMatrix{<:Integer}; ignore::Integer = VOC_IGNORE, background::Integer = 0)
    present = Int[]
    for c in unique(ids)
        (c == background || c == ignore) && continue
        push!(present, Int(c))
    end
    isempty(present) && return nothing
    cls = present[rand(1:length(present))]
    idx = findall(==(cls), ids)
    p = idx[rand(1:length(idx))]
    return p[1], p[2]
end


"""
    class_center_state(ids, crop; kwargs...) -> NTuple{2,Float64}

`randstate` for `Crop(crop, FromRandom())`. Falls back to a uniform offset
when the id matrix has no foreground.
"""
function class_center_state(ids::AbstractMatrix{<:Integer}, crop::NTuple{2,Integer}; kwargs...)
    hit = class_center(ids; kwargs...)
    hit === nothing && return (rand(), rand())
    return crop_offsets(ids, crop, hit[1], hit[2])
end
class_center_state(ids, crop::Integer; kwargs...) = class_center_state(ids, (crop, crop); kwargs...)


struct VocProjectiveCrop{F} <: Transform
    sides::Vector{Int}
    crop::NTuple{2,Int}
    center_prob::Float64
    ignore::Int
    background::Int
    ids::F
end

"""
    VocProjectiveCrop(; ids = mask -> mask.index, short_sides=320:500, crop=256, center_prob=0.5)

`ids` is the only dataset-specific piece. The default is the Pascal VOC
indexed image. A Cityscapes loader passes its own function.
"""
function VocProjectiveCrop(;
        short_sides = 320:500,
        crop::Integer = 256,
        center_prob::Real = 0.5,
        ignore::Integer = VOC_IGNORE,
        background::Integer = 0,
        ids::Function = mask -> mask.index,
)
    return VocProjectiveCrop(
        collect(Int, short_sides),
        (Int(crop), Int(crop)),
        Float64(center_prob),
        Int(ignore),
        Int(background),
        ids,
    )
end

function apply(tfm::VocProjectiveCrop, items::Tuple{Image,MaskMulti}; randstate = nothing)
    side = tfm.sides[rand(1:length(tfm.sides))]
    scaled = (
        apply(ScaleKeepAspect((side, side)), items[1]),
        apply(ScaleKeepAspect((side, side)), items[2]),
    )
    # Undo the MaskMulti shift. Class ids are whatever `ids` produced.
    raw = itemdata(scaled[2]) .- 1
    offsets = if rand() < tfm.center_prob
        class_center_state(raw, tfm.crop; ignore = tfm.ignore, background = tfm.background)
    else
        (rand(), rand())
    end
    crop = Crop(tfm.crop, DataAugmentation.FromRandom())
    cropped = (
        apply(crop, scaled[1]; randstate = offsets),
        apply(crop, scaled[2]; randstate = offsets),
    )
    return apply(PinOrigin(), cropped[1]), apply(PinOrigin(), cropped[2])
end

function apply(tfm::Sequence, items::Tuple{Image,MaskMulti}; randstate = nothing)
    state = randstate === nothing ? getrandstate(tfm) : randstate
    for (t, r) in zip(tfm.ts, state)
        items = apply(t, items; randstate = r)
    end
    return items
end

function apply(tfm::VocProjectiveCrop, item::Item; randstate = nothing)
    error("VocProjectiveCrop must be applied to an (Image, MaskMulti) pair")
end
