"""
DataAugmentation.jl wrappers for the VOC crop pipeline.

Class-center cropping reads mask labels, so it cannot be a ProjectiveTransform:
`getprojection` only sees bounds. `ShortSideScale` and the uniform crop reuse
native transforms. `ClassCenterCrop` is a plain `Transform` applied to the
`(Image, MaskMulti)` pair so both items share one window.

Label convention matches the training pipeline, which wraps the mask as
`MaskMulti(mask .+ 1, 1:256)`: background is 1, void is 256.
"""

using DataAugmentation
using OffsetArrays
import DataAugmentation: apply, getrandstate, itemdata

_plain(a) = OffsetArrays.no_offset_view(a)

struct ShortSideScale <: Transform
    sides::Vector{Int}
end

ShortSideScale(sides::AbstractVector{<:Integer}) = ShortSideScale(collect(Int, sides))

# One draw, shared when `apply` is called on an (image, mask) tuple.
getrandstate(tfm::ShortSideScale) = tfm.sides[rand(1:length(tfm.sides))]

function apply(tfm::ShortSideScale, item::Item; randstate = getrandstate(tfm))
    return apply(ScaleKeepAspect((randstate, randstate)), item)
end


struct ClassCenterCrop{N} <: Transform
    size::NTuple{N,Int}
    ignore::Int
    background::Int
end

function ClassCenterCrop(sz::NTuple{N,Integer}; ignore::Integer = 256, background::Integer = 1) where N
    return ClassCenterCrop{N}(Tuple(Int.(sz)), Int(ignore), Int(background))
end
ClassCenterCrop(sz::Integer; kwargs...) = ClassCenterCrop((sz, sz); kwargs...)

# The center is chosen from the mask, so the random state cannot be drawn
# before both items are available. `apply` on the pair samples it.
getrandstate(::ClassCenterCrop) = nothing

function apply(tfm::ClassCenterCrop, item::Item; randstate = nothing)
    error("ClassCenterCrop must be applied to an (Image, MaskMulti) pair")
end

function apply(
        tfm::ClassCenterCrop{2},
        items::Tuple{Image,MaskMulti};
        randstate = nothing,
)
    img, mask = items
    # ScaleKeepAspect returns OffsetArrays (e.g. indices 1:488 × 2:367).
    data = _plain(itemdata(mask))
    cy, cx = randstate === nothing ? _foreground_center(data, tfm.ignore, tfm.background) : randstate
    if cy === nothing
        # No foreground: same window rule as RandomCrop, including pad.
        return apply(PaddedRandomCrop(tfm.size; ignore = tfm.ignore), items)
    end
    return _crop_pair(img, mask, cy, cx, tfm.size, tfm.ignore)
end


struct PaddedRandomCrop{N} <: Transform
    size::NTuple{N,Int}
    ignore::Int
end

function PaddedRandomCrop(sz::NTuple{N,Integer}; ignore::Integer = 256) where N
    return PaddedRandomCrop{N}(Tuple(Int.(sz)), Int(ignore))
end
PaddedRandomCrop(sz::Integer; kwargs...) = PaddedRandomCrop((sz, sz); kwargs...)

function getrandstate(tfm::PaddedRandomCrop{2})
    # Placeholder. The legal range depends on the item size, so the draw
    # happens in `apply` unless the caller passes `(cy, cx)`.
    return nothing
end

function apply(tfm::PaddedRandomCrop, item::Item; randstate = nothing)
    error("PaddedRandomCrop must be applied to an (Image, MaskMulti) pair")
end

function apply(
        tfm::PaddedRandomCrop{2},
        items::Tuple{Image,MaskMulti};
        randstate = nothing,
)
    img, mask = items
    h, w = size(_plain(itemdata(mask)))
    crop = tfm.size[1]
    if randstate === nothing
        top = h > crop ? rand(1:(h - crop + 1)) : rand((h - crop + 1):1)
        left = w > crop ? rand(1:(w - crop + 1)) : rand((w - crop + 1):1)
        cy = top + crop ÷ 2
        cx = left + crop ÷ 2
    else
        cy, cx = randstate
    end
    return _crop_pair(img, mask, cy, cx, tfm.size, tfm.ignore)
end


"""
    _foreground_center(mask, ignore)

    Return the coordinates of a random foreground pixel in `mask`, excluding background (0) and `ignore`.
Returns `nothing` if no foreground pixels are present.
"""
function _foreground_center(mask, ignore, background)
    # check for foreground classes in the mask, excluding background (0) and ignore
    present = Int[]
    for c in unique(mask)
        (c == background || c == ignore) && continue
        push!(present, Int(c))
    end
    # if no foreground classes are present, return nothing
    isempty(present) && return nothing, nothing

    # select a random class and then a random pixel of that class
    cls = present[rand(1:length(present))]
    idx = findall(==(cls), mask)
    p = idx[rand(1:length(idx))]

    # return the coordinates of the selected pixel
    return p[1], p[2]
end


"""
    _crop_pair(img, mask, cy, cx, sz, ignore)

Crop a window of size `sz` centered on `(cy, cx)`.
If the window hangs off the image, the overhang is padded (black for the image, `ignore` for the mask).
"""
function _crop_pair(img, mask, cy, cx, sz, ignore)
    crop = sz[1]
    top  = cy - crop ÷ 2
    left = cx - crop ÷ 2

    src_img  = _plain(itemdata(img))
    src_mask = _plain(itemdata(mask))
    out_img  = fill(zero(eltype(src_img)), sz)
    out_mask = fill(oftype(src_mask[begin], ignore), sz)
    
    h, w = size(src_mask)
    for j in 1:crop, i in 1:crop
        y = top + i - 1
        x = left + j - 1
        if 1 ≤ y ≤ h && 1 ≤ x ≤ w
            out_img[i, j] = src_img[y, x]
            out_mask[i, j] = src_mask[y, x]
        end
    end

    return Image(out_img), MaskMulti(out_mask, mask.classes)
end


struct VocCrop <: Transform
    scale::ShortSideScale
    center::ClassCenterCrop{2}
    random::PaddedRandomCrop{2}
    center_prob::Float64
end

"""
    VocCrop(; short_sides=320:500, crop=256, center_prob=0.5)

One transform for the pair. A `Sequence` is not used: `apply` on a sequence
can map over the image and the mask separately, and the class center would
then no longer be shared.

Native `ScaleKeepAspect` does the resize. Native `RandomCrop` is not used:
it does not pad a window that hangs off the image, and it cannot read labels.
"""
function VocCrop(; short_sides = 320:500, crop::Integer = 256, center_prob::Real = 0.5)
    return VocCrop(
        ShortSideScale(short_sides),
        ClassCenterCrop(crop),
        PaddedRandomCrop(crop),
        Float64(center_prob),
    )
end

function apply(tfm::VocCrop, items::Tuple{Image,MaskMulti}; randstate = nothing)
    side = getrandstate(tfm.scale)
    scaled = apply(tfm.scale, items; randstate = side)
    crop = rand() < tfm.center_prob ? tfm.center : tfm.random
    return apply(crop, scaled)
end

function apply(tfm::VocCrop, item::Item; randstate = nothing)
    error("VocCrop must be applied to an (Image, MaskMulti) pair")
end
