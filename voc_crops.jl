using Images

const IGNORE = 0xff

"""
    scale_jitter(img, mask, short_side; ignore=0xff)

Resize so the short side equals `short_side`, preserving aspect ratio.
Image is interpolated; mask uses nearest-neighbor so class indices stay intact.
"""
function scale_jitter(img::AbstractMatrix, mask::AbstractMatrix{<:Integer}, short_side::Integer; ignore::Integer=IGNORE)
    short_side > 0 || throw(ArgumentError("short_side must be positive"))
    axes(img) == axes(mask) || throw(DimensionMismatch("image and mask axes differ"))
    h, w = size(img)
    scale = short_side / min(h, w)
    nh, nw = max(1, round(Int, h * scale)), max(1, round(Int, w * scale))
    img2 = imresize(img, (nh, nw))
    mask2 = imresize(mask, (nh, nw); method=Images.Nearest())
    return img2, mask2
end

"""
    scale_jitter(img, mask, short_sides::AbstractVector{<:Integer}; kwargs...)

Draw `short_side` uniformly from `short_sides`, then resize.
"""
function scale_jitter(img::AbstractMatrix, mask::AbstractMatrix{<:Integer}, short_sides::AbstractVector{<:Integer}; kwargs...)
    isempty(short_sides) && throw(ArgumentError("short_sides is empty"))
    return scale_jitter(img, mask, short_sides[rand(1:length(short_sides))]; kwargs...)
end

"""
    random_crop(img, mask, crop; ignore=0xff)

Uniform random crop of side `crop`. If the image is smaller than the crop,
the window hangs off the border and is padded (black image, `ignore` mask).
"""
function random_crop(img::AbstractMatrix, mask::AbstractMatrix{<:Integer}, crop::Integer; ignore::Integer=IGNORE)
    crop > 0 || throw(ArgumentError("crop must be positive"))
    axes(img) == axes(mask) || throw(DimensionMismatch("image and mask axes differ"))
    h, w = size(mask)
    # Center range that a crop of this size can occupy. Negative span means
    # the crop is larger than the image and must overhang.
    top = h > crop ? rand(1:(h - crop + 1)) : rand((h - crop + 1):1)
    left = w > crop ? rand(1:(w - crop + 1)) : rand((w - crop + 1):1)
    cy = top + crop ÷ 2
    cx = left + crop ÷ 2
    return _window(img, mask, cy, cx, crop, ignore)
end

"""
    class_center_crop(img, mask, crop; ignore=0xff)

Crop of side `crop` centered on a random foreground pixel.
Class is drawn uniformly from ids present in the mask, excluding background
`0` and `ignore`; a pixel of that class is then drawn uniformly.
Falls back to `random_crop` when the mask has no foreground.
Overhang is padded: black for the image, `ignore` for the mask.
"""
function class_center_crop(img::AbstractMatrix, mask::AbstractMatrix{<:Integer}, crop::Integer; ignore::Integer=IGNORE)
    crop > 0 || throw(ArgumentError("crop must be positive"))
    axes(img) == axes(mask) || throw(DimensionMismatch("image and mask axes differ"))
    cy, cx = _foreground_center(mask, ignore)
    if cy === nothing
        return random_crop(img, mask, crop; ignore=ignore)
    end
    return _window(img, mask, cy, cx, crop, ignore)
end

"""
    augment(img, mask; short_sides=320:640, crop=320, center=true, ignore=0xff)

Scale jitter, then either a class-center crop or a random crop.
"""
function augment(img, mask; short_sides=320:640, crop::Integer=320, center::Bool=true, ignore::Integer=IGNORE)
    img2, mask2 = scale_jitter(img, mask, collect(short_sides); ignore=ignore)
    if center
        return class_center_crop(img2, mask2, crop; ignore=ignore)
    else
        return random_crop(img2, mask2, crop; ignore=ignore)
    end
end

function _foreground_center(mask, ignore)
    present = Int[]
    for c in unique(mask)
        (c == 0 || c == ignore) && continue
        push!(present, Int(c))
    end
    isempty(present) && return nothing, nothing
    cls = present[rand(1:length(present))]
    idx = findall(==(cls), mask)
    p = idx[rand(1:length(idx))]
    return p[1], p[2]
end

function _window(img, mask, cy, cx, crop, ignore)
    top = cy - crop ÷ 2
    left = cx - crop ÷ 2
    out_img = fill(zero(eltype(img)), crop, crop)
    out_mask = fill(oftype(mask[begin], ignore), crop, crop)
    h, w = size(mask)
    for j in 1:crop, i in 1:crop
        y = top + i - 1
        x = left + j - 1
        if 1 ≤ y ≤ h && 1 ≤ x ≤ w
            out_img[i, j] = img[y, x]
            out_mask[i, j] = mask[y, x]
        end
    end
    return out_img, out_mask
end
