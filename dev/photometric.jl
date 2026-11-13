using Colors: HSV, Gray, RGB, N0f8
using ImageFiltering   # only if GaussianBlur is included

# ---------- helpers ----------
_preserve_eltype(img, out) = oftype(img, out)

# ---------- AdjustSaturation ----------
struct AdjustSaturation <: Transform
    δ::Float32
end
AdjustSaturation(δ::Real = 0.4f0) = AdjustSaturation(Float32(δ))

DataAugmentation.getrandstate(tfm::AdjustSaturation) =
    1f0 + (rand(Float32) * 2f0 - 1f0) * tfm.δ

function DataAugmentation.apply(tfm::AdjustSaturation, item::Image;
                                randstate = DataAugmentation.getrandstate(tfm))
    img = itemdata(item)
    factor = randstate
    hsv = HSV.(img)
    for i in eachindex(hsv)
        s = clamp(hsv[i].s * factor, 0f0, 1f0)
        hsv[i] = HSV(hsv[i].h, s, hsv[i].v)
    end
    # Preserve original element type (RGB{N0f8}, Gray{N0f8}, ...)
    return Image(_preserve_eltype(img, hsv))
end

# ---------- AdjustHue ----------
struct AdjustHue <: Transform
    δ::Float32
end
AdjustHue(δ::Real = 0.1f0) = AdjustHue(Float32(δ))

DataAugmentation.getrandstate(tfm::AdjustHue) =
    (rand(Float32) * 2f0 - 1f0) * tfm.δ

function DataAugmentation.apply(tfm::AdjustHue, item::Image;
                                randstate = DataAugmentation.getrandstate(tfm))
    img = itemdata(item)
    shift = randstate
    hsv = HSV.(img)
    for i in eachindex(hsv)
        h = mod(hsv[i].h + shift, 1f0)
        hsv[i] = HSV(h, hsv[i].s, hsv[i].v)
    end
    return Image(_preserve_eltype(img, hsv))
end

# ---------- Grayscale ----------
struct Grayscale <: Transform end

function DataAugmentation.apply(::Grayscale, item::Image; randstate = nothing)
    img = itemdata(item)
    gray = Gray.(img)
    # Convert back to the original color type
    return Image(_preserve_eltype(img, gray))
end

# ---------- GaussianBlur ----------
struct GaussianBlur <: Transform
    σmin::Float32
    σmax::Float32
end
GaussianBlur(σ::NTuple{2,Real} = (0.1f0, 2.0f0)) =
    GaussianBlur(Float32(σ[1]), Float32(σ[2]))

DataAugmentation.getrandstate(tfm::GaussianBlur) =
    tfm.σmin + rand(Float32) * (tfm.σmax - tfm.σmin)

function DataAugmentation.apply(tfm::GaussianBlur, item::Image;
                                randstate = DataAugmentation.getrandstate(tfm))
    img = itemdata(item)
    σ = randstate
    blurred = ImageFiltering.imfilter(img, ImageFiltering.Kernel.gaussian(σ))
    # imfilter often returns Float32 or a different color type → restore original
    return Image(_preserve_eltype(img, blurred))
end