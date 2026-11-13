using CoordinateTransformations
using ImageTransformations
using StaticArrays: SMatrix, SVector
using LinearAlgebra
using OffsetArrays

"""
    RandomPerspective(distortion_scale = 0.2)

Apply a random perspective (homography) warp.
`distortion_scale` controls how far the four corners are randomly moved
(0 = identity, 0.2 = mild, 0.5 = strong).
"""
struct RandomPerspective <: Transform
    distortion_scale::Float32
end
RandomPerspective(s::Real = 0.2f0) = RandomPerspective(Float32(s))

# 4 corners × (dx, dy) → 8 values in [-scale, scale]
function DataAugmentation.getrandstate(tfm::RandomPerspective)
    s = tfm.distortion_scale
    return ntuple(_ -> (rand(Float32) * 2f0 - 1f0) * s, 8)
end

# Callable that applies the inverse homography (required by warp)
struct HomographyInv
    Hinv::SMatrix{3,3,Float64,9}
end

function (t::HomographyInv)(x)
    p = SVector(x[1], x[2], 1.0)
    q = t.Hinv * p
    return SVector(q[1]/q[3], q[2]/q[3])
end

function DataAugmentation.apply(tfm::RandomPerspective, item::Image;
                                randstate = DataAugmentation.getrandstate(tfm))
    img = itemdata(item)
    H, W = size(img)

    src = Float32[1 1; W 1; W H; 1 H]
    dx = Float32[randstate[1], randstate[3], randstate[5], randstate[7]]
    dy = Float32[randstate[2], randstate[4], randstate[6], randstate[8]]

    dst = similar(src)
    dst[:,1] = src[:,1] .+ dx .* W
    dst[:,2] = src[:,2] .+ dy .* H

    Hmat = estimate_homography(src, dst)
    Hinv = inv(Hmat) |> SMatrix{3,3,Float64,9}
    tform = HomographyInv(Hinv)

    warped = ImageTransformations.warp(img, tform, axes(img))

    # Allocate a fresh 1-based array and copy
    plain = similar(img, size(warped))
    copyto!(plain, warped)
    return Image(plain)
end

function DataAugmentation.apply(tfm::RandomPerspective, item::MaskMulti;
                                randstate = DataAugmentation.getrandstate(tfm))
    mask = itemdata(item)
    H, W = size(mask)

    src = Float32[1 1; W 1; W H; 1 H]
    dx = Float32[randstate[1], randstate[3], randstate[5], randstate[7]]
    dy = Float32[randstate[2], randstate[4], randstate[6], randstate[8]]

    dst = similar(src)
    dst[:,1] = src[:,1] .+ dx .* W
    dst[:,2] = src[:,2] .+ dy .* H

    Hmat = estimate_homography(src, dst)
    Hinv = inv(Hmat) |> SMatrix{3,3,Float64,9}
    tform = HomographyInv(Hinv)

    warped = ImageTransformations.warp(mask, tform, axes(mask);
                                       method = ImageTransformations.Constant())

    # Fresh array + integer labels
    plain = similar(mask, size(warped))
    copyto!(plain, warped)
    warped_int = round.(eltype(mask), plain)

    # Clamp to the valid class range (avoids 0s introduced by the warp fill)
    cmin, cmax = extrema(item.classes)
    warped_int = clamp.(warped_int, cmin, cmax)

    return MaskMulti(warped_int, item.classes)
end

"""
Estimate a 3×3 homography from 4 point correspondences (src → dst).
Direct Linear Transform (DLT).
"""
function estimate_homography(src::AbstractMatrix, dst::AbstractMatrix)
    A = zeros(Float64, 8, 9)
    for i in 1:4
        x, y = src[i,1], src[i,2]
        u, v = dst[i,1], dst[i,2]
        A[2i-1,:] = [-x, -y, -1,  0,  0,  0, u*x, u*y, u]
        A[2i,  :] = [ 0,  0,  0, -x, -y, -1, v*x, v*y, v]
    end
    _, _, V = svd(A)
    h = V[:, end]
    return reshape(h, 3, 3)'
end