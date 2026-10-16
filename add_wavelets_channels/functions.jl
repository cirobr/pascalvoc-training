using Wavelets
using Images
using Interpolations
import Interpolations: BSpline
using PreprocessingImages; p=PreprocessingImages

"""
Add high-frequency detail channels from a 1-level DWT.
Input:  (H, W, C) Float32 image (C usually 3)
Output: (H, W, C + 3*C) or (H, W, C + 3) depending on the mode
"""
function add_wavelet_channels(img::AbstractArray{T,3}; 
                              wavelet = wavelet(WT.haar), 
                              level::Int = 1,
                              mode::Symbol = :details) where T
    
    img_rgb   = p.array2image(img)
    img_gray  = Gray.(img_rgb)
    img_float = channelview(img_gray)
    H, W = size(img_float)
    details = similar(img_gray, H, W, 0)

    # 1-level (or multi-level) DWT on this channel
    coeffs = dwt(img_float[:,:], wavelet, level)

    # Extract the three detail quadrants (half resolution)
    half_h, half_w = size(coeffs) .÷ 2
    LH = coeffs[1:half_h, half_w+1:end]          # or adjust indices if needed
    HL = coeffs[half_h+1:end, 1:half_w]
    HH = coeffs[half_h+1:end, half_w+1:end]

    # Upsample each detail map back to original size
    LH_up = imresize(LH, (H, W); method = BSpline(Linear()))
    HL_up = imresize(HL, (H, W); method = BSpline(Linear()))
    HH_up = imresize(HH, (H, W); method = BSpline(Linear()))

    # Optionally normalize / scale
    LH_up ./= maximum(abs, LH_up) + eps(T)
    HL_up ./= maximum(abs, HL_up) + eps(T)
    HH_up ./= maximum(abs, HH_up) + eps(T)

    if mode == :details
        details = cat(details, LH_up, HL_up, HH_up; dims = 3)
    end
   
    # Concatenate original + details
    return cat(img, details; dims = 3)
end

img = rand(Float32, (16,16,3))
x = add_wavelet_channels(img)