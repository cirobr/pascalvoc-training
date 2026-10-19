"""
Convert Float32,3 array to Float32 matrix (i.e., color to gray)
"""
function colorf32_to_grayf32(x::AbstractArray{T,3}) where T
      x_gray = p.array2image(x) .|> Gray
      return channelview(x_gray)
end


"""
Add horizontal+vertical gradient detail channel
"""
function add_grads_channel(x::AbstractArray{T,3}) where T
      x_float = colorf32_to_grayf32(x)
      x_grad  = p.transf_grad(x_float)
      return reshape(x_grad, size(x_grad)..., 1)
end


"""
Add high-frequency detail channels from a 1-level DWT.
"""
function add_wavelet_channels(x::AbstractArray{T,3}; 
                              wavelet = wavelet(WT.haar), 
                              level::Int = 1,
) where T
      x_float = colorf32_to_grayf32(x)
      H, W = size(x_float)

      # 1-level (or multi-level) DWT on this channel
      coeffs = dwt(x_float, wavelet, level)

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

      return LH_up, HL_up, HH_up
end

