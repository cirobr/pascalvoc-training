function convert_to_gray_tensor(x::AbstractArray{Float32,3})
      n = size(x)[3]
      if n == 3
            x_chw = permutedims(x, (3,1,2))
            img_color = colorview(RGB, x_chw)
            img_gray  = Gray.(img_color)
            gray_tensor = channelview(img_gray)
      elseif n == 1
            gray_tensor = x
      else
            error("input tensor with incorrect channels dimension")
      end

      return gray_tensor # Float32,3
end


"""
Add horizontal+vertical gradient detail channel
"""
function add_grads_channel(x::AbstractArray{T,N}) where {T,N}
      x_gray_tensor = convert_to_gray_tensor(x)
      x_grad = p.transf_grad(x_gray_tensor)
      return reshape(x_grad, size(x_grad)..., 1)
end


"""
Add high-frequency detail channels from a 1-level DWT.
"""
function add_wavelet_channels(x::AbstractArray{T,N}; 
                              wavelet = wavelet(WT.haar), 
                              level::Int = 1,
) where {T,N}
      x_gray_tensor = convert_to_gray_tensor(x)[:,:]
      H, W = size(x_gray_tensor)

      # 1-level (or multi-level) DWT on this channel
      coeffs = dwt(x_gray_tensor, wavelet, level)

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
