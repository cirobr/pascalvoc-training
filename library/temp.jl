function wrap_pair(img, mask)
    return Image(img), MaskMulti(mask .+ 1, 1:256)
end

function unwrap_pair(img_wrap, mask_wrap)
    img  = OffsetArrays.no_offset_view(img_wrap.data)
    mask = OffsetArrays.no_offset_view((mask_wrap.data))
    return img, mask
end

function apply_data_augmentation(
    img::AbstractArray{<:Color{N0f8}},
    mask::AbstractArray{Int16};
    intensity_tfm,
    geometric_tfm,
    image_tfm=ImageToTensor(),
    mask_tfm=OneHot(),
)
    img_wrap, mask_wrap = wrap_pair(img, mask)
    # augmentation
    img_wrap = apply(intensity_tfm, img_wrap)                           # intensity (img only)
    img_wrap, mask_wrap = apply(geometric_tfm, (img_wrap, mask_wrap))   # geometric (img, mask)
    # tensor representations
    img_wrap  = apply(image_tfm, img_wrap)
    mask_wrap = apply(mask_tfm, mask_wrap)
    # augmented output
    return unwrap_pair(img_wrap, mask_wrap)
end


function get_normalization_params(x::AbstractArray)
    dims = (1,2,4)
    μs = mean(x, dims=dims) |> x->dropdims(x, dims=dims)
    σs = std(x, dims=dims)  |> x->dropdims(x, dims=dims)
    return μs, σs
end
