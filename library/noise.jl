# using Flux


# sigma: 0.05 - 0.12
function add_gaussian_noise(x::AbstractArray{<:Real}; sigma::Real=0.08)
    σ = Flux.ofeltype(x, sigma)
    noise = randn(eltype(x), size(x)) .* σ
    return x .+ noise
end

# x=rand(Float32, (16,16,3,5))
# add_gaussian_noise(x; sigma=0.05)


