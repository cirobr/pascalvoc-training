using StatsBase: counts
import Flux: onecold, softmax

"""
    diagnostic_batch!(model, loader; device=gpu, max_batches=nothing)

Returns three things:
- pred_counts :: Vector{Int}   (length = C, how many pixels predicted as each class)
- present_counts :: Vector{Int} (length = C, how many images contain each class)
- cond_ious :: Vector{Float32}  (length = C, IoU on images that contain the class; NaN if never present)
"""
function diagnostic_batch!(model, loader; device=gpu, max_batches=nothing)
    C = nothing
    pred_counts = nothing
    present_counts = nothing
    inter = nothing
    union = nothing
    n_present = nothing

    nbatches = 0
    for (X, y) in loader
        Xd = X |> device
        yhat = model(Xd)                          # (H,W,C,N)
        C === nothing && (C = size(yhat, 3))

        if pred_counts === nothing
            pred_counts    = zeros(Int, C)
            present_counts = zeros(Int, C)
            inter          = zeros(Float32, C)
            union          = zeros(Float32, C)
            n_present      = zeros(Int, C)
        end

        # move to CPU for indexing / counts
        probs = softmax(yhat; dims=3) |> cpu

        # cleanup the memory
        Xd = nothing
        yhat = nothing
        GC.gc()

        # replace the pred = ... line with:
        pred_ci = argmax(probs; dims=3)               # Array of CartesianIndex
        pred    = map(ci -> ci[3], pred_ci)           # (H, W, N) plain Int (1-based class)
        pred    = dropdims(pred; dims=3)              # ensure no singleton dim

        # global prediction histogram
        pred_counts .+= counts(vec(pred), 1:C)

        # per-image presence + conditional IoU
        for n in 1:size(y, 4)
            yn = y[:,:,:,n]
            pn = pred[:,:,n]

            for c in 1:C
                has_c = any(yn[:,:,c])
                if has_c
                    present_counts[c] += 1
                    n_present[c] += 1

                    mask = yn[:,:,c]
                    inter[c] += sum((pn .== c) .& mask)
                    union[c] += sum((pn .== c) .| mask)
                end
            end
        end

        nbatches += 1
        max_batches !== nothing && nbatches >= max_batches && break
    end

    cond_ious = inter ./ (union .+ eps(Float32))
    cond_ious[n_present .== 0] .= NaN

    return pred_counts, present_counts, cond_ious
end