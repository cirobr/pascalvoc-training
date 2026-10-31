using ProgressBars; pb=ProgressBars

# Flux.train!(loss, model, data, opt_state)
function trainEpoch!(lossfn, model, data, optstate; verbose=false, device)
    total_loss = 0.0
    n_batches  = 0

    for (X,y) in pb.ProgressBar(data)
        Xd = X |> device
        yd = y |> device
        total_loss += Flux.trainstep!(lossfn, model, (Xd,yd), optstate)
        n_batches  += 1
    end

    return total_loss / n_batches
end

# Alias
trainModel! = trainEpoch!


function evaluateEpoch(lossfn, model, data, metrics; device=gpu)
    n_metrics = length(metrics)
    
    total_loss = 0.0
    n_batches  = 0
    total_metrics = zeros(Float32, n_metrics)

    for (X, y) in pb.ProgressBar(data)
        GC.gc(false)
        Xd = X |> device
        yd = y |> device
        total_loss += lossfn(model, Xd, yd)
        
        for (j, metric_fn) in enumerate(metrics)
            total_metrics[j] += metric_fn(model, Xd, yd)
        end
        
        n_batches += 1
    end

    avg_loss = total_loss / n_batches
    avg_metrics = total_metrics ./ n_batches

    return avg_loss, avg_metrics
end

# Aliases
testModel     = evaluateEpoch
evaluateModel = evaluateEpoch


function evaluatePerClassMetric(model, data, metric; device=gpu)
    total = nothing      # will be initialized on first batch
    n_batches = 0

    for (X, y) in pb.ProgressBar(data)
        GC.gc(false)
        X = X |> device
        y = y |> device
        scores = metric(model, X, y)   # Vector of length C

        if total === nothing
            # Initialize on first batch
            total = zeros(eltype(scores), length(scores))
        end

        total .+= scores
        n_batches += 1
    end

    if n_batches == 0
        error("Data loader is empty")
    end

    return total ./ n_batches
end