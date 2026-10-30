using ProgressBars; pb=ProgressBars

# Flux.train!(loss, model, data, opt_state)
function trainEpoch!(lossfn, model, data, optstate; verbose=false, device)
    losses = Vector{Float32}(undef, length(data))

    for (i, (X,y)) in pb.ProgressBar( enumerate(data) )
        Xd = X |> device
        yd = y |> device
        loss = Flux.trainstep!(lossfn, model, (Xd,yd), optstate)
        losses[i] = loss
    end

    epochloss = mean(losses)

    if verbose   return epochloss, losses
    else         return epochloss
    end
end

# Aliases
trainModel! = trainEpoch!


function evaluateEpoch(lossfn, model, data, metrics; device=gpu)
    n_metrics = length(metrics)
    
    total_loss = 0.0
    total_metrics = zeros(Float32, n_metrics)
    n_batches = 0

    for (X, y) in pb.ProgressBar(data)
        X = X |> device
        y = y |> device
        # yhat = model(X)
        
        total_loss += lossfn(model, X, y)
        
        for (j, metric_fn) in enumerate(metrics)
            total_metrics[j] += metric_fn(model, X, y)
        end
        
        n_batches += 1
    end

    if n_batches == 0
        error("Data loader is empty")
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
        X = X |> device
        y = y |> device
        # yhat = model(X)
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