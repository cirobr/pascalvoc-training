# Flux.train!(loss, model, data, opt_state)
function trainEpoch!(lossfn, model, data, optstate; verbose=false, device=gpu)
    losses = Vector{Float32}(undef, length(data))
    grads = nothing

    for (i, (X,y)) in pb.ProgressBar( enumerate(data) )
        LibCUDA.garbage_collection()
        X = X |> device
        y = y |> device
        loss, grads = Flux.withgradient(model) do m
            lossfn(m, X, y)
        end

        Flux.update!(optstate, model, grads[1])
        losses[i] = loss
    end

    epochloss = mean(losses)
    epochgrads = grads[1]

    if verbose   return epochloss, losses, epochgrads
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