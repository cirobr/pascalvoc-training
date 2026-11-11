using ProgressBars; pb=ProgressBars
# Flux.train!(loss, model, data, opt_state)

"""
    trainEpoch!(lossfn, model, data, optstate; device)

Self-contained function that trains a model for one epoch using the given
loss function, data, and optimizer state.

# Arguments
- `lossfn`: Loss function to use for training, which accepts **(model, X, y)** as arguments.
- `model`: Flux model to train.
- `data`: Data iterator that yields `(X, y)` pairs.
- `optstate`: Optimizer state (e.g., from `Flux.setup`).
- `device`: Function to move data to the desired device (e.g., `gpu` or `cpu`).

# Returns
Average loss over the epoch.

# Aliases
- `trainModel!`: Alias for `trainEpoch!`.
"""
function trainEpoch!(lossfn, model, data, optstate; device)
    total_loss = 0.0
    n_batches  = 0

    for (X, yhard, ysoft) in pb.ProgressBar(data)
        Xd = X |> device
        yd = ysoft |> device
        total_loss += Flux.trainstep!(lossfn, model, (Xd,yd), optstate)
        n_batches  += 1
    end

    return total_loss / n_batches
end

# Alias
trainModel! = trainEpoch!


"""
    evaluateEpoch(lossfn, model, data, metrics; device)

Self-contained function that evaluates a model for one epoch using the given
loss function, data, and metrics.

# Arguments
- `lossfn`: Loss function to use for evaluation, which accepts **(yhat, y)** as arguments.
- `model`: Flux model to evaluate.
- `data`: Data iterator that yields `(X, y)` pairs.
- `metrics`: Array of metric functions to evaluate.
- `device`: Function to move data to the desired device (e.g., `gpu` or `cpu`).

# Returns
- `avg_loss`: Average loss over the epoch.
- `avg_metrics`: Array of average metrics over the epoch.

# Aliases
- `testModel`: Alias for `evaluateEpoch`.
- `evaluateModel`: Alias for `evaluateEpoch`.
"""
function evaluateEpoch(lossfn, model, data, metrics; device)
    Flux.testmode!(model)
    n_metrics = length(metrics)
    
    total_loss = 0.0
    n_batches  = 0
    total_metrics = zeros(Float32, n_metrics)

    for (X, yhard, ysoft) in pb.ProgressBar(data)
        Xd = X |> device
        yd = yhard |> device
        yhat = model(Xd)
        total_loss += lossfn(yhat, yd)
        
        for (i, metric_fn) in enumerate(metrics)
            total_metrics[i] += metric_fn(yhat, yd)
        end
        
        n_batches += 1
        GC.gc(false)
    end

    avg_loss = total_loss / n_batches
    avg_metrics = total_metrics ./ n_batches

    return avg_loss, avg_metrics
end

# Aliases
testModel     = evaluateEpoch
evaluateModel = evaluateEpoch


"""
    evaluatePerClassMetric(model, data, metric; device)

Self-contained function that evaluates a model for one epoch using the given
per-class metric function.

# Arguments
- `model`: Flux model to evaluate.
- `data`: Data iterator that yields `(X, y)` pairs.
- `metric`: Function that accepts **(yhat, y)** as arguments and computes a per-class metric, returning a vector of length `C`.
- `device`: Function to move data to the desired device (e.g., `gpu` or `cpu`).

# Returns
Vector of average per-class metrics over the epoch.
"""
function evaluatePerClassMetric(model, data, metric; device)
    Flux.testmode!(model)
    total = nothing   # will be initialized on first batch
    n_batches = 0

    for (X, yhard, ysoft) in pb.ProgressBar(data)
        Xd = X |> device
        yd = yhard |> device
        yhat = model(Xd)
        scores = metric(yhat, yd)   # Vector of length C

        if total === nothing
            # Initialize on first batch
            total = zeros(eltype(scores), length(scores))
        end

        total .+= scores
        n_batches += 1
        GC.gc(false)
    end

    return total ./ n_batches
end