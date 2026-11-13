"""
Author: cirobr@GitHub
Date: 29-Oct-2026

Template for executing multiple scripts in series.
"""

### arguments
envpath       = "../"
cudadevice    = 1
epochs        = 500
train_batch   = 1
accum_steps   = 1
valid_batch   = train_batch * accum_steps
debugflag     = false


### projects
scripts = [
    # "ce.jl",      # os16, auxhead, 1e-4 regularizer, weighted loss, soft target
    # "ce_p2.jl",   # weights[1] = 0 (dominant in previous run)
    # "dice.jl",    # best model
    # "gdl.jl",     # not working at all
    # "cedice.jl",
    # "focal.jl",
    "dice_aug.jl",   # same as dice.jl + augmentation strategy
]
scriptfolders = [s[1:end-3] for s in scripts]

# cleanup folders
models = @. "models/" * scriptfolders * "/"
tblogs = @. "tblogs/" * scriptfolders * "/"

@info "Batch started"
@. rm(models, force=true, recursive=true)
@. rm(tblogs, force=true, recursive=true)
@. include(scripts)
@info "Batch completed!"
