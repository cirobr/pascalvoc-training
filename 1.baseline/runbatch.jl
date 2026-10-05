"""
Author: cirobr@GitHub
Date: 18-Sep-2026

Template for executing multiple scripts in series.
    * Each project lives on a separate folder.
"""

### arguments
envpath       = "../"
cudadevice    = 1
epochs        = 30
minibatchsize = 6
accum_steps   = 2
debugflag     = true


### projects
scripts = [
    # "baseline-iou_loss_5e-3.jl",
    # "baseline-iou_loss_1e-3.jl",
    # "baseline-ce_loss_5e-3.jl",
    # "baseline-ce_loss_1e-3.jl",
    # "baseline-gdfl_1e-3.jl",
    "baseline-dice_loss_1e-3.jl",
    "baseline-dicesq_loss_1e-3.jl",
    "baseline-focal_loss_1e-3.jl",
    "baseline-gdl_1e-3.jl",
    "baseline-gdlsq_1e-3.jl",
    # "baseline-iou_loss_1e-3_weights.jl",   # bug fix
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
