"""
Author: cirobr@GitHub
Date: 18-Sep-2026

Template for executing multiple scripts in series.
    * Each project lives on a separate folder.
"""

### arguments
envpath       = "../"
cudadevice    = 1
epochs        = 500
minibatchsize = 6
accum_steps   = 2
debugflag     = false


### projects
scripts = [
    "baseline-iou_loss.jl",
    # "baseline-ce_loss.jl",
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
