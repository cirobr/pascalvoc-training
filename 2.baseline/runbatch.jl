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
    # "run_a.jl",
    # "run_b.jl",      ### best baseline model, iou_loss, lr=2e-3 ###
    # "run_c.jl",
    # "focal_2e-3.jl", #1: load baseline, focal loss, lr=2e-3
    # "dfl_2e-3.jl",   #2: load #1, focal_weight=1.0, dice_weight=0.2, lr=2e-3
    "dfl_3e-3.jl";   #3: load #2, focal_weight=1.0, dice_weight=0.5, lr=3e-3
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
