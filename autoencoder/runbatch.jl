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
    # "unet5ae_mse_logits.jl",
    # "unet5ae_mae_logits.jl",
    # "unet5ae_dssim_logits.jl",         # best model
    # "unet5ae_dssim_logits_step1.jl",   # no improvement from best
    "unet5ae_dssim_logits_step2.jl",
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
