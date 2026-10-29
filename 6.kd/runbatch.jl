"""
Author: cirobr@GitHub
Date: 29-Oct-2026

Template for executing multiple scripts in series.
"""

### arguments
envpath       = "../"
cudadevice    = 0
epochs        = 500
train_batch   = 8
accum_steps   = 2
valid_batch   = train_batch * accum_steps
debugflag     = true


### projects
scripts = [
    "unet5_encoder.jl",
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
