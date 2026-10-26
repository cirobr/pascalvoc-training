"""
Author: cirobr@GitHub
Date: 18-Sep-2026

Template for executing multiple scripts in series.
"""

### arguments
# envpath       = "../"
# cudadevice    = 1
# epochs        = 500
# minibatchsize = 8
# accum_steps   = 1
# debugflag     = false


### projects
scripts = [
    # "wce.jl",    # weighted, 256x256
    # "wce2.jl",   # weighted, slower rise, 256x256
    # "wce3.jl",   # weighted, slower rise, 500x500
    # "wce3_2.jl",   # weighted, same rise, 256x256
    # "wce4.jl",   # same as above, imagenet stats
]
scriptfolders = [s[1:end-3] for s in scripts]

# cleanup folders
models = @. "models/" * scriptfolders * "/"
tblogs = @. "tblogs/" * scriptfolders * "/"

@info "Batch started"
@. rm(models, force=true, recursive=true)
@. rm(tblogs, force=true, recursive=true)
# @. include(scripts)
@info "Batch completed!"