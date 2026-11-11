"""
Author: cirobr@GitHub
Date: 29-Oct-2026

Template for executing multiple scripts in series.
"""

### arguments
envpath       = "../"
cudadevice    = 1
epochs        = 500
train_batch   = 4
accum_steps   = 4
valid_batch   = train_batch * accum_steps
debugflag     = true


### projects
scripts = [
    # "wdfl_os32_p1.jl",
    # "wdfl_os16_p1.jl",
    # "wdfl_os16_auxhead_p1.jl",
    # "wdfl_os16_auxhead_p2.jl",
    # "wfocal_os8_auxhead_p1.jl",
    # "wfocal_os8_auxhead_p2.jl",
    # "wfocal_os16_auxhead_p1.jl",
    # "wfocal_os16_auxhead_p1a.jl",   # 1e-4 regularizer
    "wfocal_os16_auxhead_smooth_p1.jl",   # label smoothing
    "wdice_os16_auxhead_smooth_p1.jl",
    "wdfl_os16_auxhead_smooth_p1.jl",
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
