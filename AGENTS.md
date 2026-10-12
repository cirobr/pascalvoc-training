# AGENTS.md

PASCAL VOC 2012 semantic segmentation experiments (UNet via TinyMachines + Flux).

## Running experiments (non-obvious)

- `cd` into experiment subdir first (e.g. `cd 2.baseline`), because scripts do `cd(@__DIR__)` + relative paths like `datasetfolder = "../dataset/"`.
- Most run_*.jl have their arg block commented and **require** these vars defined in caller scope before `include` (runbatch.jl does this):
  ```julia
  envpath = "../"; cudadevice = 0; epochs = 2; minibatchsize = 1; accum_steps = 1; debugflag = true
  include("run_a.jl")
  ```
- `include("runbatch.jl")` runs the listed scripts (after `@. rm` of models/tblogs subdirs).
- Exception: `dev-crops/train2.jl` and `dataset/augm.jl` define the args themselves (no outer scope needed).
- Fast checks: `debugflag=true` reduces to 3 train + 2 valid samples + 2 epochs.
- Entry: `julia --project=.` (then cd + include inside REPL). Each script does `Pkg.activate(expanduser(envpath))`.
- Do not run most `*.jl` directly via `julia foo.jl` (UndefVarError on envpath etc.).

## Experiment groups (real boundaries)

- `1.preliminary/`: loss ablations (each overrides `trainLossFunction`); all `include("../dev-crops/DataAugmentationCrops.jl")` then locally define `data_augmentation` (plus `wrap_pair`/`unwrap_pair`).
- `2.baseline/`: dropout variants of baseline (a/b/c); `using LibDataAug; import LibDataAug: data_augmentation` (no dev-crops include).
- `dev-crops/`: crop dev. `train.jl` + `DataAugmentationCrops.jl` (local aug path, like 1.prelim); `train2.jl` (uses LibDataAug, like baseline).
- `dataset/`: committed `dftrain.csv`/`dftest.csv` (hardcoded `~` paths); notebooks for regen. `augm.jl` is self-contained but pulls undeclared `PreprocessingImages`.

## Setup / env

- `julia --project=.` (then `] instantiate` if sources not resolved; [sources] in `Project.toml` pull LibCUDA/LibDataAug/LibFluxML/PascalVocTools/TinyMachines -- ssh for some).
- VOC data is external (gitignored; symlinked under `dataset/pascalvoc`); `df*.csv` hardcode `~` paths passed to `expanduser`.
- GPU setup in scripts: `if CUDA.has_cuda_gpu() ... CUDA.device!(cudadevice) ...`; falls back to CPU. `LibCUDA.cleangpu()` called early + in training loop.
- No tests, CI, linters, format config, or task runner.

## Gotchas (will break without this)

- ~15 run scripts duplicate the entire training boilerplate (data loading, normalization, class weights, loop, logging, checkpointing). Edit one == edit all or it will drift.
- `trainEpoch!`, `evaluateEpoch`, `EarlyStopper`, `early_stop!`, `saveModelState`, scores, losses, `onehot_fast` etc. all come from `LibFluxML` (not Flux stdlib).
- Model checkpoint uses `lossfn` (MSE) for `Flux.setup`; real losses only in the `train/validLossFunction` wrappers.
- Per-run `outputfolder = basename(@__FILE__)[1:end-3] * "/"`.
- `runbatch.jl` always `rm` + `include`; individual scripts also `rm` their own `models/`/`tblogs/` on start.
- TensorBoard: `tblogs/<name>/`; `reset!(logger)`.
- At end: renames `model.jld2` -> `bestmodel.jld2`; `close(logger)`.
- Custom crops (`RandomCentricCrop`, `ClassCentricCrop`) only available if the dev include happened or via `LibDataAug`.
