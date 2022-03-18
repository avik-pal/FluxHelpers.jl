# FluxExperimental

Experimental features for Flux.

## Explicit Parameter/State Models

Flux by default relies on storing parameters and states in its model structs. `ExplicitLayers` is an initial prototype
to make Flux explicit-parameter first.

An `ExplicitLayer` is simply the minimal set of fixed attributes required to define a layer, i.e. an `ExplicitLayer` is
always immutable. It doesn't contain any parameters or state variables. As an example consider `BatchNorm`

```julia
struct BatchNorm{F1,F2,F3,N} <: ExplicitLayer
    λ::F1
    ϵ::N
    momentum::N
    chs::Int
    initβ::F2
    initγ::F3
    affine::Bool
    track_stats::Bool
end
```

None of these attributes of BatchNorm change over time. Next each layer needs to have the following functions defined

1. `initialparameters(rng::AbstractRNG, layer::CustomExplicitLayer)` -- This returns a NamedTuple containing the trainable
   parameters for the layer. For `BatchNorm`, this would contain `γ` and `β` if `affine` is set to `true` else it should
   be an empty `NamedTuple`.
2. `initialstates(rng::AbstractRNG, layer::CustomExplicitLayer)` -- This returns a NamedTuple containing the current
   state for the layer. For most layers this is typically empty. Layers that would potentially contain this include
   `BatchNorm`, Recurrent Neural Networks, etc. For `BatchNorm`, this would contain `μ`, `σ²`, and `training`.
3. `parameterlength(layer::CustomExplicitLayer)` & `statelength(layer::CustomExplicitLayer)` -- These can be automatically
   calculated, but it is better to define these else we construct the parameter and then count the values which is quite
   wasteful.

Additionally each ExplicitLayer must return a Tuple of length 2 with the first element being the computed result and the
second element being the new state.

### Usage

```julia
using FluxExperimental, Random, Flux, Optimisers

# Construct the layer
model = ExplicitLayers.Chain(
    ExplicitLayers.BatchNorm(128),
    ExplicitLayers.Dense(128, 256, tanh),
    ExplicitLayers.BatchNorm(256),
    ExplicitLayers.Chain(
        ExplicitLayers.Dense(256, 1, tanh),
        ExplicitLayers.Dense(1, 10)
    )
)

# Parameter and State Variables
ps, st = ExplicitLayers.setup(MersenneTwister(0), model)

# Dummy Input
x = rand(MersenneTwister(0), Float32, 128, 2);

# Run the model
y, st = ExplicitLayers.apply(model, x, ps, st)

# Gradients
gs = gradient(p -> sum(ExplicitLayers.apply(model, x, p, st)[1]), ps)[1]

# Optimisation
st_opt = Optimisers.setup(Optimisers.ADAM(0.0001), ps)
st_opt, ps = Optimisers.update(st_opt, ps, gs)
```

### Currently Implemented Explicit Layers (none of these are exported)

These layers have the same API as their Flux counterparts.

* `Chain`
* `Dense`
* `Conv`
* `BatchNorm`
* `WeightNorm`

## Destructuring Flux Models

`Flux.destructure` takes a flux model and constructs a vector containing all abstract array leaves from it. This has weird
consequences in packages like `DiffEqFlux` where it expects that function to only return the `parameters` so that we can
optimize them. Currently, we end up optimizing even the `states`. Here, we make an explicit distinction of the leaf abstract
arrays into `parameters` and `states`.

* `destructure_parameters`
* `destructure_parameters_states`

Performance Implications: **Not really!!**

```julia
using Flux, FluxExperimental, BenchmarkTools, Metalhead

model = VGG19(; batchnorm=true)

@btime Flux.destructure($model) # 209.156 ms (966 allocations: 548.25 MiB)
@btime destructure_parameters($model) # 196.696 ms (2127 allocations: 548.34 MiB)

@btime begin
    p, re = Flux.destructure($model)
    re(p)
end # 446.756 ms (2567 allocations: 1.07 GiB)

@btime begin
    p, re = destructure_parameters($model)
    re(p)
end # 427.532 ms (4011 allocations: 1.07 GiB)
```

## Performance

* `enable_fast_mode!`

## Loading & Saving Models

* `save_flux_model`
* `load_flux_model`

## Some Standard Flux Layers (not Explicit)

## Common Helper Functions

* `conv1x1`, `conv3x3`, `conv5x5`
* `conv_norm`, `conv1x1_norm`, `conv3x3_norm`, `conv5x5_norm`
* `upsample_module`, `downsample_module`

### Proper Layers

* `WeightNorm`, `SpectralNorm`
* `VariationalHiddenDropout`, `update_is_variational_hidden_dropout_mask_reset_allowed`
* `AGNConv`, `AGNMaxPool`, `AGNMeanPool`
* `GroupNormV2`, `BatchNormV2`
* `FChain`
* `ReshapeLayer`, `FlattenLayer`, `SelectDim`, `NoOpLayer`

## Graph Neural Networks

* `batch_graph_data`, `BatchedAtomicGraph`

## Logging

* `ParameterStateGradientWatcher` -- Currently has Wandb bindings. Similar to `wandb.watch` for Pytorch

## API Reference

All the exported functions have associated docstrings. At the moment there is no explicit documentation page,
so please check out the docstrings in the `?help` mode.
