# FluxExperimental

Experimental features for Flux.

## Explicit Parameter/State Models

See [ExplicitFluxLayers.jl](https://github.com/avik-pal/ExplicitFluxLayers.jl)

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

### Common Helper Functions

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
