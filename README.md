# FluxExperimental

Experimental features for Flux.

## Exported Functions

### Destructuring Flux Models

`Flux.destructure` takes a flux model and constructs a vector containing all abstract array leaves from it. This has weird
consequences in packages like `DiffEqFlux` where it expects that function to only return the `parameters` so that we can
optimize them. Currently, we end up optimizing even the `states`. Here, we make an explicit distinction of the leaf abstract
arrays into `parameters` and `states`.

* `destructure_parameters`
* `destructure_parameters_states`

Performance Implications: **Not really!!** We are marginally faster

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

### Performance

* `enable_fast_mode!`

### Loading & Saving Models

* `save_flux_model`
* `load_flux_model`

## API Reference

All the exported functions have associated docstrings. At the moment there is no explicit documentation page,
so please check out the docstrings in the `?help` mode.