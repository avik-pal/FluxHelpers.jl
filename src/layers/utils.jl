# Hack around Flux to allow multiple inputs to `Chain`
Flux.applychain(::Tuple{}, x1, xs...) = (x1, xs...)
Flux.applychain(fs::Tuple, x1, xs...) = Flux.applychain(Base.tail(fs), first(fs)(x1, xs...))

(c::Chain)(x1, xs...) = Flux.applychain(Tuple(c.layers), x1, xs...)