# Code adapted from https://github.com/Chemellia/AtomicGraphNets.jl/blob/main/src/layers.jl
# Layers have been GPUified (will try to get them upstreamed)
# AGN stands for AtomicGraphNets
function pool_out_features(num_f::Int64, dim::Int64, stride::Int64, pad::Int64)
    return Int64(floor((num_f + 2 * pad - dim) / stride + 1))
end

function compute_pool_params(num_f_in::Int64, num_f_out::Int64, dim_frac::AbstractFloat;
                             start_dim=Int64(round(dim_frac * num_f_in)), start_str=Int64(floor(num_f_in / num_f_out)))
    # take starting guesses
    dim = start_dim
    str = start_str
    p_numer = str * (num_f_out - 1) - (num_f_in - dim)
    if p_numer < 0
        p_numer == -1 ? dim = dim + 1 : str = str + 1
    end
    p_numer = str * (num_f_out - 1) - (num_f_in - dim)
    if p_numer < 0
        error("problem, negative p!")
    end
    if p_numer % 2 == 0
        pad = Int64(p_numer / 2)
    else
        dim = dim - 1
        pad = Int64((str * (num_f_out - 1) - (num_f_in - dim)) / 2)
    end
    out_fea_len = pool_out_features(num_f_in, dim, str, pad)
    if !(out_fea_len == num_f_out)
        print("problem, output feature wrong length!")
    end
    # check if pad gets comparable to width...
    if pad >= 0.8 * dim
        @warn "specified pooling width was hard to satisfy without nonsensically large padding relative to width, had to increase from desired width"
        dim, str, pad = compute_pool_params(num_f_in, num_f_out, dim_frac; start_dim=Int64(round(1.2 * start_dim)))
    end
    return dim, str, pad
end

abstract type AtomicGraphLayer end

(l::AtomicGraphLayer)(x::Tuple) = l(x[1], x[2])

struct AGNConv{W,B,F} <: AtomicGraphLayer
    weight::W
    bias::B
    activation::F
end

function AGNConv(ch::Pair{<:Integer,<:Integer}, activation=gelu; initW=Flux.glorot_uniform, initb=zeros,
                 T::DataType=Float32)
    weight = T(2) .* T.(initW(ch[2], ch[1]))
    b = initb(ch[2], 1)
    if !(b isa Flux.Zeros)
        b = T.(b)
    end
    return AGNConv(weight, b, activation)
end

Flux.@functor AGNConv

function (l::AGNConv)(lapl::AbstractMatrix, X::AbstractMatrix)
    out_mat = normalise01(l.activation.(l.weight * X * (lapl + I) .+ l.bias))
    return lapl, out_mat
end

struct AGNMaxPool <: AtomicGraphLayer
    dim::Int64
    stride::Int64
    pad::Int64
end

struct AGNMeanPool <: AtomicGraphLayer
    dim::Int64
    stride::Int64
    pad::Int64
end

function AGNPool(pool_type::Symbol, ch::Pair{<:Integer,<:Integer}, pool_width_frac::AbstractFloat)
    in_num_features, out_num_features = ch
    dim, stride, pad = compute_pool_params(in_num_features, out_num_features, pool_width_frac)
    return (pool_type == :max ? AGNMaxPool : AGNMeanPool)(dim, stride, pad)
end

AGNMaxPool(args...) = AGNPool(:max, args...)

AGNMeanPool(args...) = AGNPool(:mean, args...)

function (m::AGNMaxPool)(lapl::AbstractMatrix, X::AbstractMatrix)
    x = reshape(X, (size(X)..., 1, 1))
    pdims = PoolDims(x, (m.dim, 1); padding=(m.pad, 0), stride=(m.stride, 1))
    return lapl, mean(maxpool(x, pdims); dims=2)[:, :, 1, 1]
end

function (m::AGNMeanPool)(lapl::AbstractMatrix, X::AbstractMatrix)
    x = reshape(X, (size(X)..., 1, 1))
    pdims = PoolDims(x, (m.dim, 1); padding=(m.pad, 0), stride=(m.stride, 1))
    return lapl, mean(meanpool(x, pdims); dims=2)[:, :, 1, 1]
end
