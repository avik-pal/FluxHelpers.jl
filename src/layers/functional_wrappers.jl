# Basic Convolution Wrappers
function conv1x1(mapping, activation=identity; stride::Int=1, bias=false, kwargs...)
    return Conv((1, 1), mapping, activation; stride=stride, bias=bias, kwargs...)
end

function conv3x3(mapping, activation=identity; stride::Int=1, bias=false, kwargs...)
    return Conv((3, 3), mapping, activation; stride=stride, pad=1, bias=bias, kwargs...)
end

function conv5x5(mapping, activation=identity; stride::Int=1, bias=false, kwargs...)
    return Conv((5, 5), mapping, activation; stride=stride, pad=2, bias=bias, kwargs...)
end

function conv_norm(kernelsize, mapping, activation=identity; first_conv::Bool=true, norm_layer=BatchNorm,
                   group_count::Union{Nothing,Int}=nothing, conv_kwargs=Dict(), norm_kwargs=Dict(), which_chain=FChain)
    activation_norm, activation_conv, norm_channels = first_conv ? (activation, identity, last(mapping)) :
                                                      (identity, activation, first(mapping))
    conv_layer = Conv(kernelsize, mapping, activation_conv; conv_kwargs...)
    norm = if norm_layer ∈ (GroupNorm, GroupNormV2)
        @assert group_count !== nothing
        norm_layer(norm_channels, group_count, activation_norm; norm_kwargs...)
    elseif norm_layer ∈ (BatchNorm, InstanceNorm, BatchNormV2)
        norm_layer(norm_channels, activation_norm; norm_kwargs...)
    else
        error("Unknown norm layer: " * string(norm_layer))
    end

    return first_conv ? which_chain(conv_layer, norm) : which_chain(norm, conv_layer)
end

function conv1x1_norm(mapping, args...; conv_kwargs=Dict(), kwargs...)
    conv_kwargs = copy(conv_kwargs)
    conv_kwargs[:pad] = 0
    return conv_norm((1, 1), mapping, args...; conv_kwargs=conv_kwargs, kwargs...)
end

function conv3x3_norm(mapping, args...; conv_kwargs=Dict(), kwargs...)
    conv_kwargs = copy(conv_kwargs)
    conv_kwargs[:pad] = 1
    return conv_norm((3, 3), mapping, args...; conv_kwargs=conv_kwargs, kwargs...)
end

function conv5x5_norm(mapping, args...; conv_kwargs=Dict(), kwargs...)
    conv_kwargs = copy(conv_kwargs)
    conv_kwargs[:pad] = 2
    return conv_norm((5, 5), mapping, args...; conv_kwargs=conv_kwargs, kwargs...)
end

# Downsample Module
function downsample_module(mapping, resolution_mapping, args...; conv_kwargs=Dict(), which_chain=FChain, kwargs...)
    in_resolution, out_resolution = resolution_mapping
    in_channels, out_channels = mapping
    @assert in_resolution > out_resolution
    @assert ispow2(in_resolution ÷ out_resolution)
    level_diff = Int(log2(in_resolution ÷ out_resolution))

    intermediate_mapping(i) =
        if in_channels * (2^level_diff) == out_channels
            (in_channels * (2^(i - 1))) => (in_channels * (2^i))
        else
            i == level_diff ? in_channels => out_channels : in_channels => in_channels
        end

    conv_kwargs = copy(conv_kwargs)
    conv_kwargs[:stride] = 2
    return which_chain(vcat(map(x -> [x...],
                                [conv3x3_norm(intermediate_mapping(i), args...; which_chain=which_chain,
                                              conv_kwargs=conv_kwargs, kwargs...).layers for i in 1:level_diff])...)...)
end

# Upsample Module
function upsample_module(mapping, resolution_mapping, args...; upsample_mode::Symbol=:nearest, which_chain=FChain,
                         kwargs...)
    in_resolution, out_resolution = resolution_mapping
    in_channels, out_channels = mapping
    @assert in_resolution < out_resolution
    @assert ispow2(out_resolution ÷ in_resolution)
    level_diff = Int(log2(out_resolution ÷ in_resolution))

    intermediate_mapping(i) =
        if out_channels * (2^level_diff) == in_channels
            (in_channels ÷ (2^(i - 1))) => (in_channels ÷ (2^i))
        else
            i == level_diff ? in_channels => out_channels : in_channels => in_channels
        end

    return which_chain(vcat(map(x -> [x...],
                                [(conv1x1_norm(intermediate_mapping(i), args...; which_chain=which_chain,
                                               kwargs...).layers..., Upsample(upsample_mode; scale=2))
                                 for i in 1:level_diff])...)...)
end
