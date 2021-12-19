using ChemistryFeaturization

# Batching Utilities
"""
    batch_graph_data(laplacians, encoded_features)

Takes vectors of laplacians and encoded features and joins them
into a single graph of disjoint subgraphs. The resulting graph
is massive and hence the return types are sparse. Few of the layers
don't work with Sparse Arrays (specifically on GPUs), so it would
make sense to convert them to dense.
"""
struct BatchedAtomicGraph{T1,T2,S}
    laplacians::T1
    encoded_features::T2
    sizes::S
end

Flux.@functor BatchedAtomicGraph (laplacians, encoded_features)

batch_graph_data(t::Tuple) = batch_graph_data(t[1], t[2])

function batch_graph_data(laplacians, encoded_features)
    # Ideally should be sparse arrays but currently doesn't work well on GPUs
    _sizes = map(x -> size(x, 1), laplacians)
    total_nodes = sum(_sizes)
    batched_laplacian = zeros(eltype(laplacians[1]), total_nodes, total_nodes)
    idx = 1
    for i in 1:length(laplacians)
        batched_laplacian[idx:(idx + _sizes[i] - 1), idx:(idx + _sizes[i] - 1)] .= laplacians[i]
        idx += _sizes[i]
    end
    _sizes = vcat(0, cumsum(_sizes))
    enc_feats = hcat(encoded_features...)
    return BatchedAtomicGraph(batched_laplacian, enc_feats, _sizes)
end

function BatchedAtomicGraph(batch_size::Int, atoms::Vector{FeaturizedAtoms})
    return BatchedAtomicGraph(batch_size, map(x -> x.atoms.laplacian, atoms), map(x -> x.encoded_features, atoms))
end

function BatchedAtomicGraph(batch_size::Int, laplacians, encoded_features)
    return batch_graph_data.(zip(Iterators.partition(laplacians, batch_size),
                                 Iterators.partition(encoded_features, batch_size)))
end
