no_op(args...; kwargs...) = nothing

function fit_one_epoch!(model, data_iterator, loss_function; device=gpu)
    time_data_transfer, time_forward_backward, time_parameter_update = 0, 0, 0
    epoch_loss, total_data_count = 0, 0

    # Transfer model to device
    model = device(model)
    ps = Flux.params(model)

    # Core training loop
    for (i, data) in enumerate(data_iterator)
        fdata = first(data)
        batch_size = size(fdata, ndims(fdata))

        # Transfer Data to Device
        data_transfer_start_time = time()
        data = device.(data)
        time_data_transfer += time() - data_transfer_start_time

        # Compute Gradients
        forward_backward_start_time = time()
        _val = Zygote.withgradient(() -> loss_function(model, data...), ps)
        time_forward_backward += time() - forward_backward_start_time

        # Update Parameters
        loss = _val.val
        parameter_update_start_time = time()
        Flux.Optimise.update!(opt, ps, _val.grad)
        time_parameter_update += time() - parameter_update_start_time

        # Update logs
        epoch_loss += loss * batch_size
        total_data_count += batch_size

        # Force GC after a specific number of iterations
        i % 25 == 0 && GC.gc(true)
    end

    epoch_loss /= total_data_count

    # Force GC at the end of an epoch
    GC.gc(true)
    CUDA.reclaim()

    training_statistics = (time_data_transfer=time_data_transfer, time_forward_backward=time_forward_backward,
                           time_parameter_update=time_parameter_update, epoch_loss=epoch_loss,
                           total_data_count=total_data_count)

    return model, training_statistics
end
