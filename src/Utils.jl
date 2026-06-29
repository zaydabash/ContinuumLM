"""
    Utils.jl

Utility functions for device selection, seeding, and other helpers.
"""
module Utils

using CUDA
import Random

export select_device, set_seed, ensure_dir

"""
    select_device(device::String)

Return a function that moves data to the specified device.
Returns `CUDA.cuda` for GPU or identity function for CPU.
"""
function select_device(device::String)
    dev = device == "auto" ? (CUDA.has_cuda() ? "gpu" : "cpu") : device
    if dev == "gpu" && CUDA.has_cuda()
        return CUDA.cuda
    else
        return x -> x
    end
end

"""
    set_seed(seed::Integer)

Set global RNG seed for reproducibility.
"""
function set_seed(s::Integer)
    Random.seed!(s)
    if CUDA.has_cuda()
        CUDA.seed!(s)
    end
end

"""
    ensure_dir(path::String)

Ensure a directory exists, creating it if necessary.
"""
function ensure_dir(path::String)
    if !isdir(path)
        mkpath(path)
    end
end

end # module

