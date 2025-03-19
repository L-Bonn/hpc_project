using CUDA
using JLD2
using Printf
using FixedPointNumbers
using FileIO
using Base64


WarpsT = 8
Bx = 32
By = 32
thrds2D = (WarpsT, WarpsT)
blcks2D = (Bx, By)

# Size
Nx = Ti(WarpsT * Bx)
Ny = Ti(WarpsT * By)
# Initialize the arrays
u = CuArray{Float64}(undef, Nx, Ny)
v = CuArray{Float64}(undef, Nx, Ny)

Δu = CuArray{Float64}(undef, Nx, Ny)
Δv = CuArray{Float64}(undef, Nx, Ny)

rhs_u = CuArray{Float64}(undef, Nx, Ny)
rhs_v = CuArray{Float64}(undef, Nx, Ny)

#randoms = CuArray{Float64}(undef, Nx, Ny, 2)

randoms = CUDA.rand(Float64, Nx, Ny, 2);


#intialise 
function kernel_init!(u, v, randoms)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    #d2 = (i-posx)^2 + (j-posz)^2
    u[i,j] = randoms[i, j, 1] - 0.5
    v[i,j] = randoms[i, j, 2] - 0.5
   
    return nothing
end

@inline build_init_cond!(u, v, randoms) = @cuda threads = thrds2D blocks = blcks2D kernel_init!(u, v, randoms)

function kernel_comp_∇_Δ!(Δu, Δv, u, v, Nx, Ny)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    im = mod(i-1,1:Nx); ip = mod(i+1,1:Nx); 
    jm = mod(j-1,1:Ny); jp = mod(j+1,1:Ny); 
    
    @inbounds begin
        Δu[i,j] = u[im,j] + u[ip,j] + u[i,jm] + u[i,jp] - 4*u[i,j]
        Δv[i,j] = v[im,j] + v[ip,j] + v[i,jm] + v[i,jp] - 4*v[i,j]
    end
    return nothing
end
function kernel_comp_time_step!(rhs_u, rhs_v, Δu, Δv, u, v, α, β, Nx, Ny)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    

    @inbounds begin
        fu = u[i, j] - (u[i, j] - β * v[i, j]) * (u[i, j]*u[i, j] + v[i, j]*v[i, j])
        fv = v[i, j] - (β * u[i, j] + v[i, j]) * (u[i, j]*u[i, j] + v[i, j]*v[i, j])

        rhs_u[i,j] = (Δu[i, j] - α * Δv[i, j]) + fu 
        rhs_v[i,j] = (α * Δu[i, j] + Δv[i, j]) + fv
    end
    return nothing
end

function kernel_update_fields!(u, v, rhs_u, rhs_v, dt, Nx, Ny)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    @inbounds begin
        u[i, j] += dt * rhs_u[i, j]
        v[i, j] += dt * rhs_v[i, j]
    end

    return nothing
end
function all_comp_kernels!(u, v, dt, α, β, Nx, Ny)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    im = mod(i-1,1:Nx); ip = mod(i+1,1:Nx); 
    jm = mod(j-1,1:Ny); jp = mod(j+1,1:Ny); 
    
    @inbounds begin
        Δu = u[im,j] + u[ip,j] + u[i,jm] + u[i,jp] - 4*u[i,j]
        Δv = v[im,j] + v[ip,j] + v[i,jm] + v[i,jp] - 4*v[i,j]

        fu = u[i, j] - (u[i, j] - β * v[i, j]) * (u[i, j]*u[i, j] + v[i, j]*v[i, j])
        fv = v[i, j] - (β * u[i, j] + v[i, j]) * (u[i, j]*u[i, j] + v[i, j]*v[i, j])

        rhs_u = (Δu - α * Δv) + fu 
        rhs_v = (α * Δu + Δv) + fv

        u[i, j] += dt * rhs_u
        v[i, j] += dt * rhs_v
    end
    return nothing
end

Nt = 500
tsave = 5
dt=0.01
dir_save = string("GPUData/")
mkpath(dir_save)

α = 1.0
β = -0.2

for t=0:Nt/dt

    @cuda threads = thrds2D blocks = blcks2D all_comp_kernels!(u, v, dt, α, β, Nx, Ny)

    #comute laplacians 
    #@cuda threads = thrds2D blocks = blcks2D kernel_comp_∇_Δ!(Δu, Δv, u, v, Nx, Ny)
    
    #compute right and side
    #@cuda threads = thrds2D blocks = blcks2D kernel_comp_time_step!(rhs_u, rhs_v, Δu, Δv, u, v, α, β, Nx, Ny)
    
    #update u and v
    #@cuda threads = thrds2D blocks = blcks2D kernel_update_fields!(u, v, rhs_u, rhs_v, dt, Nx, Ny)
    
    #if t%(tsave/dt)==0
    #    println(t*dt,", ")
    #    any(isnan, u) && return 1
    #    save(joinpath(dir_save,"data_"*@sprintf("%08i",t*dt)*".jld2"), "u", Array(u), "v", Array(v))
    #end
end