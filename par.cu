#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>       // For random number generation
#include <ctime>        // For time-based seed
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;

// Error checking macro for CUDA
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Structure to hold grid information
struct GridInfo {
    int n;           // Grid size (n x n)
    double Lx;       // Domain size in x
    double Ly;       // Domain size in y
    double dx;       // Grid spacing in x
    double dy;       // Grid spacing in y
    double dt;       // Time step
    double alpha;    // PDE parameter alpha
    double beta;     // PDE parameter beta
    int num_steps;   // Number of time steps
    int nsave;       // Save interval
};

// Kernel for initializing arrays with random values
__global__ void initialize_random_kernel(double *u, double *v, int n, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < n && j < n) {
        int idx = j * n + i;
        
        // Simple CUDA-compatible random number generation
        // Using a different seed for each thread
        unsigned int thread_seed = seed + idx;
        thread_seed = (thread_seed * 1664525 + 1013904223) % UINT_MAX;
        double rand1 = (thread_seed / (double)UINT_MAX) - 0.5; // Random in [-0.5, 0.5]
        
        thread_seed = (thread_seed * 1664525 + 1013904223) % UINT_MAX;
        double rand2 = (thread_seed / (double)UINT_MAX) - 0.5; // Random in [-0.5, 0.5]
        
        u[idx] = rand1;
        v[idx] = rand2;
    }
}

// Kernel for performing the simulation step
__global__ void simulate_kernel(double *u, double *v, double *u_new, double *v_new, 
                               int n, double dx, double dt, double alpha, double beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < n && j < n) {
        int idx = j * n + i;
        
        // Handle periodic boundary conditions
        int im = (i - 1 + n) % n;
        int ip = (i + 1) % n;
        int jm = (j - 1 + n) % n;
        int jp = (j + 1) % n;
        
        double u_val = u[idx];
        double v_val = v[idx];
        
        // Neighbors (periodic boundary conditions)
        double u_ip = u[j * n + ip];
        double u_im = u[j * n + im];
        double u_jp = u[jp * n + i];
        double u_jm = u[jm * n + i];
        
        double v_ip = v[j * n + ip];
        double v_im = v[j * n + im];
        double v_jp = v[jp * n + i];
        double v_jm = v[jm * n + i];
        
        // Laplacians (5-point stencil, dx=dy)
        double lap_u = (u_ip + u_im + u_jp + u_jm - 4.0 * u_val) / (dx * dx);
        double lap_v = (v_ip + v_im + v_jp + v_jm - 4.0 * v_val) / (dx * dx);
        
        // PDE system
        double mag_sq = u_val * u_val + v_val * v_val;
        
        double rhs_u = (lap_u - alpha * lap_v) 
                     + u_val 
                     - (u_val - beta * v_val) * mag_sq;
        
        double rhs_v = (alpha * lap_u + lap_v) 
                     + v_val 
                     - (beta * u_val + v_val) * mag_sq;
        
        // Explicit update
        u_new[idx] = u_val + dt * rhs_u;
        v_new[idx] = v_val + dt * rhs_v;
    }
}

// Kernel for computing checksum
__global__ void compute_checksum_kernel(double *u, double *v, double *partial_sums, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    extern __shared__ double sdata[];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;
    sdata[tid] = 0.0;
    
    if (i < n && j < n) {
        int idx = j * n + i;
        double u_val = u[idx];
        double v_val = v[idx];
        double mag_sq = u_val * u_val + v_val * v_val;
        sdata[tid] = mag_sq;
    }
    
    __syncthreads();
    
    // Perform parallel reduction in shared memory
    for (int s = block_size/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        partial_sums[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];
    }
}

void write_to_csv(const vector<double> &u, const vector<double> &v, int n, const string &prefix) {
    ofstream outfile_u("data/" + prefix + "_u.csv");
    ofstream outfile_v("data/" + prefix + "_v.csv");
    
    if (!outfile_u.is_open() || !outfile_v.is_open()) {
        cerr << "Error: Could not open output CSV files." << endl;
        return;
    }
    
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            outfile_u << u[j * n + i];
            outfile_v << v[j * n + i];
            if (i < n - 1) {
                outfile_u << ",";
                outfile_v << ",";
            }
        }
        outfile_u << "\n";
        outfile_v << "\n";
    }
    
    outfile_u.close();
    outfile_v.close();
}

int main(int argc, char **argv) {
    // -------------------------------
    // 1) Simulation Parameters
    // -------------------------------
    GridInfo grid;
    grid.n = 200;             // Number of grid points in each dimension
    grid.Lx = 200.0;          // Domain size in x-direction
    grid.Ly = 200.0;          // Domain size in y-direction
    grid.dt = 0.01;           // Time step
    grid.num_steps = 5000;    // Number of time steps
    grid.nsave = 1000;        // Save interval
    
    bool random_seed = false;
    
    // PDE parameters
    grid.alpha = 2.0;
    grid.beta = -0.5;

    // Declare beta_arg to fix the error
    double beta_arg = 0.0;

    // Parse command line arguments
    std::vector<std::string> argument({argv, argv+argc});

    for (long unsigned int i = 1; i<argument.size() ; i += 2){
        std::string arg = argument[i];
        if(arg=="-h"){ // Write help
            std::cout << "./cuda_cgle --n <Number of grid points in each dimension>"
                      << " --Lx, --Ly <Domain size in x/y-direction>"
                      << " --dt <timestep size>"
                      << " --num_steps <Number of timesteps>"
                      << " --alpha <alpha CGLE>"
                      << " --beta <beta CGLE>"
                      << " --random_seed <whether to seed initial conditions randomly>\n";
            exit(0);
        } else if (i == argument.size() - 1){
            throw std::invalid_argument("The last argument (" + arg +") must have a value");
        } else if(arg=="--n"){
            if ((grid.n = std::stoi(argument[i+1])) < 0) 
                throw std::invalid_argument("n must be positive (e.g. --n 1000)");
        } else if(arg=="--Lx"){
            if ((grid.Lx = std::stoi(argument[i+1])) < 0) 
                throw std::invalid_argument("Lx must be positive (e.g. --Lx 1000)");
        } else if(arg=="--Ly"){
            if ((grid.Ly = std::stoi(argument[i+1])) < 0) 
                throw std::invalid_argument("Ly must be positive (e.g. --Ly 1000)");
        } else if(arg=="--dt"){
            if ((grid.dt = std::stod(argument[i+1])) < 0) 
                throw std::invalid_argument("dt must be positive (e.g. --dt 0.01)");
        } else if(arg=="--num_steps"){
            if ((grid.num_steps = std::stoi(argument[i+1])) < 0) 
                throw std::invalid_argument("num_steps must be positive (e.g. --num_steps 1000)");
        } else if(arg=="--nsave"){
            if ((grid.nsave = std::stoi(argument[i+1])) < 0) 
                throw std::invalid_argument("nsave must be positive (e.g. --nsave 1000)");
        } else if(arg=="--alpha"){
            if ((grid.alpha = std::stod(argument[i+1])) < 0) 
                throw std::invalid_argument("alpha must be positive (e.g. --alpha 1.1)");
        } else if(arg=="--beta"){
            beta_arg = std::stod(argument[i+1]);
            grid.beta = beta_arg;
        } else if(arg=="--random_seed"){
            (random_seed = std::stoi(argument[i+1]));
        } else{
            std::cout << "---> error: the argument type is not recognized \n";
        }
    }

    // Ensure Ly = Lx for square domain
    grid.Ly = grid.Lx;
    std::cout << "Set Ly = Lx because non-square domain is not implemented!" << std::endl;
    
    grid.dx = grid.Lx / grid.n;
    grid.dy = grid.Ly / grid.n;
    
    // -------------------------------
    // 2) Allocate host memory
    // -------------------------------
    size_t grid_size = grid.n * grid.n;
    size_t grid_bytes = grid_size * sizeof(double);
    
    vector<double> h_u(grid_size);
    vector<double> h_v(grid_size);
    
    // -------------------------------
    // 3) Allocate device memory
    // -------------------------------
    double *d_u = nullptr;
    double *d_v = nullptr;
    double *d_u_new = nullptr;
    double *d_v_new = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_u, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_v, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_new, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_v_new, grid_bytes));
    
    // -------------------------------
    // 4) Initialize with random values
    // -------------------------------
    // Define block and grid dimensions
    dim3 block_dim(16, 16);
    dim3 grid_dim((grid.n + block_dim.x - 1) / block_dim.x, 
                  (grid.n + block_dim.y - 1) / block_dim.y);
    
    unsigned int seed = 42;
    if (random_seed) {
        seed = static_cast<unsigned int>(std::time(nullptr));
    }
    
    initialize_random_kernel<<<grid_dim, block_dim>>>(d_u, d_v, grid.n, seed);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy data back to host for initial conditions
    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, grid_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v.data(), d_v, grid_bytes, cudaMemcpyDeviceToHost));
    
    // Write initial conditions to file
    write_to_csv(h_u, h_v, grid.n, "initial");
    
    // -------------------------------
    // 5) Main time-stepping loop
    // -------------------------------
    auto tstart = std::chrono::high_resolution_clock::now();
    
    // Allocate memory for checksum computation
    int num_blocks = grid_dim.x * grid_dim.y;
    double *d_partial_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, num_blocks * sizeof(double)));
    vector<double> h_partial_sums(num_blocks);
    
    double checksum = 0.0;
    
    for (int step = 0; step < grid.num_steps; ++step) {
        // Run simulation kernel
        simulate_kernel<<<grid_dim, block_dim>>>(d_u, d_v, d_u_new, d_v_new, 
                                                grid.n, grid.dx, grid.dt, 
                                                grid.alpha, grid.beta);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Swap pointers for next iteration
        std::swap(d_u, d_u_new);
        std::swap(d_v, d_v_new);
        
        // Save results at specified intervals
        if (step % grid.nsave == 0) {
            CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, grid_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_v.data(), d_v, grid_bytes, cudaMemcpyDeviceToHost));
            write_to_csv(h_u, h_v, grid.n, "u" + std::to_string(step));
        }
    }
    
    // Compute final checksum
    compute_checksum_kernel<<<grid_dim, block_dim, block_dim.x * block_dim.y * sizeof(double)>>>(
        d_u, d_v, d_partial_sums, grid.n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy partial sums back to host
    CUDA_CHECK(cudaMemcpy(h_partial_sums.data(), d_partial_sums, 
                         num_blocks * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Sum up partial results on CPU
    for (int i = 0; i < num_blocks; ++i) {
        checksum += h_partial_sums[i];
    }
    
    auto tend = std::chrono::high_resolution_clock::now();
    
    // -------------------------------
    // 6) Copy final results back to host
    // -------------------------------
    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, grid_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v.data(), d_v, grid_bytes, cudaMemcpyDeviceToHost));
    
    // Write final results
    write_to_csv(h_u, h_v, grid.n, "diffusion");
    
    // -------------------------------
    // 7) Display results and clean up
    // -------------------------------
    // Calculate center values
    int center_i = grid.n / 2;
    int center_j = grid.n / 2;
    double center_u = h_u[center_j * grid.n + center_i];
    double center_v = h_v[center_j * grid.n + center_i];
    
    cout << "CUDA simulation complete.\n"
         << "Final u at center : " << center_u << "\n"
         << "Final v at center : " << center_v << "\n"
         << "Checksum mag_sq   : " << checksum << "\n"
         << "Elapsed time [s]  : " << std::setw(6) << std::setprecision(5) 
         << (tend - tstart).count()*1e-9 << "\n"
         << "Initial conditions: data/initial_u.csv, data/initial_v.csv\n"
         << "Final conditions:   data/diffusion_u.csv, data/diffusion_v.csv\n";
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_u_new));
    CUDA_CHECK(cudaFree(d_v_new));
    CUDA_CHECK(cudaFree(d_partial_sums));
    
    return 0;
}