#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>       // For random number generation
#include <ctime>        // For time-based seed (optional)
#include <chrono>
#include <iomanip>


using namespace std;

void initialise_random(vector<double> &u, vector<double> &v, int &n, int &seed, bool &random_seed){
    // initialise field with random number
    // Use a Mersenne Twister PRNG and a uniform distribution [0,1].
    std::mt19937 rng(seed);
    if (random_seed==true){
        std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)));
    }
        

    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            u[j * n + i] = dist(rng);  // random in [0,1]
            v[j * n + i] = dist(rng);  // random in [0,1]
        }
    }
}

void simulate(int &num_steps, int &n, vector<double> &u, vector<double> &v, const double &dx, double &dt,
              double &alpha, double &beta, double &checksum){


    /*
    Main timestepping loop
    */

    //vector<double> u_new(n * n, 0.0);
    //vector<double> v_new(n * n, 0.0);

    for (int step = 0; step < num_steps; ++step) {
        
        // Update all grid points with periodic BC
        for (int j = 0; j < n; ++j) {
            int jm = (j - 1 + n) % n;  
            int jp = (j + 1) % n;

            for (int i = 0; i < n; ++i) {
                int im = (i - 1 + n) % n;  
                int ip = (i + 1) % n;

                int idx = j * n + i;

                double u_val = u[idx];
                double v_val = v[idx];

                // Periodic neighbors
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

                // PDE system:
                // du/dt = (Δu - αΔv) + u - (u - βv)(u^2 + v^2)
                // dv/dt = (αΔu + Δv) + v - (βu + v)(u^2 + v^2)
                double mag_sq = u_val * u_val + v_val * v_val;
                checksum += mag_sq;
                double rhs_u = (lap_u - alpha * lap_v) 
                             + u_val 
                             - (u_val - beta * v_val) * mag_sq;

                double rhs_v = (alpha * lap_u + lap_v) 
                             + v_val 
                             - (beta * u_val + v_val) * mag_sq;

                // Explicit update
                u[idx] = u_val + dt * rhs_u;
                v[idx] = v_val + dt * rhs_v;
            }
        }

        // Swap old and new
        //u = u_new;
        //v = v_new;
    }
}


int main(int argc, char **argv) {
    // -------------------------------
    // 1) Simulation Parameters
    // -------------------------------
    int n = 200;          // Number of grid points in each dimension
    double Lx = 200.0;      // Domain size in x-direction
    double Ly = 200.0;      // Domain size in y-direction

    // Time-stepping parameters
    double dt = 0.01;   
    int num_steps = 5000;
    bool random_seed = false;
    
    //tmax = 500
    //dt
    // PDE parameters
    //const double alpha = 1.0;
    double alpha = 2.0;
    double beta  = -0.5; 

    // checksum
    double checksum = 0;

    std::vector <std::string> argument({argv, argv+argc});

    for (long unsigned int i = 1; i<argument.size() ; i += 2){
        std::string arg = argument[i];
        if(arg=="-h"){ // Write help
            std::cout << "./fwc --n <Number of grid points in each dimension>"
                      << " --Lx, Ly <Domain size in x/y-direction>"
                      << " --dt <timestep size>"
                      << " --num_steps <Number of timesteps>"
                      << " --alpha <alpha CGLE>"
                      << " --beta <beta CGLE>"
                      << " --random_seed <whether to seed initial conditions randomly>\n";

            exit(0);
        } else if (i == argument.size() - 1){
            throw std::invalid_argument("The last argument (" + arg +") must have a value");
        } else if(arg=="--n"){
            if ((n = std::stoi(argument[i+1])) < 0) 
                throw std::invalid_argument("n must be positive (e.g. -n 1000)");
        } else if(arg=="--Lx"){
            if ((Lx = std::stoi(argument[i+1])) < 0) 
                throw std::invalid_argument("Lx must be positive (e.g. -Lx 1000)");
        } else if(arg=="--Ly"){
            if ((Ly = std::stoi(argument[i+1])) < 0) 
                throw std::invalid_argument("Ly must be positive (e.g. -Ly 1000)");
        } else if(arg=="--dt"){
            if ((dt = std::stod(argument[i+1])) < 0) 
                throw std::invalid_argument("dt must be positive (e.g. -dt 0.01)");
        } else if(arg=="--num_steps"){
            if ((num_steps = std::stoi(argument[i+1])) < 0) 
                throw std::invalid_argument("num_steps must be positive (e.g. -num_steps 1000)");
        } else if(arg=="--alpha"){
            if ((alpha = std::stod(argument[i+1])) < 0) 
                throw std::invalid_argument("alpha must be positive (e.g. -alpha 1.1)");
        } else if(arg=="--beta"){
            if ((beta = std::stod(argument[i+1])) < 0) 
                throw std::invalid_argument("alpha must be positive (e.g. -beta 1.1)");
        } else if(arg=="--random_seed"){
            (random_seed = std::stoi(argument[i+1]));
        } else{
            std::cout << "---> error: the argument type is not recognized \n";
        }
    }

    Ly = Lx; // dx!=dy not implemented!
    std::cout << "Set Ly = Lx because non square is not implemented!" << std::endl;
    const double dx = Lx / n;
    const double dy = Ly / n;

    // -------------------------------
    // 2) Allocate and Initialize Grids
    // -------------------------------
    vector<double> u(n * n, 0.0);
    vector<double> v(n * n, 0.0);

    // -------------------------------
    // 2a) Random Initial Conditions
    // -------------------------------
    int seed = 42;
    initialise_random(u, v, n, seed, random_seed);

    // -------------------------------
    // 2b) Write INITIAL conditions to CSV
    // -------------------------------
    {
        ofstream init_u("initial_u.csv");
        ofstream init_v("initial_v.csv");

        if (!init_u.is_open() || !init_v.is_open()) {
            cerr << "Error: Could not open initial condition CSV files." << endl;
            return 1;
        }

        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                init_u << u[j * n + i];
                init_v << v[j * n + i];
                if (i < n - 1) {
                    init_u << ",";
                    init_v << ",";
                }
            }
            init_u << "\n";
            init_v << "\n";
        }
        init_u.close();
        init_v.close();
    }

    // -------------------------------
    // 3) Main Time-Stepping Loop
    // -------------------------------
    auto tstart = std::chrono::high_resolution_clock::now();

    // num_steps, n, u, v, dx, alpha, beta, checksum
    simulate(num_steps, n, u, v, dx, dt, alpha, beta, checksum);
    auto tend = std::chrono::high_resolution_clock::now();


    // -------------------------------
    // 4) Write FINAL Grids to CSV
    // -------------------------------
    ofstream outfile_u("diffusion_u.csv");
    ofstream outfile_v("diffusion_v.csv");

    if (!outfile_u.is_open() || !outfile_v.is_open()) {
        cerr << "Error: Could not open output CSV files." << endl;
        return 1;
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

    // Print final center values for a quick check
    int center_idx = (n / 2) * n + (n / 2);
    cout << "Simulation complete.\n"
         << "Final u at center : " << u[center_idx] << "\n"
         << "Final v at center : " << v[center_idx] << "\n"
         << "Checksum mag_sq   : " << checksum << "\n"
         << "Elapsed time [s]  : " << std::setw(6) << std::setprecision(5) << (tend - tstart).count()*1e-9 << "\n"
         << "Initial conditions: initial_u.csv, initial_v.csv\n"
         << "Final conditions:   diffusion_u.csv, diffusion_v.csv\n";

    return 0;
}
