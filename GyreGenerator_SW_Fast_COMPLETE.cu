#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>

//============================================================
// Constants & Error Checking
//============================================================
const int WIDTH = 3600;
const int HEIGHT = 1800;
const int MONTHS = 12;


#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

//============================================================
// Physics Parameters
//============================================================
struct PhysicsParams {
    double R;              // Planet radius (m)
    double Omega;          // Rotation rate (rad/s)
    double g;              // Gravity (m/s²)
    double rho0;           // Reference density (kg/m³)
    double A_h;            // Horizontal viscosity (m²/s)
    double r_bottom;       // Bottom friction (1/s)
    double H_min;          // Minimum depth (m)
    double dt;             // Timestep (s)
    double dlat;           // Latitude spacing (rad)
    double dlon;           // Longitude spacing (rad)
    int steps_per_day;
    int spin_up_days;
};

//============================================================
// Device Helper Functions
//============================================================
__host__ __device__ void getWindStress(double lat, int month,
                                       double& tau_x, double& tau_y,
                                       const PhysicsParams& params)
{
    (void)month;
    (void)params;
    const double M_PI = 3.14159265358979323846;
    double lat_deg = lat * 180.0 / M_PI;
    double abs_lat = fabs(lat_deg);
    double tau0 = 0.15;  // N/m²
    
    if (abs_lat < 15.0) {
        tau_x = -tau0;              // tropical easterlies
    } else if (abs_lat < 45.0) {
        tau_x = +tau0;              // mid-lat westerlies
    } else if (abs_lat <= 75.0) {
        tau_x = -0.5 * tau0;        // polar easterlies
    } else {
        tau_x = 0.0;
    }
    
    tau_y = 0.0;  // purely zonal wind
}

__device__ __forceinline__ void getGridMetrics(double lat, const PhysicsParams& params,
                                               double& dx, double& dy)
{
    const double M_PI = 3.14159265358979323846;
    // Clamp latitude to avoid cos(lat)=0 at exact poles
    const double lat_max = 89.9 * M_PI / 180.0;
    double lat_safe = fmax(-lat_max, fmin(lat_max, lat));
    
    dx = params.R * cos(lat_safe) * params.dlon;
    dy = params.R * params.dlat;
}

__device__ __forceinline__ double coriolisF(double lat, const PhysicsParams& params)
{
    return 2.0 * params.Omega * sin(lat);
}

//============================================================
// Fused Dynamics Kernel - COMPLETE PHYSICS
//============================================================
__global__ void fused_dynamics_kernel(
    const double* __restrict__ u,
    const double* __restrict__ v,
    const double* __restrict__ eta,
    const double* __restrict__ H,
    const unsigned char* __restrict__ mask,
    double* __restrict__ dudt,
    double* __restrict__ dvdt,
    double* __restrict__ detadt,
    int month,
    int step,
    PhysicsParams params,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;
    int idx = y * width + x;

    // Land cells
    if (mask[idx] == 0) {
        dudt[idx] = 0.0;
        dvdt[idx] = 0.0;
        detadt[idx] = 0.0;
        return;
    }

    // Current state
    double u_curr = u[idx];
    double v_curr = v[idx];
    double eta_curr = eta[idx];
    double H_curr = fmax(H[idx], params.H_min);
    
    // Lat/lon
    double lat = (y - height / 2.0) * params.dlat;
    double lon = x * params.dlon;
    
    // Grid metrics (with pole protection)
    double dx, dy;
    getGridMetrics(lat, params, dx, dy);
    
    // Coriolis
    double f = coriolisF(lat, params);
    
    // Neighbor indices (periodic in x, clamped in y)
    int xp = (x + 1) % width;
    int xm = (x - 1 + width) % width;
    int yp = (y + 1 < height) ? y + 1 : y;
    int ym = (y > 0) ? y - 1 : y;
    
    int idx_xp = y * width + xp;
    int idx_xm = y * width + xm;
    int idx_yp = yp * width + x;
    int idx_ym = ym * width + x;
    
    // Read neighbors (global memory)
    double u_xp = u[idx_xp];
    double u_xm = u[idx_xm];
    double u_yp = u[idx_yp];
    double u_ym = u[idx_ym];
    
    double v_xp = v[idx_xp];
    double v_xm = v[idx_xm];
    double v_yp = v[idx_yp];
    double v_ym = v[idx_ym];
    
    double eta_xp = eta[idx_xp];
    double eta_xm = eta[idx_xm];
    double eta_yp = eta[idx_yp];
    double eta_ym = eta[idx_ym];
    
    // Derivatives
    double dudx = (u_xp - u_xm) / (2.0 * dx);
    double dudy = (u_yp - u_ym) / (2.0 * dy);
    double dvdx = (v_xp - v_xm) / (2.0 * dx);
    double dvdy = (v_yp - v_ym) / (2.0 * dy);
    
    double detadx = (eta_xp - eta_xm) / (2.0 * dx);
    double detady = (eta_yp - eta_ym) / (2.0 * dy);
    
    // Vorticity
    double zeta = dvdx - dudy;
    
    // Advection
    double advect_u = u_curr * dudx + v_curr * dudy;
    double advect_v = u_curr * dvdx + v_curr * dvdy;
    
    // Viscosity (Laplacian)
    double d2udx2 = (u_xp - 2.0 * u_curr + u_xm) / (dx * dx);
    double d2udy2 = (u_yp - 2.0 * u_curr + u_ym) / (dy * dy);
    double visc_u = params.A_h * (d2udx2 + d2udy2);
    
    double d2vdx2 = (v_xp - 2.0 * v_curr + v_xm) / (dx * dx);
    double d2vdy2 = (v_yp - 2.0 * v_curr + v_ym) / (dy * dy);
    double visc_v = params.A_h * (d2vdx2 + d2vdy2);
    
    // Wind stress
    double tau_x, tau_y;
    getWindStress(lat, month, tau_x, tau_y, params);
    
    double lon_perturbation = 0.15 * sin(3.0 * lon);
    tau_x *= (1.0 + lon_perturbation);
    tau_y *= (1.0 + lon_perturbation);
    
    double H_wind = 100.0;  // Surface mixed layer depth (m)
    
    // MOMENTUM EQUATIONS (complete shallow water)
    dudt[idx] = -advect_u 
                + (f + zeta) * v_curr 
                - params.g * detadx
                + tau_x / (params.rho0 * H_wind)
                - params.r_bottom * u_curr
                + visc_u;
    
    dvdt[idx] = -advect_v 
                - (f + zeta) * u_curr
                - params.g * detady
                + tau_y / (params.rho0 * H_wind)
                - params.r_bottom * v_curr
                + visc_v;
    
    // CONTINUITY EQUATION: d(eta)/dt = -div(D*u)
    // Read H at neighbors
    double H_xp = fmax(H[idx_xp], params.H_min);
    double H_xm = fmax(H[idx_xm], params.H_min);
    double H_yp = fmax(H[idx_yp], params.H_min);
    double H_ym = fmax(H[idx_ym], params.H_min);
    
    double D_xp = H_xp + eta_xp;
    double D_xm = H_xm + eta_xm;
    double D_yp = H_yp + eta_yp;
    double D_ym = H_ym + eta_ym;
    
    double Du_xp = D_xp * u_xp;
    double Du_xm = D_xm * u_xm;
    double dDudx = (Du_xp - Du_xm) / (2.0 * dx);
    
    double Dv_yp = D_yp * v_yp;
    double Dv_ym = D_ym * v_ym;
    double dDvdy = (Dv_yp - Dv_ym) / (2.0 * dy);
    
    detadt[idx] = -(dDudx + dDvdy);
}

//============================================================
// RK3 Update Kernels - CORRECT 3-STAGE
//============================================================
__global__ void rk3_stage1_kernel(
    double* __restrict__ field,
    const double* __restrict__ field0,
    const double* __restrict__ k1,
    const unsigned char* __restrict__ mask,
    double dt,
    int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    
    if (idx >= total) return;
    
    if (mask[idx] == 0) {
        field[idx] = 0.0;
        return;
    }
    
    field[idx] = field0[idx] + 0.5 * dt * k1[idx];
}

__global__ void rk3_stage2_kernel(
    double* __restrict__ field,
    const double* __restrict__ field0,
    const double* __restrict__ k1,
    const double* __restrict__ k2,
    const unsigned char* __restrict__ mask,
    double dt,
    int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    
    if (idx >= total) return;
    
    if (mask[idx] == 0) {
        field[idx] = 0.0;
        return;
    }
    
    field[idx] = field0[idx] + dt * (2.0 * k2[idx] - k1[idx]);
}

__global__ void rk3_final_kernel(
    double* __restrict__ field,
    const double* __restrict__ field0,
    const double* __restrict__ k1,
    const double* __restrict__ k2,
    const double* __restrict__ k3,
    const unsigned char* __restrict__ mask,
    double dt,
    int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    
    if (idx >= total) return;
    
    if (mask[idx] == 0) {
        field[idx] = 0.0;
        return;
    }
    
    field[idx] = field0[idx] + dt * (k1[idx] + 4.0 * k2[idx] + k3[idx]) / 6.0;
}

//============================================================
// Boundary Conditions
//============================================================
__global__ void apply_bc_kernel(
    double* __restrict__ u,
    double* __restrict__ v,
    const unsigned char* __restrict__ mask,
    int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    
    if (idx >= total) return;
    
    if (mask[idx] == 0) {
        u[idx] = 0.0;
        v[idx] = 0.0;
    }
}

//============================================================
// Monthly Accumulation
//============================================================
__global__ void accumulate_monthly_kernel(
    const double* __restrict__ u,
    const double* __restrict__ v,
    double* __restrict__ u_monthly,
    double* __restrict__ v_monthly,
    const unsigned char* __restrict__ mask,
    int month,
    int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    
    if (idx >= total) return;
    
    if (mask[idx] == 1) {
        int month_idx = idx * MONTHS + month;
        u_monthly[month_idx] += u[idx];
        v_monthly[month_idx] += v[idx];
    }
}

//============================================================
// File I/O
//============================================================
void loadBathymetry(const char* filename, std::vector<double>& H, std::vector<unsigned char>& mask)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        exit(1);
    }
    
    // Read header (28 bytes)
    char magic[4];
    uint32_t width, height, channels, dtype, version, depth;
    file.read(magic, 4);
    file.read(reinterpret_cast<char*>(&width), 4);
    file.read(reinterpret_cast<char*>(&height), 4);
    file.read(reinterpret_cast<char*>(&channels), 4);
    file.read(reinterpret_cast<char*>(&dtype), 4);
    file.read(reinterpret_cast<char*>(&version), 4);
    file.read(reinterpret_cast<char*>(&depth), 4);
    
    if (width != WIDTH || height != HEIGHT) {
        std::cerr << "Error: Bathymetry dimensions mismatch\n";
        exit(1);
    }
    
    // Read data
    std::vector<float> data(WIDTH * HEIGHT);
    file.read(reinterpret_cast<char*>(data.data()), WIDTH * HEIGHT * sizeof(float));
    
    // Convert to double and create mask
    int ocean_count = 0;
    double min_depth = 1e10, max_depth = -1e10;
    
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] < 0) {  // Ocean (negative depth)
            H[i] = -data[i];
            mask[i] = 1;
            ocean_count++;
            min_depth = std::min(min_depth, H[i]);
            max_depth = std::max(max_depth, H[i]);
        } else {  // Land
            H[i] = 0.0;
            mask[i] = 0;
        }
    }
    
    std::cout << "✓ Bathymetry loaded\n";
    std::cout << "   Ocean coverage: " << (100.0 * ocean_count / data.size()) << "% ";
    std::cout << "(" << ocean_count << " cells)\n";
    std::cout << "   Depth range: " << min_depth << " to " << max_depth << " m\n";
}

void saveVelocityField(const char* filename, const std::vector<double>& field)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot write " << filename << std::endl;
        return;
    }
    
    // Write header
    char magic[4] = {'C', 'L', 'M', 'T'};
    uint32_t width = WIDTH, height = HEIGHT, channels = MONTHS;
    uint32_t dtype = 2, version = 1, depth = 64;  // dtype=2 for double
    
    file.write(magic, 4);
    file.write(reinterpret_cast<const char*>(&width), 4);
    file.write(reinterpret_cast<const char*>(&height), 4);
    file.write(reinterpret_cast<const char*>(&channels), 4);
    file.write(reinterpret_cast<const char*>(&dtype), 4);
    file.write(reinterpret_cast<const char*>(&version), 4);
    file.write(reinterpret_cast<const char*>(&depth), 4);
    
    // Write data
    file.write(reinterpret_cast<const char*>(field.data()), field.size() * sizeof(double));
    
    double mb = (field.size() * sizeof(double)) / (1024.0 * 1024.0);
    std::cout << "✓ Saved " << mb << " MB to " << filename << std::endl;
}

//============================================================
// Main
//============================================================
int main(int argc, char** argv)
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <bathymetry.bath> <output_u.bin> <output_v.bin>\n";
        return 1;
    }
    
    std::cout << "🌊 Ocean Gyre Generator - COMPLETE PHYSICS\n";
    std::cout << "==================================================\n";
    const double M_PI = 3.14159265358979323846;
    // Physics parameters
    PhysicsParams params;
    params.R = 6.371e6;                    // Earth radius (m)
    params.Omega = 7.2921e-5;              // Earth rotation (rad/s)
    params.g = 9.81;                       // Gravity (m/s²)
    params.rho0 = 1025.0;                  // Seawater density (kg/m³)
    params.A_h = 2000.0;                   // Horizontal viscosity (m²/s)
    params.r_bottom = 1.0 / (30.0 * 86400.0);  // 30-day damping
    params.H_min = 10.0;                   // Minimum depth (m)
    params.spin_up_days = 365;             // 1 year spin-up
    params.dlat = M_PI / HEIGHT;
    params.dlon = 2.0 * M_PI / WIDTH;
    
    // CFL-safe timestep
    double dx_min = params.R * params.dlon;  // ~11 km at equator
    double c_gravity = sqrt(params.g * 5000.0);  // ~220 m/s
    double dt_cfl = 0.5 * dx_min / c_gravity;    // ~25 s
    params.dt = 17.7509;  // Slightly smaller for safety
    params.steps_per_day = static_cast<int>(86400.0 / params.dt);
    
    std::cout << "Grid: " << WIDTH << " × " << HEIGHT << "\n";
    std::cout << "Timestep: " << params.dt << " s (" << params.steps_per_day << " steps/day)\n";
    std::cout << "Spin-up: " << params.spin_up_days << " days\n";
    
    // Load bathymetry
    size_t N = WIDTH * HEIGHT;
    size_t bytes = N * sizeof(double);
    size_t bytes_mask = N * sizeof(unsigned char);
    
    std::vector<double> H(N);
    std::vector<unsigned char> mask(N);
    
    std::cout << "📂 Loading bathymetry: " << argv[1] << "\n";
    loadBathymetry(argv[1], H, mask);
    
    // Host verification of wind stress
    std::cout << "\n🌬️  Wind Stress Check:\n";
    for (double lat_deg : {0.0, 15.0, 30.0, 45.0}) {
        double lat = lat_deg * M_PI / 180.0;
        double tau_x, tau_y;
        getWindStress(lat, 0, tau_x, tau_y, params);
        std::cout << "   " << lat_deg << "°: tau_x=" << tau_x << " N/m², tau_y=" << tau_y << " N/m²\n";
    }
    
    // Allocate device memory
    double *d_u, *d_v, *d_eta, *d_H;
    double *d_u0, *d_v0, *d_eta0;
    double *d_k1_u, *d_k1_v, *d_k1_eta;
    double *d_k2_u, *d_k2_v, *d_k2_eta;
    double *d_k3_u, *d_k3_v, *d_k3_eta;
    double *d_u_monthly, *d_v_monthly;
    unsigned char *d_mask;
    
    CUDA_CHECK(cudaMalloc(&d_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMalloc(&d_eta, bytes));
    CUDA_CHECK(cudaMalloc(&d_H, bytes));
    CUDA_CHECK(cudaMalloc(&d_u0, bytes));
    CUDA_CHECK(cudaMalloc(&d_v0, bytes));
    CUDA_CHECK(cudaMalloc(&d_eta0, bytes));
    CUDA_CHECK(cudaMalloc(&d_k1_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_k1_v, bytes));
    CUDA_CHECK(cudaMalloc(&d_k1_eta, bytes));
    CUDA_CHECK(cudaMalloc(&d_k2_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_k2_v, bytes));
    CUDA_CHECK(cudaMalloc(&d_k2_eta, bytes));
    CUDA_CHECK(cudaMalloc(&d_k3_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_k3_v, bytes));
    CUDA_CHECK(cudaMalloc(&d_k3_eta, bytes));
    CUDA_CHECK(cudaMalloc(&d_u_monthly, N * MONTHS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v_monthly, N * MONTHS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mask, bytes_mask));
    
    double gpu_mb = (bytes * 17 + N * MONTHS * sizeof(double) * 2 + bytes_mask) / (1024.0 * 1024.0);
    std::cout << "\n🎮 GPU memory: ~" << gpu_mb << " MB\n";
    
    // Initialize
    CUDA_CHECK(cudaMemset(d_u, 0, bytes));
    CUDA_CHECK(cudaMemset(d_v, 0, bytes));
    CUDA_CHECK(cudaMemset(d_eta, 0, bytes));
    CUDA_CHECK(cudaMemcpy(d_H, H.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, mask.data(), bytes_mask, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_u_monthly, 0, N * MONTHS * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_v_monthly, 0, N * MONTHS * sizeof(double)));
    
    // Grid configuration
    dim3 block_2d(32, 32);
    dim3 grid_2d((WIDTH + 31) / 32, (HEIGHT + 31) / 32);
    
    dim3 block_1d(256);
    dim3 grid_1d((N + 255) / 256);
    
    // Time integration
    std::cout << "\n🌊 Spinning up ocean gyres (COMPLETE PHYSICS)\n";
    std::cout << "   Days: " << params.spin_up_days << "\n";
    
    int total_steps = params.spin_up_days * params.steps_per_day;
    int accumulation_start = total_steps * 2 / 3;  // Last 1/3 of simulation
    std::vector<int> month_samples(MONTHS, 0);
    
    for (int step = 0; step < total_steps; step++) {
        int month = ((step / params.steps_per_day) % 365) / 30;
        month = std::min(month, MONTHS - 1);
        
        // Save initial state
        CUDA_CHECK(cudaMemcpy(d_u0, d_u, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_v0, d_v, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_eta0, d_eta, bytes, cudaMemcpyDeviceToDevice));
        
        // === RK3 STAGE 1 ===
        fused_dynamics_kernel<<<grid_2d, block_2d>>>(
            d_u, d_v, d_eta, d_H, d_mask, d_k1_u, d_k1_v, d_k1_eta, month, step, params, WIDTH, HEIGHT);
        
        rk3_stage1_kernel<<<grid_1d, block_1d>>>(d_u, d_u0, d_k1_u, d_mask, params.dt, WIDTH, HEIGHT);
        rk3_stage1_kernel<<<grid_1d, block_1d>>>(d_v, d_v0, d_k1_v, d_mask, params.dt, WIDTH, HEIGHT);
        rk3_stage1_kernel<<<grid_1d, block_1d>>>(d_eta, d_eta0, d_k1_eta, d_mask, params.dt, WIDTH, HEIGHT);
        
        // === RK3 STAGE 2 ===
        fused_dynamics_kernel<<<grid_2d, block_2d>>>(
            d_u, d_v, d_eta, d_H, d_mask, d_k2_u, d_k2_v, d_k2_eta, month, step, params, WIDTH, HEIGHT);
        
        rk3_stage2_kernel<<<grid_1d, block_1d>>>(d_u, d_u0, d_k1_u, d_k2_u, d_mask, params.dt, WIDTH, HEIGHT);
        rk3_stage2_kernel<<<grid_1d, block_1d>>>(d_v, d_v0, d_k1_v, d_k2_v, d_mask, params.dt, WIDTH, HEIGHT);
        rk3_stage2_kernel<<<grid_1d, block_1d>>>(d_eta, d_eta0, d_k1_eta, d_k2_eta, d_mask, params.dt, WIDTH, HEIGHT);
        
        // === RK3 STAGE 3 ===
        fused_dynamics_kernel<<<grid_2d, block_2d>>>(
            d_u, d_v, d_eta, d_H, d_mask, d_k3_u, d_k3_v, d_k3_eta, month, step, params, WIDTH, HEIGHT);
        
        rk3_final_kernel<<<grid_1d, block_1d>>>(d_u, d_u0, d_k1_u, d_k2_u, d_k3_u, d_mask, params.dt, WIDTH, HEIGHT);
        rk3_final_kernel<<<grid_1d, block_1d>>>(d_v, d_v0, d_k1_v, d_k2_v, d_k3_v, d_mask, params.dt, WIDTH, HEIGHT);
        rk3_final_kernel<<<grid_1d, block_1d>>>(d_eta, d_eta0, d_k1_eta, d_k2_eta, d_k3_eta, d_mask, params.dt, WIDTH, HEIGHT);
        
        apply_bc_kernel<<<grid_1d, block_1d>>>(d_u, d_v, d_mask, WIDTH, HEIGHT);
        
        // Accumulate monthly averages
        if (step >= accumulation_start) {
            accumulate_monthly_kernel<<<grid_1d, block_1d>>>(
                d_u, d_v, d_u_monthly, d_v_monthly, d_mask, month, WIDTH, HEIGHT);
            month_samples[month]++;
        }
        
        if (step % (params.steps_per_day * 30) == 0) {
            std::cout << "  Day " << (step / params.steps_per_day) << "/" << params.spin_up_days << "\r" << std::flush;
        }
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "\n✓ Spin-up complete\n";
    
    // Normalize and save
    std::vector<double> u_monthly(N * MONTHS);
    std::vector<double> v_monthly(N * MONTHS);
    std::vector<double> h_u_monthly(N * MONTHS);
    std::vector<double> h_v_monthly(N * MONTHS);
    
    CUDA_CHECK(cudaMemcpy(h_u_monthly.data(), d_u_monthly, N * MONTHS * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v_monthly.data(), d_v_monthly, N * MONTHS * sizeof(double), cudaMemcpyDeviceToHost));
    
    double max_u = 0.0, max_v = 0.0, max_speed = 0.0, sum_speed = 0.0;
    int ocean_count = 0;
    
    for (int m = 0; m < MONTHS; m++) {
        if (month_samples[m] > 0) {
            double norm = 1.0 / month_samples[m];
            for (size_t i = 0; i < N; i++) {
                u_monthly[i * MONTHS + m] = h_u_monthly[i * MONTHS + m] * norm;
                v_monthly[i * MONTHS + m] = h_v_monthly[i * MONTHS + m] * norm;
                
                if (mask[i] == 1) {
                    double u_val = u_monthly[i * MONTHS + m];
                    double v_val = v_monthly[i * MONTHS + m];
                    double speed = hypot(u_val, v_val);
                    
                    max_u = std::max(max_u, fabs(u_val));
                    max_v = std::max(max_v, fabs(v_val));
                    max_speed = std::max(max_speed, speed);
                    sum_speed += speed;
                    ocean_count++;
                }
            }
        }
    }
    
    std::cout << "\n📊 VELOCITY DIAGNOSTICS:\n";
    std::cout << "   Max |u|: " << max_u << " m/s\n";
    std::cout << "   Max |v|: " << max_v << " m/s\n";
    std::cout << "   Max speed: " << max_speed << " m/s\n";
    std::cout << "   Mean speed: " << (sum_speed / ocean_count) << " m/s\n";
    
    if (max_v < 0.01) {
        std::cout << "   ⚠️  WARNING: v-component very weak - check Coriolis term!\n";
    }
    
    saveVelocityField(argv[2], u_monthly);
    saveVelocityField(argv[3], v_monthly);
    
    // Cleanup
    cudaFree(d_u); cudaFree(d_v); cudaFree(d_eta); cudaFree(d_H);
    cudaFree(d_u0); cudaFree(d_v0); cudaFree(d_eta0);
    cudaFree(d_k1_u); cudaFree(d_k1_v); cudaFree(d_k1_eta);
    cudaFree(d_k2_u); cudaFree(d_k2_v); cudaFree(d_k2_eta);
    cudaFree(d_k3_u); cudaFree(d_k3_v); cudaFree(d_k3_eta);
    cudaFree(d_u_monthly); cudaFree(d_v_monthly); cudaFree(d_mask);
    
    std::cout << "\n✓ Complete!\n";
    return 0;
}
