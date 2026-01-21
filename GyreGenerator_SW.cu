//============================================================
// GyreGenerator_SW.cu
// Barotropic Shallow-Water Ocean Gyre Generator
// Physics-based monthly ocean circulation using GPU acceleration
//
// Solves 1-layer shallow-water equations with:
// - Wind stress forcing (seasonal variation)
// - Coriolis effects (spherical coordinates)
// - Bottom friction
// - Horizontal viscosity
// - Free surface dynamics
//
// Author: Mark Devereux (2025)
//============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include "ClimateFileFormat.h"

constexpr int WIDTH = STANDARD_WIDTH;
constexpr int HEIGHT = STANDARD_HEIGHT;
constexpr int MONTHS = 12;
constexpr double M_PI = 3.14159265358979323846;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

//============================================================
// Physical Constants & Parameters
//============================================================
struct PhysicsParams {
    // Planetary
    double R;           // Planet radius (m) - 6.371e6
    double omega;       // Rotation rate (rad/s) - 7.2921e-5
    double g;           // Gravity (m/s²) - 9.81
    double rho0;        // Reference density (kg/m³) - 1025
    
    // Grid
    double dlon;        // Longitude spacing (rad)
    double dlat;        // Latitude spacing (rad)
    
    // Friction/viscosity
    double r_bottom;    // Bottom friction (1/s) - 1/(30 days)
    double A_h;         // Horizontal viscosity (m²/s) - 1000-5000
    
    // Time stepping
    double dt;          // Time step (s)
    int steps_per_day;  // Timesteps in one day
    int spin_up_days;   // Days to spin up (3650 = 10 years)
    
    // Minimum depth for stability
    double H_min;       // Minimum ocean depth (m) - 50
};

//============================================================
// Wind Stress - Analytic Seasonal Forcing
//============================================================
__device__ void getWindStress(double lat, int month, 
                              double& tau_x, double& tau_y,
                              const PhysicsParams& params)
{
    // Zonal wind stress (tau_x) - trades, westerlies, easterlies
    // Seasonal modulation with 6-month cycle
    double phase = 2.0 * M_PI * (month - 1) / 12.0;
    double seasonal = 1.0 + 0.3 * cos(phase);
    
    double lat_deg = lat * 180.0 / M_PI;
    
    // Subtropical westerlies (30-60°)
    double west_strength = 0.15 * seasonal; // N/m²
    double west_lat_center = 45.0;
    double west_lat_width = 15.0;
    double westerlies = west_strength * exp(-pow((lat_deg - west_lat_center) / west_lat_width, 2));
    westerlies += west_strength * exp(-pow((lat_deg + west_lat_center) / west_lat_width, 2));
    
    // Trade winds (0-30°)
    double trade_strength = -0.08 * seasonal; // Easterlies
    double trade_lat_center = 15.0;
    double trade_lat_width = 12.0;
    double trades = trade_strength * exp(-pow((lat_deg - trade_lat_center) / trade_lat_width, 2));
    trades += trade_strength * exp(-pow((lat_deg + trade_lat_center) / trade_lat_width, 2));
    
    tau_x = westerlies + trades;
    
    // Meridional wind stress (tau_y) - weaker cross-equatorial flow
    tau_y = 0.02 * sin(2.0 * lat) * seasonal;
}

//============================================================
// Grid Metrics on Sphere
//============================================================
__device__ void getGridMetrics(double lat, const PhysicsParams& params,
                               double& dx, double& dy)
{
    dx = params.R * cos(lat) * params.dlon;
    dy = params.R * params.dlat;
}

__device__ double coriolisF(double lat, const PhysicsParams& params)
{
    return 2.0 * params.omega * sin(lat);
}

__device__ double coriolisBeta(double lat, const PhysicsParams& params)
{
    return 2.0 * params.omega * cos(lat) / params.R;
}

//============================================================
// Spatial Derivatives (2nd-order centered)
//============================================================
__device__ double ddx(const double* __restrict__ field, int x, int y,
                     int width, int height, double dx)
{
    int xp = (x + 1) % width;
    int xm = (x - 1 + width) % width;
    return (field[y * width + xp] - field[y * width + xm]) / (2.0 * dx);
}

__device__ double ddy(const double* __restrict__ field, int x, int y,
                     int width, int height, double dy)
{
    if (y == 0 || y == height - 1) return 0.0; // Boundary
    return (field[(y + 1) * width + x] - field[(y - 1) * width + x]) / (2.0 * dy);
}

__device__ double laplacian(const double* __restrict__ field, int x, int y,
                           int width, int height, double dx, double dy,
                           const unsigned char* __restrict__ mask)
{
    int idx = y * width + x;
    if (mask[idx] == 0) return 0.0; // Land
    
    int xp = (x + 1) % width;
    int xm = (x - 1 + width) % width;
    
    double d2dx2 = 0.0, d2dy2 = 0.0;
    
    // x-direction (periodic)
    d2dx2 = (field[y * width + xp] - 2.0 * field[idx] + field[y * width + xm]) / (dx * dx);
    
    // y-direction (with boundaries)
    if (y > 0 && y < height - 1) {
        d2dy2 = (field[(y + 1) * width + x] - 2.0 * field[idx] + field[(y - 1) * width + x]) / (dy * dy);
    }
    
    return d2dx2 + d2dy2;
}

//============================================================
// Momentum Equations (RK3 step)
//============================================================
__global__ void momentum_kernel(
    const double* __restrict__ u,
    const double* __restrict__ v,
    const double* __restrict__ eta,
    const double* __restrict__ H,
    const unsigned char* __restrict__ mask,
    double* __restrict__ dudt,
    double* __restrict__ dvdt,
    int month,
    PhysicsParams params,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    if (mask[idx] == 0) {
        dudt[idx] = 0.0;
        dvdt[idx] = 0.0;
        return;
    }
    
    double lat = (y - height / 2.0) * params.dlat;
    double lon = x * params.dlon;
    
    double dx, dy;
    getGridMetrics(lat, params, dx, dy);
    
    double f = coriolisF(lat, params);
    
    // Current state
    double u_curr = u[idx];
    double v_curr = v[idx];
    double eta_curr = eta[idx];
    double H_curr = fmax(H[idx], params.H_min);
    double D = H_curr + eta_curr;
    
    // Relative vorticity: zeta = dv/dx - du/dy
    double dvdx = ddx(v, x, y, width, height, dx);
    double dudy = ddy(u, x, y, width, height, dy);
    double zeta = dvdx - dudy;
    
    // Pressure gradient
    double detadx = ddx(eta, x, y, width, height, dx);
    double detady = ddy(eta, x, y, width, height, dy);
    
    // Advection terms
    double dudx = ddx(u, x, y, width, height, dx);     
    double dvdy = ddy(v, x, y, width, height, dy);
    
    double advect_u = u_curr * dudx + v_curr * dudy;
    double advect_v = u_curr * dvdx + v_curr * dvdy;
    
    // Wind stress
    double tau_x, tau_y;
    getWindStress(lat, month, tau_x, tau_y, params);
    
    // Viscosity
    double visc_u = params.A_h * laplacian(u, x, y, width, height, dx, dy, mask);
    double visc_v = params.A_h * laplacian(v, x, y, width, height, dx, dy, mask);
    
    // Momentum equations
    // du/dt = -(u·∇u) + (f+ζ)v - g∂η/∂x + τx/(ρ₀H) - r·u + Ah∇²u
    dudt[idx] = -advect_u 
                + (f + zeta) * v_curr 
                - params.g * detadx
                + tau_x / (params.rho0 * H_curr)
                - params.r_bottom * u_curr
                + visc_u;
    
    // dv/dt = -(v·∇v) - (f+ζ)u - g∂η/∂y + τy/(ρ₀H) - r·v + Ah∇²v
    dvdt[idx] = -advect_v 
                - (f + zeta) * u_curr
                - params.g * detady
                + tau_y / (params.rho0 * H_curr)
                - params.r_bottom * v_curr
                + visc_v;
}

//============================================================
// Continuity Equation (RK3 step)
//============================================================
__global__ void continuity_kernel(
    const double* __restrict__ u,
    const double* __restrict__ v,
    const double* __restrict__ eta,
    const double* __restrict__ H,
    const unsigned char* __restrict__ mask,
    double* __restrict__ detadt,
    PhysicsParams params,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    if (mask[idx] == 0) {
        detadt[idx] = 0.0;
        return;
    }
    
    double lat = (y - height / 2.0) * params.dlat;
    double dx, dy;
    getGridMetrics(lat, params, dx, dy);
    
    double H_curr = fmax(H[idx], params.H_min);
    double D = H_curr + eta[idx];
    
    // Transport divergence: ∂(Du)/∂x + ∂(Dv)/∂y
    int xp = (x + 1) % width;
    int xm = (x - 1 + width) % width;
    
    double Du_xp = (H[y * width + xp] + eta[y * width + xp]) * u[y * width + xp];
    double Du_xm = (H[y * width + xm] + eta[y * width + xm]) * u[y * width + xm];
    double dDudx = (Du_xp - Du_xm) / (2.0 * dx);
    
    double dDvdy = 0.0;
    if (y > 0 && y < height - 1) {
        double Dv_yp = (H[(y + 1) * width + x] + eta[(y + 1) * width + x]) * v[(y + 1) * width + x];
        double Dv_ym = (H[(y - 1) * width + x] + eta[(y - 1) * width + x]) * v[(y - 1) * width + x];
        dDvdy = (Dv_yp - Dv_ym) / (2.0 * dy);
    }
    
    // dη/dt = -∇·(D u)
    detadt[idx] = -(dDudx + dDvdy);
}

//============================================================
// RK3 Time Integration
//============================================================
__global__ void rk3_update_kernel(
    double* __restrict__ field,
    const double* __restrict__ field0,
    const double* __restrict__ k1,
    const double* __restrict__ k2,
    const double* __restrict__ k3,
    const unsigned char* __restrict__ mask,
    double dt,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    if (mask[idx] == 0) {
        field[idx] = 0.0;
        return;
    }
    
    // RK3: y_n+1 = y_n + dt/6 * (k1 + 4*k2 + k3)
    field[idx] = field0[idx] + dt * (k1[idx] + 4.0 * k2[idx] + k3[idx]) / 6.0;
}

//============================================================
// Apply Boundary Conditions (no-normal-flow, free-slip)
//============================================================
__global__ void apply_bc_kernel(
    double* __restrict__ u,
    double* __restrict__ v,
    const unsigned char* __restrict__ mask,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Land cells: zero velocity
    if (mask[idx] == 0) {
        u[idx] = 0.0;
        v[idx] = 0.0;
        return;
    }
    
    // Ocean cells adjacent to land: enforce no-normal-flow
    int xp = (x + 1) % width;
    int xm = (x - 1 + width) % width;
    
    bool land_east = (mask[y * width + xp] == 0);
    bool land_west = (mask[y * width + xm] == 0);
    bool land_north = (y < height - 1) ? (mask[(y + 1) * width + x] == 0) : true;
    bool land_south = (y > 0) ? (mask[(y - 1) * width + x] == 0) : true;
    
    if (land_east || land_west) u[idx] = 0.0;
    if (land_north || land_south) v[idx] = 0.0;
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
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int month_idx = idx * MONTHS + month;
    
    if (mask[idx] == 1) {
        u_monthly[month_idx] += u[idx];
        v_monthly[month_idx] += v[idx];
    }
}

//============================================================
// Main Integration Loop
//============================================================
void runShallowWaterModel(
    std::vector<double>& u_monthly,
    std::vector<double>& v_monthly,
    const std::vector<unsigned char>& mask,
    const std::vector<double>& bathymetry,
    PhysicsParams params)
{
    size_t N = WIDTH * HEIGHT;
    size_t bytes = N * sizeof(double);
    size_t bytes_mask = N * sizeof(unsigned char);
    
    // Allocate device memory
    double *d_u, *d_v, *d_eta, *d_H;
    double *d_u0, *d_v0, *d_eta0;
    double *d_dudt, *d_dvdt, *d_detadt;
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
    CUDA_CHECK(cudaMalloc(&d_dudt, bytes));
    CUDA_CHECK(cudaMalloc(&d_dvdt, bytes));
    CUDA_CHECK(cudaMalloc(&d_detadt, bytes));
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
    
    // Initialize: rest state
    CUDA_CHECK(cudaMemset(d_u, 0, bytes));
    CUDA_CHECK(cudaMemset(d_v, 0, bytes));
    CUDA_CHECK(cudaMemset(d_eta, 0, bytes));
    CUDA_CHECK(cudaMemset(d_u_monthly, 0, N * MONTHS * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_v_monthly, 0, N * MONTHS * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(d_H, bathymetry.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, mask.data(), bytes_mask, cudaMemcpyHostToDevice));
    
    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
    
    std::cout << "🌊 Spinning up ocean gyres (" << params.spin_up_days << " days)...\n";
    
    int total_steps = params.spin_up_days * params.steps_per_day;
    int accumulation_start = (params.spin_up_days - 365) * params.steps_per_day; // Last year
    int month_samples[12] = {0};
    
    for (int step = 0; step < total_steps; step++) {
        
        // Current month (repeating annual cycle)
        int day_of_year = (step / params.steps_per_day) % 365;
        int month = day_of_year / 30; // Rough monthly division
        if (month >= 12) month = 11;
        
        // Save initial state for RK3
        CUDA_CHECK(cudaMemcpy(d_u0, d_u, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_v0, d_v, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_eta0, d_eta, bytes, cudaMemcpyDeviceToDevice));
        
        // RK3 Stage 1: k1 = f(y_n)
        momentum_kernel<<<grid, block>>>(d_u, d_v, d_eta, d_H, d_mask, d_k1_u, d_k1_v, month, params, WIDTH, HEIGHT);
        continuity_kernel<<<grid, block>>>(d_u, d_v, d_eta, d_H, d_mask, d_k1_eta, params, WIDTH, HEIGHT);
        
        // Update to y_n + dt/2 * k1
        rk3_update_kernel<<<grid, block>>>(d_u, d_u0, d_k1_u, d_k1_u, d_k1_u, d_mask, params.dt / 2.0, WIDTH, HEIGHT);
        rk3_update_kernel<<<grid, block>>>(d_v, d_v0, d_k1_v, d_k1_v, d_k1_v, d_mask, params.dt / 2.0, WIDTH, HEIGHT);
        rk3_update_kernel<<<grid, block>>>(d_eta, d_eta0, d_k1_eta, d_k1_eta, d_k1_eta, d_mask, params.dt / 2.0, WIDTH, HEIGHT);
        apply_bc_kernel<<<grid, block>>>(d_u, d_v, d_mask, WIDTH, HEIGHT);
        
        // RK3 Stage 2: k2 = f(y_n + dt/2 * k1)
        momentum_kernel<<<grid, block>>>(d_u, d_v, d_eta, d_H, d_mask, d_k2_u, d_k2_v, month, params, WIDTH, HEIGHT);
        continuity_kernel<<<grid, block>>>(d_u, d_v, d_eta, d_H, d_mask, d_k2_eta, params, WIDTH, HEIGHT);
        
        // Update to y_n + dt * (2*k2 - k1)
        CUDA_CHECK(cudaMemcpy(d_u, d_u0, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, d_v0, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_eta, d_eta0, bytes, cudaMemcpyDeviceToDevice));
        
        rk3_update_kernel<<<grid, block>>>(d_u, d_u0, d_k2_u, d_k2_u, d_k1_u, d_mask, params.dt, WIDTH, HEIGHT);
        rk3_update_kernel<<<grid, block>>>(d_v, d_v0, d_k2_v, d_k2_v, d_k1_v, d_mask, params.dt, WIDTH, HEIGHT);
        rk3_update_kernel<<<grid, block>>>(d_eta, d_eta0, d_k2_eta, d_k2_eta, d_k1_eta, d_mask, params.dt, WIDTH, HEIGHT);
        apply_bc_kernel<<<grid, block>>>(d_u, d_v, d_mask, WIDTH, HEIGHT);
        
        // RK3 Stage 3: k3 = f(y_n + dt * (2*k2 - k1))
        momentum_kernel<<<grid, block>>>(d_u, d_v, d_eta, d_H, d_mask, d_k3_u, d_k3_v, month, params, WIDTH, HEIGHT);
        continuity_kernel<<<grid, block>>>(d_u, d_v, d_eta, d_H, d_mask, d_k3_eta, params, WIDTH, HEIGHT);
        
        // Final update: y_n+1 = y_n + dt/6 * (k1 + 4*k2 + k3)
        rk3_update_kernel<<<grid, block>>>(d_u, d_u0, d_k1_u, d_k2_u, d_k3_u, d_mask, params.dt, WIDTH, HEIGHT);
        rk3_update_kernel<<<grid, block>>>(d_v, d_v0, d_k1_v, d_k2_v, d_k3_v, d_mask, params.dt, WIDTH, HEIGHT);
        rk3_update_kernel<<<grid, block>>>(d_eta, d_eta0, d_k1_eta, d_k2_eta, d_k3_eta, d_mask, params.dt, WIDTH, HEIGHT);
        apply_bc_kernel<<<grid, block>>>(d_u, d_v, d_mask, WIDTH, HEIGHT);
        
        // Accumulate monthly averages (last year only)
        if (step >= accumulation_start) {
            accumulate_monthly_kernel<<<grid, block>>>(d_u, d_v, d_u_monthly, d_v_monthly, d_mask, month, WIDTH, HEIGHT);
            month_samples[month]++;
        }
        
        if (step % (params.steps_per_day * 30) == 0) {
            std::cout << "  Day " << (step / params.steps_per_day) << "/" << params.spin_up_days << "\r" << std::flush;
        }
    }
    
    std::cout << "\n✓ Spin-up complete\n";
    
    // Copy results and normalize
    std::vector<double> h_u_monthly(N * MONTHS);
    std::vector<double> h_v_monthly(N * MONTHS);
    CUDA_CHECK(cudaMemcpy(h_u_monthly.data(), d_u_monthly, N * MONTHS * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v_monthly.data(), d_v_monthly, N * MONTHS * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Normalize by sample count
    for (int m = 0; m < MONTHS; m++) {
        if (month_samples[m] > 0) {
            double norm = 1.0 / month_samples[m];
            for (size_t i = 0; i < N; i++) {
                u_monthly[i * MONTHS + m] = h_u_monthly[i * MONTHS + m] * norm;
                v_monthly[i * MONTHS + m] = h_v_monthly[i * MONTHS + m] * norm;
            }
        }
    }
    
    // Cleanup
    cudaFree(d_u); cudaFree(d_v); cudaFree(d_eta); cudaFree(d_H);
    cudaFree(d_u0); cudaFree(d_v0); cudaFree(d_eta0);
    cudaFree(d_dudt); cudaFree(d_dvdt); cudaFree(d_detadt);
    cudaFree(d_k1_u); cudaFree(d_k1_v); cudaFree(d_k1_eta);
    cudaFree(d_k2_u); cudaFree(d_k2_v); cudaFree(d_k2_eta);
    cudaFree(d_k3_u); cudaFree(d_k3_v); cudaFree(d_k3_eta);
    cudaFree(d_u_monthly); cudaFree(d_v_monthly);
    cudaFree(d_mask);
}

//============================================================
// Main
//============================================================
int main(int argc, char** argv)
{
    std::cout << "🌊 Ocean Gyre Generator - Shallow Water Model\n";
    std::cout << "================================================\n\n";
    
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <bathymetry.bath> <output_u> <output_v>\n";
        return 1;
    }
    
    std::string bath_file = argv[1];
    std::string output_u = argv[2];
    std::string output_v = argv[3];
    
    // Load bathymetry (same format as PNPLGenerator)
    std::cout << "📊 Loading bathymetry: " << bath_file << "\n";
    std::ifstream bathFile(bath_file, std::ios::binary);
    if (!bathFile) {
        std::cerr << "❌ Cannot open bathymetry file\n";
        return 1;
    }
    
    ClimateFileHeader bath_header;
    if (!readClimateHeader(bathFile, bath_header)) {
        std::cerr << "❌ Failed to read bathymetry header\n";
        return 1;
    }
    
    if (bath_header.width != WIDTH || bath_header.height != HEIGHT) {
        std::cerr << "❌ Bathymetry size mismatch\n";
        return 1;
    }
    
    // Read float bathymetry data (starts at byte 28)
    std::vector<float> bath_float(WIDTH * HEIGHT);
    bathFile.read(reinterpret_cast<char*>(bath_float.data()), 
                  WIDTH * HEIGHT * sizeof(float));
    bathFile.close();
    
    // Convert to double and create mask
    std::vector<unsigned char> mask(WIDTH * HEIGHT);
    std::vector<double> bathymetry(WIDTH * HEIGHT);
    
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        float depth = bath_float[i];
        if (depth > 0.0f) {
            mask[i] = 1; // Ocean
            bathymetry[i] = static_cast<double>(depth);
        } else {
            mask[i] = 0; // Land
            bathymetry[i] = 0.0;
        }
    }
    
    std::cout << "✅ Bathymetry loaded\n";
    
    // Physics parameters
    PhysicsParams params;
    params.R = 6.371e6;                  // Earth radius
    params.omega = 7.2921e-5;            // Earth rotation
    params.g = 9.81;
    params.rho0 = 1025.0;
    params.dlon = 2.0 * M_PI / WIDTH;
    params.dlat = M_PI / HEIGHT;
    params.r_bottom = 1.0 / (30.0 * 86400.0); // 30-day damping
    params.A_h = 2000.0;                 // Horizontal viscosity
    params.H_min = 50.0;                 // Minimum depth
    
    // CFL-stable timestep
    double dx_min = params.R * params.dlon * cos(M_PI / 4.0);
    double c_max = sqrt(params.g * 5000.0); // Assume max depth 5km
    params.dt = 0.5 * dx_min / c_max;    // CFL factor 0.5
    params.steps_per_day = (int)(86400.0 / params.dt);
    params.spin_up_days = 3650;          // 10 years
    
    std::cout << "Grid: " << WIDTH << " × " << HEIGHT << "\n";
    std::cout << "Timestep: " << params.dt << " s (" << params.steps_per_day << " steps/day)\n";
    std::cout << "Spin-up: " << params.spin_up_days << " days\n\n";
    
    // Run model
    std::vector<double> u_monthly(WIDTH * HEIGHT * MONTHS, 0.0);
    std::vector<double> v_monthly(WIDTH * HEIGHT * MONTHS, 0.0);
    
    runShallowWaterModel(u_monthly, v_monthly, mask, bathymetry, params);
    
    // Save outputs
    ClimateFileHeader header_u = createStandardHeader(OCEANWORLD_MAGIC, 1, MONTHS);
    ClimateFileHeader header_v = createStandardHeader(OCEANWORLD_MAGIC, 1, MONTHS);
    
    header_u.channels = MONTHS;
    header_v.channels = MONTHS;
    
    saveClimateData(output_u, u_monthly, header_u);
    saveClimateData(output_v, v_monthly, header_v);
    
    std::cout << "\n✓ Complete\n";
    
    return 0;
}
