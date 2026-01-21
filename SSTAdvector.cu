//============================================================
// SSTAdvector.cu
// Transports SST using ocean gyre velocity fields
// Physics: ∂SST/∂t = -u·∇SST + κ∇²SST
//
// Replaces PNPL heuristics with physics-based heat transport
// Author: Mark Devereux (2025)
//============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include "ClimateFileFormat.h"

constexpr int WIDTH = STANDARD_WIDTH;
constexpr int HEIGHT = STANDARD_HEIGHT;
constexpr int MONTHS = 12;
constexpr int VARIABLES = 14;
constexpr double M_PI = 3.14159265358979323846;

constexpr int VAR_SST = 1;  // SST is variable index 1

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

//============================================================
// Physics Parameters
//============================================================
struct AdvectionParams {
    double kappa;           // Diffusion coefficient (m²/s) - 100-500
    double dt;              // Timestep (s)
    int steps_per_month;    // Integration steps per month
    double R;               // Planet radius (m)
    double dlon;            // Grid spacing (rad)
    double dlat;            // Grid spacing (rad)
};

//============================================================
// Device: Spatial Derivatives
//============================================================
__device__ double ddx_sst(const double* __restrict__ sst, int x, int y,
                          int width, int height, double dx)
{
    int xp = (x + 1) % width;
    int xm = (x - 1 + width) % width;
    return (sst[y * width + xp] - sst[y * width + xm]) / (2.0 * dx);
}

__device__ double ddy_sst(const double* __restrict__ sst, int x, int y,
                          int width, int height, double dy)
{
    if (y == 0 || y == height - 1) return 0.0;
    return (sst[(y + 1) * width + x] - sst[(y - 1) * width + x]) / (2.0 * dy);
}

__device__ double laplacian_sst(const double* __restrict__ sst, int x, int y,
                                int width, int height, double dx, double dy,
                                const unsigned char* __restrict__ mask)
{
    int idx = y * width + x;
    if (mask[idx] == 0) return 0.0;  // Land
    
    int xp = (x + 1) % width;
    int xm = (x - 1 + width) % width;
    
    double d2dx2 = (sst[y * width + xp] - 2.0 * sst[idx] + sst[y * width + xm]) / (dx * dx);
    
    double d2dy2 = 0.0;
    if (y > 0 && y < height - 1) {
        d2dy2 = (sst[(y + 1) * width + x] - 2.0 * sst[idx] + sst[(y - 1) * width + x]) / (dy * dy);
    }
    
    return d2dx2 + d2dy2;
}

//============================================================
// Kernel: SST Advection-Diffusion Step
//============================================================
__global__ void advect_sst_kernel(
    const double* __restrict__ sst_in,
    double* __restrict__ sst_out,
    const double* __restrict__ u,
    const double* __restrict__ v,
    const unsigned char* __restrict__ mask,
    AdvectionParams params,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Land pixels: no change
    if (mask[idx] == 0) {
        sst_out[idx] = sst_in[idx];
        return;
    }
    
    double lat = (y - height / 2.0) * params.dlat;
    double dx = params.R * cos(lat) * params.dlon;
    double dy = params.R * params.dlat;
    
    // Current SST and velocity
    double sst_curr = sst_in[idx];
    double u_curr = u[idx];
    double v_curr = v[idx];
    
    // Advection: -u·∇SST
    double dSSTdx = ddx_sst(sst_in, x, y, width, height, dx);
    double dSSTdy = ddy_sst(sst_in, x, y, width, height, dy);
    double advection = -(u_curr * dSSTdx + v_curr * dSSTdy);
    
    // Diffusion: κ∇²SST
    double diffusion = params.kappa * laplacian_sst(sst_in, x, y, width, height, dx, dy, mask);
    
    // Forward Euler: SST_new = SST_old + dt * (advection + diffusion)
    double sst_new = sst_curr + params.dt * (advection + diffusion);
    
    // Clamp to valid range [0, 1]
    sst_out[idx] = fmax(0.0, fmin(1.0, sst_new));
}

//============================================================
// Apply Boundary Conditions
//============================================================
__global__ void apply_sst_bc_kernel(
    double* __restrict__ sst,
    const unsigned char* __restrict__ mask,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Ocean cells adjacent to land: prevent unrealistic gradients
    if (mask[idx] == 1) {
        int xp = (x + 1) % width;
        int xm = (x - 1 + width) % width;
        
        bool land_adjacent = false;
        if (mask[y * width + xp] == 0) land_adjacent = true;
        if (mask[y * width + xm] == 0) land_adjacent = true;
        if (y > 0 && mask[(y - 1) * width + x] == 0) land_adjacent = true;
        if (y < height - 1 && mask[(y + 1) * width + x] == 0) land_adjacent = true;
        
        // Slight smoothing at coastlines
        if (land_adjacent) {
            int count = 0;
            double sum = 0.0;
            
            if (mask[y * width + xp] == 1) { sum += sst[y * width + xp]; count++; }
            if (mask[y * width + xm] == 1) { sum += sst[y * width + xm]; count++; }
            if (y > 0 && mask[(y - 1) * width + x] == 1) { sum += sst[(y - 1) * width + x]; count++; }
            if (y < height - 1 && mask[(y + 1) * width + x] == 1) { sum += sst[(y + 1) * width + x]; count++; }
            
            if (count > 0) {
                // Blend current value with neighbors (90% current, 10% neighbors)
                sst[idx] = 0.9 * sst[idx] + 0.1 * (sum / count);
            }
        }
    }
}

//============================================================
// Accumulate Monthly Averages
//============================================================
__global__ void accumulate_kernel(
    const double* __restrict__ sst_current,
    double* __restrict__ sst_accumulated,
    const unsigned char* __restrict__ mask,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    if (mask[idx] == 1) {
        sst_accumulated[idx] += sst_current[idx];
    }
}

//============================================================
// Extract Ocean Mask from Climate Data
//============================================================
void extractOceanMask(const std::vector<double>& climate_data,
                      std::vector<unsigned char>& mask)
{
    mask.resize(WIDTH * HEIGHT);
    
    // SST < 0.01 or elevation > 0 → land
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        int sst_idx = i * VARIABLES * MONTHS + VAR_SST * MONTHS;  // First month SST
        double sst = climate_data[sst_idx];
        
        // Ocean if SST is reasonable
        mask[i] = (sst > 0.01 && sst < 0.99) ? 1 : 0;
    }
    
    size_t ocean_count = 0;
    for (auto m : mask) if (m == 1) ocean_count++;
    
    std::cout << "   Ocean: " << (ocean_count * 100.0 / mask.size()) << "%\n";
}

//============================================================
// Process One Month
//============================================================
void processMonth(int month,
                  const std::vector<double>& base_sst,
                  const std::vector<double>& u_velocity,
                  const std::vector<double>& v_velocity,
                  const std::vector<unsigned char>& mask,
                  std::vector<double>& sst_monthly,
                  AdvectionParams params)
{
    size_t N = WIDTH * HEIGHT;
    size_t bytes = N * sizeof(double);
    size_t bytes_mask = N * sizeof(unsigned char);
    
    // Device memory
    double *d_sst, *d_sst_tmp, *d_sst_accum;
    double *d_u, *d_v;
    unsigned char *d_mask;
    
    CUDA_CHECK(cudaMalloc(&d_sst, bytes));
    CUDA_CHECK(cudaMalloc(&d_sst_tmp, bytes));
    CUDA_CHECK(cudaMalloc(&d_sst_accum, bytes));
    CUDA_CHECK(cudaMalloc(&d_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMalloc(&d_mask, bytes_mask));
    
    // Initialize with base SST for this month
    std::vector<double> sst_month(N);
    for (int i = 0; i < N; i++) {
        sst_month[i] = base_sst[i * MONTHS + month];
    }
    
    CUDA_CHECK(cudaMemcpy(d_sst, sst_month.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_sst_accum, 0, bytes));
    
    // Upload velocity for this month
    std::vector<double> u_month(N), v_month(N);
    for (int i = 0; i < N; i++) {
        u_month[i] = u_velocity[i * MONTHS + month];
        v_month[i] = v_velocity[i * MONTHS + month];
    }
    
    CUDA_CHECK(cudaMemcpy(d_u, u_month.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, v_month.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, mask.data(), bytes_mask, cudaMemcpyHostToDevice));
    
    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
    
    // Time integration for this month
    for (int step = 0; step < params.steps_per_month; step++) {
        
        // Advection-diffusion step
        advect_sst_kernel<<<grid, block>>>(d_sst, d_sst_tmp, d_u, d_v, d_mask, params, WIDTH, HEIGHT);
        
        // Swap buffers
        std::swap(d_sst, d_sst_tmp);
        
        // Apply boundary conditions
        apply_sst_bc_kernel<<<grid, block>>>(d_sst, d_mask, WIDTH, HEIGHT);
        
        // Accumulate for monthly average
        accumulate_kernel<<<grid, block>>>(d_sst, d_sst_accum, d_mask, WIDTH, HEIGHT);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Compute monthly average
    std::vector<double> sst_accum(N);
    CUDA_CHECK(cudaMemcpy(sst_accum.data(), d_sst_accum, bytes, cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < N; i++) {
        sst_monthly[i * MONTHS + month] = sst_accum[i] / params.steps_per_month;
    }
    
    // Cleanup
    cudaFree(d_sst);
    cudaFree(d_sst_tmp);
    cudaFree(d_sst_accum);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_mask);
}

//============================================================
// Main
//============================================================
int main(int argc, char** argv)
{
    std::cout << "🌡️  SST Advector - Physics-Based Heat Transport\n";
    std::cout << "================================================\n\n";
    
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <climate_in> <gyre_u> <gyre_v> <climate_out>\n";
        return 1;
    }
    
    std::string climate_in = argv[1];
    std::string gyre_u = argv[2];
    std::string gyre_v = argv[3];
    std::string climate_out = argv[4];
    
    // Load climate data
    ClimateFileHeader climate_header;
    std::vector<double> climate_data;
    if (!loadClimateData(climate_in, climate_data, climate_header)) {
        return 1;
    }
    
    // Load velocity fields
    ClimateFileHeader u_header, v_header;
    std::vector<double> u_data, v_data;
    
    if (!loadClimateData(gyre_u, u_data, u_header)) {
        return 1;
    }
    
    if (!loadClimateData(gyre_v, v_data, v_header)) {
        return 1;
    }
    
    // Extract ocean mask
    std::vector<unsigned char> mask;
    extractOceanMask(climate_data, mask);
    
    // Extract base SST from climate data
    std::cout << "📊 Extracting base SST...\n";
    std::vector<double> base_sst(WIDTH * HEIGHT * MONTHS);
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        for (int m = 0; m < MONTHS; m++) {
            int idx = i * VARIABLES * MONTHS + VAR_SST * MONTHS + m;
            base_sst[i * MONTHS + m] = climate_data[idx];
        }
    }
    
    // Physics parameters
    AdvectionParams params;
    params.kappa = 50;                    // Diffusion (m²/s)
    params.R = 6.371e6;                      // Earth radius
    params.dlon = 2.0 * M_PI / WIDTH;
    params.dlat = M_PI / HEIGHT;
    params.dt = 3600.0;                      // 1 hour timesteps
    params.steps_per_month = 180;        // 30 days × 24 hours
    
    std::cout << "⚙️  Parameters:\n";
    std::cout << "   Diffusion: " << params.kappa << " m²/s\n";
    std::cout << "   Timestep: " << params.dt << " s\n";
    std::cout << "   Steps/month: " << params.steps_per_month << "\n\n";
    
    // Process each month
    std::vector<double> sst_advected(WIDTH * HEIGHT * MONTHS);
    
    std::cout << "🌊 Transporting SST with ocean currents...\n";
    for (int m = 0; m < MONTHS; m++) {
        std::cout << "   Month " << (m + 1) << "/12...\r" << std::flush;
        processMonth(m, base_sst, u_data, v_data, mask, sst_advected, params);
    }
    std::cout << "\n✅ Advection complete\n\n";
    
    // Update climate data with advected SST
    std::cout << "📝 Updating climate data...\n";
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        for (int m = 0; m < MONTHS; m++) {
            int idx = i * VARIABLES * MONTHS + VAR_SST * MONTHS + m;
            
            // Only update ocean pixels
            if (mask[i] == 1) {
                climate_data[idx] = sst_advected[i * MONTHS + m];
            }
        }
    }
    
    // Save output
    if (!saveClimateData(climate_out, climate_data, climate_header)) {
        return 1;
    }
    
    std::cout << "\n✅ SST advection complete\n";
    std::cout << "   Output: " << climate_out << "\n";
    std::cout << "   SST now includes gyre-driven heat transport\n";
    
    return 0;
}
