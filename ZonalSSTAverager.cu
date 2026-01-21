//============================================================
// ZonalSSTAverager.cu
// Remove basin-scale SST patterns by averaging within latitude bands
// Author: Claude (2025-11-12)
// Purpose: Create pure ocean world SST without continental memory
//============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "ClimateFileFormat.h"

constexpr int WIDTH = STANDARD_WIDTH;
constexpr int HEIGHT = STANDARD_HEIGHT;
constexpr int MONTHS = STANDARD_MONTHS;
constexpr int VARIABLES = STANDARD_VARIABLES;
constexpr int VAR_SST = 1;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

//============================================================
// Device: Get/Set Climate Variable
//============================================================
__device__ __forceinline__ double getClimateValue(const double* data,
                                                   int x, int y, int varIdx, 
                                                   int month, int width, int height)
{
    size_t cellIdx = y * width + x;
    size_t cellBase = cellIdx * VARIABLES * MONTHS;
    size_t dataIdx = cellBase + varIdx * MONTHS + month;
    return data[dataIdx];
}

__device__ __forceinline__ void setClimateValue(double* data,
                                                int x, int y, int varIdx,
                                                int month, int width, int height,
                                                double value)
{
    size_t cellIdx = y * width + x;
    size_t cellBase = cellIdx * VARIABLES * MONTHS;
    size_t dataIdx = cellBase + varIdx * MONTHS + month;
    data[dataIdx] = value;
}

//============================================================
// Kernel 1: Compute Zonal Mean SST for Each Latitude Band
//============================================================
__global__ void computeZonalMeanKernel(const double* __restrict__ climateData,
                                       double* __restrict__ zonalMeans,
                                       int* __restrict__ zonalCounts,
                                       int month, int width, int height)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (y >= height) return;
    
    // For this latitude (y), compute mean SST across all longitudes
    // OceanWorld = all ocean, no land mask needed
    double sum = 0.0;
    int count = 0;
    
    for (int x = 0; x < width; x++) {
        double sst = getClimateValue(climateData, x, y, VAR_SST, month, width, height);
        if (isfinite(sst)) {
            sum += sst;
            count++;
        }
    }
    
    // Store results
    zonalMeans[y * MONTHS + month] = (count > 0) ? (sum / count) : 0.0;
    zonalCounts[y * MONTHS + month] = count;
}

//============================================================
// Kernel 2: Apply Zonal Mean with Smoothing
//============================================================
__global__ void applyZonalMeanKernel(double* __restrict__ climateData,
                                     const double* __restrict__ zonalMeans,
                                     double smoothingFactor,
                                     int month, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    
    int x = idx % width;
    int y = idx / width;
    
    // Get current SST
    double currentSST = getClimateValue(climateData, x, y, VAR_SST, month, width, height);
    
    if (!isfinite(currentSST)) return;
    
    // Get zonal mean for this latitude
    double zonalMean = zonalMeans[y * MONTHS + month];
    
    // Blend between current SST and zonal mean
    // smoothingFactor = 1.0 → pure zonal (no longitude variation)
    // smoothingFactor = 0.5 → 50% reduction in basin patterns
    // smoothingFactor = 0.0 → no change
    double newSST = currentSST * (1.0 - smoothingFactor) + zonalMean * smoothingFactor;
    
    // Write back
    setClimateValue(climateData, x, y, VAR_SST, month, width, height, newSST);
}

//============================================================
// Kernel 3: Latitude-Weighted Smoothing (Optional)
//============================================================
__global__ void latitudeWeightedSmoothingKernel(double* __restrict__ climateData,
                                                const double* __restrict__ zonalMeans,
                                                int month, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    
    int x = idx % width;
    int y = idx / width;
    
    // Get latitude in degrees
    double lat = 90.0 - (y + 0.5) * 180.0 / height;
    double absLat = fabs(lat);
    
    // Smoothing factor based on latitude
    // Equator (0°): Keep some basin variation (factor = 0.7)
    // Subtropics (30°): Maximum smoothing (factor = 0.95)
    // Poles (90°): Keep some variation (factor = 0.8)
    double factor;
    if (absLat < 15.0) {
        // Tropics: moderate smoothing (ITCZ has real longitude variation)
        factor = 0.70;
    } else if (absLat < 45.0) {
        // Subtropics: strong smoothing (gyre centers should be uniform)
        factor = 0.90 + 0.05 * (absLat - 15.0) / 30.0; // 0.90 to 0.95
    } else {
        // High latitudes: moderate smoothing
        factor = 0.85;
    }
    
    // Get current SST and zonal mean
    double currentSST = getClimateValue(climateData, x, y, VAR_SST, month, width, height);
    
    if (!isfinite(currentSST)) return;
    
    double zonalMean = zonalMeans[y * MONTHS + month];
    
    // Apply weighted blend
    double newSST = currentSST * (1.0 - factor) + zonalMean * factor;
    
    // Write back
    setClimateValue(climateData, x, y, VAR_SST, month, width, height, newSST);
}

//============================================================
// Host: Load/Save Climate Data
//============================================================
bool loadClimateData(const std::string& filename, std::vector<double>& data)
{
    ClimateFileHeader header;
    return loadClimateData(filename, data, header);
}

bool saveClimateData(const std::string& filename, const std::vector<double>& data)
{
    ClimateFileHeader header = createStandardHeader(CLIMATE_MAGIC);
    return saveClimateData(filename, data, header);
}


//============================================================
// Main
//============================================================
int main(int argc, char* argv[])
{
    std::cout << "=== Zonal SST Averager for OceanWorld ===" << std::endl;
    
    // Parse command line
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input.bin> <output.bin> <smoothing_mode> [factor]" << std::endl;
        std::cerr << "  input.bin        - Input climate file (e.g., OceanWorld_Climate.bin)" << std::endl;
        std::cerr << "  output.bin       - Output climate file" << std::endl;
        std::cerr << "  smoothing_mode:" << std::endl;
        std::cerr << "    uniform <factor>  - Uniform smoothing (0.0-1.0)" << std::endl;
        std::cerr << "    latitude          - Latitude-weighted smoothing" << std::endl;
        return 1;
    }
    
    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    std::string mode = argv[3];
    double smoothingFactor = 0.85;  // default
    
    if (mode == "uniform" && argc >= 5) {
        smoothingFactor = atof(argv[4]);
    }
    
    std::cout << "Input:  " << inputFile << std::endl;
    std::cout << "Output: " << outputFile << std::endl;
    std::cout << "Mode:   " << mode;
    if (mode == "uniform") {
        std::cout << " (factor=" << smoothingFactor << ")";
    }
    std::cout << std::endl;
    
    // Load data
    std::cout << "Loading climate data..." << std::endl;
    std::vector<double> climateData;
    if (!loadClimateData(inputFile, climateData)) {
        std::cerr << "Failed to load climate data!" << std::endl;
        return 1;
    }
    
    std::cout << "  Loaded " << (climateData.size() * sizeof(double) / (1024.0*1024.0)) 
              << " MB" << std::endl;
    
    // Allocate GPU memory (NO land mask needed for OceanWorld)
    double* d_climateData;
    double* d_zonalMeans;
    int* d_zonalCounts;
    
    size_t climateSize = climateData.size() * sizeof(double);
    size_t zonalSize = HEIGHT * MONTHS * sizeof(double);
    size_t countSize = HEIGHT * MONTHS * sizeof(int);
    
    CUDA_CHECK(cudaMalloc(&d_climateData, climateSize));
    CUDA_CHECK(cudaMalloc(&d_zonalMeans, zonalSize));
    CUDA_CHECK(cudaMalloc(&d_zonalCounts, countSize));
    
    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(d_climateData, climateData.data(), climateSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_zonalMeans, 0, zonalSize));
    CUDA_CHECK(cudaMemset(d_zonalCounts, 0, countSize));
    
    // Process each month
    int blockSize = 256;
    int numBlocksLat = (HEIGHT + blockSize - 1) / blockSize;
    int numBlocksPixels = (WIDTH * HEIGHT + blockSize - 1) / blockSize;
    
    std::cout << "Processing months..." << std::endl;
    
    for (int month = 0; month < MONTHS; month++) {
        std::cout << "  Month " << (month + 1) << "/12..." << std::endl;
        
        // Step 1: Compute zonal means
        computeZonalMeanKernel<<<numBlocksLat, blockSize>>>(
            d_climateData, d_zonalMeans, d_zonalCounts, month, WIDTH, HEIGHT
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 2: Apply smoothing
        if (mode == "latitude") {
            latitudeWeightedSmoothingKernel<<<numBlocksPixels, blockSize>>>(
                d_climateData, d_zonalMeans, month, WIDTH, HEIGHT
            );
        } else {
            applyZonalMeanKernel<<<numBlocksPixels, blockSize>>>(
                d_climateData, d_zonalMeans, smoothingFactor, month, WIDTH, HEIGHT
            );
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Copy results back
    std::cout << "Copying results..." << std::endl;
    CUDA_CHECK(cudaMemcpy(climateData.data(), d_climateData, climateSize, cudaMemcpyDeviceToHost));
    
    // Save
    std::cout << "Saving zonal-averaged data..." << std::endl;
    if (!saveClimateData(outputFile, climateData)) {
        std::cerr << "Failed to save!" << std::endl;
        return 1;
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_climateData));
    CUDA_CHECK(cudaFree(d_zonalMeans));
    CUDA_CHECK(cudaFree(d_zonalCounts));
    
    std::cout << "✓ Complete! Basin patterns removed from SST." << std::endl;
    std::cout << "  Next: Apply PNPL for realistic ocean circulation." << std::endl;
    
    return 0;
}
