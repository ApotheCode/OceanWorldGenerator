//============================================================
// RainfallGradientAnalyzer.cu
// Analyze Earth rainfall vs distance-to-coast to calibrate parameters
// Author: Claude (2025-11-13)
// Purpose: Extract realistic inland decay rates from real data
// 
// FAST VERSION: Uses binary files (~10 seconds load time)
// Run CSV_to_Binary_Converter first to create binary files!
//
// DATA LAYOUT: Binary file has month-by-month layout:
//   [all_pixels_month_0, all_pixels_month_1, ..., all_pixels_month_11]
//   Access pattern: rainfall[month * (WIDTH*HEIGHT) + pixel_index]
//============================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>

constexpr int WIDTH = 3600;
constexpr int HEIGHT = 1800;
constexpr int MONTHS = 12;
constexpr int MAX_DISTANCE_KM = 2000;  // Max distance to analyze
constexpr int DISTANCE_BINS = 200;      // 10km bins
constexpr double KM_PER_BIN = MAX_DISTANCE_KM / (double)DISTANCE_BINS;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

//============================================================
// Custom atomicAdd for double (for older GPUs)
//============================================================
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// Native atomicAdd for double exists on compute capability 6.0+
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    
    return __longlong_as_double(old);
}
#endif

//============================================================
// Kernel: Bin Rainfall by Distance to Coast
//============================================================
__global__ void binRainfallByDistanceKernel(
    const float* __restrict__ rainfall,
    const float* __restrict__ elevation,
    const float* __restrict__ distanceToCoast,
    const unsigned char* __restrict__ isLand,
    double* __restrict__ rainfallSums,
    int* __restrict__ pixelCounts,
    int width, int height, int month)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    
    // Only analyze land pixels
    if (isLand[idx] == 0) return;
    
    // Get data - NOTE: Binary file has month-by-month layout
    float rain = rainfall[month * totalPixels + idx];
    float dist = distanceToCoast[idx];
    // Note: elevation available but not used in this kernel
    
    // Skip invalid data
    if (rain < 0 || dist < 0 || dist > MAX_DISTANCE_KM) return;
    
    // Determine bin
    int binIdx = (int)(dist / KM_PER_BIN);
    if (binIdx >= DISTANCE_BINS) binIdx = DISTANCE_BINS - 1;
    
    // Accumulate
    atomicAdd(&rainfallSums[binIdx], (double)rain);
    atomicAdd(&pixelCounts[binIdx], 1);
}

//============================================================
// Kernel: Bin by Latitude Zones
//============================================================
__global__ void binRainfallByLatitudeKernel(
    const float* __restrict__ rainfall,
    const float* __restrict__ distanceToCoast,
    const unsigned char* __restrict__ isLand,
    double* __restrict__ tropicalSums,
    double* __restrict__ subtropicalSums,
    double* __restrict__ temperateSums,
    int* __restrict__ tropicalCounts,
    int* __restrict__ subtropicalCounts,
    int* __restrict__ temperateCounts,
    int width, int height, int month)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    
    int y = idx / width;
    
    // Only analyze land pixels
    if (isLand[idx] == 0) return;
    
    // Get latitude
    double lat = 90.0 - (y + 0.5) * 180.0 / height;
    double absLat = fabs(lat);
    
    // Get data - NOTE: Binary file has month-by-month layout
    float rain = rainfall[month * totalPixels + idx];
    float dist = distanceToCoast[idx];
    
    // Skip invalid
    if (rain < 0 || dist < 0 || dist > MAX_DISTANCE_KM) return;
    
    // Determine bin
    int binIdx = (int)(dist / KM_PER_BIN);
    if (binIdx >= DISTANCE_BINS) binIdx = DISTANCE_BINS - 1;
    
    // Classify by latitude
    if (absLat < 23.5) {
        // Tropics
        atomicAdd(&tropicalSums[binIdx], (double)rain);
        atomicAdd(&tropicalCounts[binIdx], 1);
    } else if (absLat < 40.0) {
        // Subtropics
        atomicAdd(&subtropicalSums[binIdx], (double)rain);
        atomicAdd(&subtropicalCounts[binIdx], 1);
    } else {
        // Temperate
        atomicAdd(&temperateSums[binIdx], (double)rain);
        atomicAdd(&temperateCounts[binIdx], 1);
    }
}

//============================================================
// Kernel: Analyze Island vs Continental Decay
//============================================================
__global__ void analyzeIslandDecayKernel(
    const float* __restrict__ rainfall,
    const float* __restrict__ distanceToCoast,
    const unsigned char* __restrict__ isLand,
    double* __restrict__ islandSums,      // Distance < 300km from all coasts
    double* __restrict__ continentalSums, // Distance > 300km exists
    int* __restrict__ islandCounts,
    int* __restrict__ continentalCounts,
    int width, int height, int month)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    
    int x = idx % width;
    int y = idx / width;
    
    // Only analyze land pixels
    if (isLand[idx] == 0) return;
    
    // Get data - NOTE: Binary file has month-by-month layout
    float rain = rainfall[month * totalPixels + idx];
    float dist = distanceToCoast[idx];
    
    if (rain < 0 || dist < 0 || dist > MAX_DISTANCE_KM) return;
    
    int binIdx = (int)(dist / KM_PER_BIN);
    if (binIdx >= DISTANCE_BINS) binIdx = DISTANCE_BINS - 1;
    
    // Classify as island or continental based on max distance
    // Check local region (50x50 pixels) for maximum distance
    float maxLocalDist = dist;
    for (int dy = -25; dy <= 25; dy++) {
        for (int dx = -25; dx <= 25; dx++) {
            int nx = (x + dx + width) % width;
            int ny = y + dy;
            if (ny < 0 || ny >= height) continue;
            
            int nIdx = ny * width + nx;
            if (isLand[nIdx] == 1) {
                float nDist = distanceToCoast[nIdx];
                maxLocalDist = fmax(maxLocalDist, nDist);
            }
        }
    }
    
    // Island: max distance in region < 300km
    // Continental: max distance > 300km
    if (maxLocalDist < 300.0f) {
        atomicAdd(&islandSums[binIdx], (double)rain);
        atomicAdd(&islandCounts[binIdx], 1);
    } else {
        atomicAdd(&continentalSums[binIdx], (double)rain);
        atomicAdd(&continentalCounts[binIdx], 1);
    }
}

//============================================================
// Host: Fit Exponential Decay
//============================================================
struct DecayFit {
    double decayLength;    // km
    double baseRainfall;   // at coast
    double r_squared;      // fit quality
};

DecayFit fitExponentialDecay(const std::vector<double>& means, 
                              const std::vector<int>& counts)
{
    // Fit: rainfall = base * exp(-dist / length)
    // Log-linearize: ln(rain) = ln(base) - dist/length
    
    DecayFit result = {500.0, 1.0, 0.0};
    
    // Collect valid points
    std::vector<double> distances;
    std::vector<double> logRainfall;
    
    for (int i = 0; i < DISTANCE_BINS; i++) {
        if (counts[i] > 10 && means[i] > 0.01) {
            double dist = (i + 0.5) * KM_PER_BIN;
            distances.push_back(dist);
            logRainfall.push_back(log(means[i]));
        }
    }
    
    if (distances.size() < 5) return result;
    
    // Linear regression on log data
    int n = distances.size();
    double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    
    for (int i = 0; i < n; i++) {
        sumX += distances[i];
        sumY += logRainfall[i];
        sumXY += distances[i] * logRainfall[i];
        sumXX += distances[i] * distances[i];
    }
    
    // Slope = -1/length, Intercept = ln(base)
    double slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    double intercept = (sumY - slope * sumX) / n;
    
    result.decayLength = -1.0 / slope;
    result.baseRainfall = exp(intercept);
    
    // Compute R²
    double meanY = sumY / n;
    double ssTotal = 0, ssResid = 0;
    for (int i = 0; i < n; i++) {
        double predicted = intercept + slope * distances[i];
        ssTotal += pow(logRainfall[i] - meanY, 2);
        ssResid += pow(logRainfall[i] - predicted, 2);
    }
    result.r_squared = 1.0 - (ssResid / ssTotal);
    
    return result;
}

//============================================================
// Host: Load Rainfall Binary File
//============================================================
bool loadRainfallBinary(const std::string& filename, std::vector<float>& rainfall)
{
    std::cout << "Loading rainfall binary from " << filename << "..." << std::endl;
    
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t expectedSize = WIDTH * HEIGHT * MONTHS * sizeof(float);
    
    if (fileSize != expectedSize) {
        std::cerr << "ERROR: File size mismatch!" << std::endl;
        std::cerr << "  Expected: " << expectedSize << " bytes" << std::endl;
        std::cerr << "  Got: " << fileSize << " bytes" << std::endl;
        return false;
    }
    
    // Allocate and read
    rainfall.resize(WIDTH * HEIGHT * MONTHS);
    file.read(reinterpret_cast<char*>(rainfall.data()), expectedSize);
    
    if (!file) {
        std::cerr << "Failed to read rainfall data" << std::endl;
        return false;
    }
    
    file.close();
    
    // Calculate statistics
    size_t validPixels = 0;
    double sumRainfall = 0.0;
    float minRain = rainfall[0];
    float maxRain = rainfall[0];
    
    for (size_t i = 0; i < rainfall.size(); i++) {
        float val = rainfall[i];
        if (val > 0.0f) {
            validPixels++;
            sumRainfall += val;
        }
        if (val < minRain) minRain = val;
        if (val > maxRain) maxRain = val;
    }
    
    double avgRainfall = (validPixels > 0) ? (sumRainfall / validPixels) : 0.0;
    
    std::cout << "✓ Loaded rainfall data" << std::endl;
    std::cout << "  Total pixels: " << (WIDTH * HEIGHT * MONTHS) << std::endl;
    std::cout << "  Valid pixels: " << validPixels << std::endl;
    std::cout << "  Min: " << minRain << " mm/month" << std::endl;
    std::cout << "  Max: " << maxRain << " mm/month" << std::endl;
    std::cout << "  Avg: " << avgRainfall << " mm/month (over valid pixels)" << std::endl;
    
    return true;
}

//============================================================
// Host: Load Elevation Binary and Generate Land Mask
//============================================================
bool loadElevationBinary(const std::string& filename, 
                         std::vector<float>& elevation,
                         std::vector<unsigned char>& isLand)
{
    std::cout << "Loading elevation binary from " << filename << "..." << std::endl;
    
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t expectedSize = WIDTH * HEIGHT * sizeof(float);
    
    if (fileSize != expectedSize) {
        std::cerr << "ERROR: File size mismatch!" << std::endl;
        std::cerr << "  Expected: " << expectedSize << " bytes" << std::endl;
        std::cerr << "  Got: " << fileSize << " bytes" << std::endl;
        return false;
    }
    
    // Allocate and read
    elevation.resize(WIDTH * HEIGHT);
    isLand.resize(WIDTH * HEIGHT);
    
    file.read(reinterpret_cast<char*>(elevation.data()), expectedSize);
    
    if (!file) {
        std::cerr << "Failed to read elevation data" << std::endl;
        return false;
    }
    
    file.close();
    
    // Generate land mask and calculate statistics
    size_t landCount = 0;
    float minElev = elevation[0];
    float maxElev = elevation[0];
    
    for (size_t i = 0; i < WIDTH * HEIGHT; i++) {
        float elev = elevation[i];
        
        // Land = elevation > 0, Ocean = elevation <= 0
        if (elev > 0.0f) {
            isLand[i] = 1;
            landCount++;
        } else {
            isLand[i] = 0;
        }
        
        if (elev < minElev) minElev = elev;
        if (elev > maxElev) maxElev = elev;
    }
    
    std::cout << "✓ Loaded elevation data" << std::endl;
    std::cout << "  Total pixels: " << (WIDTH * HEIGHT) << std::endl;
    std::cout << "  Land pixels: " << landCount << " (" 
              << (100.0 * landCount / (WIDTH * HEIGHT)) << "%)" << std::endl;
    std::cout << "  Ocean pixels: " << (WIDTH * HEIGHT - landCount) << " ("
              << (100.0 * (WIDTH * HEIGHT - landCount) / (WIDTH * HEIGHT)) << "%)" << std::endl;
    std::cout << "  Elevation range: " << minElev << "m to " << maxElev << "m" << std::endl;
    
    return true;
}

//============================================================
// Kernel: Initialize Distance Field
//============================================================
__global__ void initializeDistanceKernel(const unsigned char* __restrict__ isLand,
                                         int* __restrict__ distance,
                                         int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    
    // Ocean = 0 distance, Land = -1 (unassigned)
    distance[idx] = (isLand[idx] == 0) ? 0 : -1;
}

//============================================================
// Kernel: Flood Fill Distance Propagation
//============================================================
__global__ void floodFillDistanceKernel(const unsigned char* __restrict__ isLand,
                                        int* __restrict__ distance,
                                        const int* __restrict__ distancePrev,
                                        int width, int height,
                                        int* __restrict__ changeCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    if (isLand[idx] == 0) return;  // Skip ocean
    if (distance[idx] != -1) return;  // Already assigned
    
    int x = idx % width;
    int y = idx / width;
    
    // Check 4 cardinal neighbors
    int minNeighborDist = -1;
    
    // Left
    int nx = (x - 1 + width) % width;
    int nIdx = y * width + nx;
    int nDist = distancePrev[nIdx];
    if (nDist >= 0) {
        minNeighborDist = (minNeighborDist < 0) ? nDist : min(minNeighborDist, nDist);
    }
    
    // Right
    nx = (x + 1) % width;
    nIdx = y * width + nx;
    nDist = distancePrev[nIdx];
    if (nDist >= 0) {
        minNeighborDist = (minNeighborDist < 0) ? nDist : min(minNeighborDist, nDist);
    }
    
    // Up
    if (y > 0) {
        nIdx = (y - 1) * width + x;
        nDist = distancePrev[nIdx];
        if (nDist >= 0) {
            minNeighborDist = (minNeighborDist < 0) ? nDist : min(minNeighborDist, nDist);
        }
    }
    
    // Down
    if (y < height - 1) {
        nIdx = (y + 1) * width + x;
        nDist = distancePrev[nIdx];
        if (nDist >= 0) {
            minNeighborDist = (minNeighborDist < 0) ? nDist : min(minNeighborDist, nDist);
        }
    }
    
    // If any neighbor has distance, adopt distance+1
    if (minNeighborDist >= 0) {
        distance[idx] = minNeighborDist + 1;
        atomicAdd(changeCount, 1);
    }
}

//============================================================
// Kernel: Convert Distance to Kilometers
//============================================================
__global__ void convertDistanceToKmKernel(const int* __restrict__ distancePixels,
                                          float* __restrict__ distanceKm,
                                          int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    
    int y = idx / width;
    int dist = distancePixels[idx];
    
    if (dist < 0) {
        distanceKm[idx] = -1.0f;
        return;
    }
    
    // Calculate km per pixel at this latitude
    float lat = 90.0f - (y + 0.5f) * 180.0f / height;
    float latRad = lat * 3.14159265f / 180.0f;
    
    // At 0.1° resolution (~3600 pixels = 360°)
    float kmPerPixel = 111.32f * cosf(latRad) * (360.0f / width);  // Longitude
    
    // For simplicity, average lat/lon distance
    // (More accurate would be to track actual path, but this is good approximation)
    distanceKm[idx] = dist * kmPerPixel;
}

//============================================================
// Host: Calculate Distance Using Flood-Fill
//============================================================
void calculateDistanceToCoast(const std::vector<unsigned char>& isLand,
                              std::vector<float>& distanceToCoast)
{
    std::cout << "Calculating distance to coast using flood-fill..." << std::endl;
    
    size_t totalPixels = WIDTH * HEIGHT;
    distanceToCoast.resize(totalPixels);
    
    // GPU arrays
    unsigned char* d_isLand;
    int* d_distance;
    int* d_distancePrev;
    int* d_changeCount;
    float* d_distanceKm;
    
    CUDA_CHECK(cudaMalloc(&d_isLand, totalPixels * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_distance, totalPixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_distancePrev, totalPixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changeCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_distanceKm, totalPixels * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_isLand, isLand.data(), 
                         totalPixels * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int numBlocks = (totalPixels + blockSize - 1) / blockSize;
    
    // Initialize: ocean = 0, land = -1
    std::cout << "  Initializing..." << std::endl;
    initializeDistanceKernel<<<numBlocks, blockSize>>>(d_isLand, d_distance, WIDTH, HEIGHT);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Flood-fill iterations
    std::cout << "  Flood-filling..." << std::endl;
    int iteration = 0;
    int maxIterations = 2000;  // Max ~2000km inland
    
    while (iteration < maxIterations) {
        // Copy current to prev
        CUDA_CHECK(cudaMemcpy(d_distancePrev, d_distance, 
                             totalPixels * sizeof(int), cudaMemcpyDeviceToDevice));
        
        // Reset change counter
        int h_changeCount = 0;
        CUDA_CHECK(cudaMemcpy(d_changeCount, &h_changeCount, sizeof(int), cudaMemcpyHostToDevice));
        
        // Run flood-fill step
        floodFillDistanceKernel<<<numBlocks, blockSize>>>(
            d_isLand, d_distance, d_distancePrev, WIDTH, HEIGHT, d_changeCount);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Check if any changes
        CUDA_CHECK(cudaMemcpy(&h_changeCount, d_changeCount, sizeof(int), cudaMemcpyDeviceToHost));
        
        iteration++;
        
        if (iteration % 100 == 0) {
            std::cout << "    Iteration " << iteration << ", changed: " << h_changeCount << std::endl;
        }
        
        // Stop if no more changes
        if (h_changeCount == 0) {
            std::cout << "  ✓ Converged after " << iteration << " iterations" << std::endl;
            break;
        }
    }
    
    // Convert to kilometers
    std::cout << "  Converting to kilometers..." << std::endl;
    convertDistanceToKmKernel<<<numBlocks, blockSize>>>(d_distance, d_distanceKm, WIDTH, HEIGHT);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Download results
    CUDA_CHECK(cudaMemcpy(distanceToCoast.data(), d_distanceKm, 
                         totalPixels * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_isLand));
    CUDA_CHECK(cudaFree(d_distance));
    CUDA_CHECK(cudaFree(d_distancePrev));
    CUDA_CHECK(cudaFree(d_changeCount));
    CUDA_CHECK(cudaFree(d_distanceKm));
    
    std::cout << "✓ Distance calculation complete" << std::endl;
}

//============================================================
// Main
//============================================================
int main(int argc, char* argv[])
{
    std::cout << "=== Rainfall Gradient Analyzer ===" << std::endl;
    std::cout << "Analyzing Earth data to calibrate decay parameters..." << std::endl << std::endl;
    
    // Parse arguments
    std::string rainfallBinary = "Earth_Rainfall.bin";
    std::string elevationBinary = "Earth_Elevation.bin";
    std::string outputFile = "rainfall_decay_params.txt";
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-r") == 0 && i+1 < argc)
            rainfallBinary = argv[++i];
        else if (strcmp(argv[i], "-e") == 0 && i+1 < argc)
            elevationBinary = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc)
            outputFile = argv[++i];
        else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  -r <file>     Rainfall binary file (default: Earth_Rainfall.bin)\n";
            std::cout << "  -e <file>     Elevation binary file (default: Earth_Elevation.bin)\n";
            std::cout << "  -o <file>     Output parameters file (default: rainfall_decay_params.txt)\n";
            std::cout << "\nBinary files must be created using CSV_to_Binary_Converter first!\n";
            return 0;
        }
    }
    
    // Load rainfall data
    std::cout << "Loading rainfall data from " << rainfallBinary << "..." << std::endl;
    std::vector<float> rainfall;
    if (!loadRainfallBinary(rainfallBinary, rainfall)) {
        std::cerr << "\n✗ Failed to load rainfall data" << std::endl;
        std::cerr << "Make sure you've run CSV_to_Binary_Converter first!" << std::endl;
        return EXIT_FAILURE;
    }
    
    // Load elevation and generate land mask
    std::cout << "\nLoading elevation from " << elevationBinary << "..." << std::endl;
    std::vector<float> elevation;
    std::vector<unsigned char> isLand;
    if (!loadElevationBinary(elevationBinary, elevation, isLand)) {
        std::cerr << "\n✗ Failed to load elevation data" << std::endl;
        std::cerr << "Make sure you've run CSV_to_Binary_Converter first!" << std::endl;
        return EXIT_FAILURE;
    }
    
    // Calculate distance to coast
    std::vector<float> distanceToCoast;
    calculateDistanceToCoast(isLand, distanceToCoast);
    
    // Upload to GPU
    std::cout << "\nUploading data to GPU..." << std::endl;
    size_t totalPixels = WIDTH * HEIGHT;
    
    float* d_rainfall;
    float* d_elevation;
    float* d_distance;
    unsigned char* d_isLand;
    
    CUDA_CHECK(cudaMalloc(&d_rainfall, totalPixels * MONTHS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_elevation, totalPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_distance, totalPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_isLand, totalPixels * sizeof(unsigned char)));
    
    CUDA_CHECK(cudaMemcpy(d_rainfall, rainfall.data(), totalPixels * MONTHS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_elevation, elevation.data(), totalPixels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_distance, distanceToCoast.data(), totalPixels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_isLand, isLand.data(), totalPixels * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    // Allocate result arrays
    double* d_rainfallSums;
    int* d_pixelCounts;
    
    double* d_tropicalSums;
    double* d_subtropicalSums;
    double* d_temperateSums;
    int* d_tropicalCounts;
    int* d_subtropicalCounts;
    int* d_temperateCounts;
    
    double* d_islandSums;
    double* d_continentalSums;
    int* d_islandCounts;
    int* d_continentalCounts;
    
    size_t binSize = DISTANCE_BINS * sizeof(double);
    size_t countSize = DISTANCE_BINS * sizeof(int);
    
    CUDA_CHECK(cudaMalloc(&d_rainfallSums, binSize));
    CUDA_CHECK(cudaMalloc(&d_pixelCounts, countSize));
    CUDA_CHECK(cudaMalloc(&d_tropicalSums, binSize));
    CUDA_CHECK(cudaMalloc(&d_subtropicalSums, binSize));
    CUDA_CHECK(cudaMalloc(&d_temperateSums, binSize));
    CUDA_CHECK(cudaMalloc(&d_tropicalCounts, countSize));
    CUDA_CHECK(cudaMalloc(&d_subtropicalCounts, countSize));
    CUDA_CHECK(cudaMalloc(&d_temperateCounts, countSize));
    CUDA_CHECK(cudaMalloc(&d_islandSums, binSize));
    CUDA_CHECK(cudaMalloc(&d_continentalSums, binSize));
    CUDA_CHECK(cudaMalloc(&d_islandCounts, countSize));
    CUDA_CHECK(cudaMalloc(&d_continentalCounts, countSize));
    
    // Run analysis for annual average
    std::cout << "\nAnalyzing rainfall gradients..." << std::endl;
    
    int blockSize = 256;
    int numBlocks = (totalPixels + blockSize - 1) / blockSize;
    
    // Initialize
    CUDA_CHECK(cudaMemset(d_rainfallSums, 0, binSize));
    CUDA_CHECK(cudaMemset(d_pixelCounts, 0, countSize));
    CUDA_CHECK(cudaMemset(d_tropicalSums, 0, binSize));
    CUDA_CHECK(cudaMemset(d_subtropicalSums, 0, binSize));
    CUDA_CHECK(cudaMemset(d_temperateSums, 0, binSize));
    CUDA_CHECK(cudaMemset(d_tropicalCounts, 0, countSize));
    CUDA_CHECK(cudaMemset(d_subtropicalCounts, 0, countSize));
    CUDA_CHECK(cudaMemset(d_temperateCounts, 0, countSize));
    CUDA_CHECK(cudaMemset(d_islandSums, 0, binSize));
    CUDA_CHECK(cudaMemset(d_continentalSums, 0, binSize));
    CUDA_CHECK(cudaMemset(d_islandCounts, 0, countSize));
    CUDA_CHECK(cudaMemset(d_continentalCounts, 0, countSize));
    
    // Run all analyses (averaging across all months)
    for (int month = 0; month < MONTHS; month++) {
        binRainfallByDistanceKernel<<<numBlocks, blockSize>>>(
            d_rainfall, d_elevation, d_distance, d_isLand,
            d_rainfallSums, d_pixelCounts, WIDTH, HEIGHT, month);
        
        binRainfallByLatitudeKernel<<<numBlocks, blockSize>>>(
            d_rainfall, d_distance, d_isLand,
            d_tropicalSums, d_subtropicalSums, d_temperateSums,
            d_tropicalCounts, d_subtropicalCounts, d_temperateCounts,
            WIDTH, HEIGHT, month);
        
        analyzeIslandDecayKernel<<<numBlocks, blockSize>>>(
            d_rainfall, d_distance, d_isLand,
            d_islandSums, d_continentalSums,
            d_islandCounts, d_continentalCounts,
            WIDTH, HEIGHT, month);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Download results
    std::cout << "Downloading results..." << std::endl;
    
    std::vector<double> rainfallSums(DISTANCE_BINS);
    std::vector<int> pixelCounts(DISTANCE_BINS);
    std::vector<double> tropicalSums(DISTANCE_BINS);
    std::vector<double> subtropicalSums(DISTANCE_BINS);
    std::vector<double> temperateSums(DISTANCE_BINS);
    std::vector<int> tropicalCounts(DISTANCE_BINS);
    std::vector<int> subtropicalCounts(DISTANCE_BINS);
    std::vector<int> temperateCounts(DISTANCE_BINS);
    std::vector<double> islandSums(DISTANCE_BINS);
    std::vector<double> continentalSums(DISTANCE_BINS);
    std::vector<int> islandCounts(DISTANCE_BINS);
    std::vector<int> continentalCounts(DISTANCE_BINS);
    
    CUDA_CHECK(cudaMemcpy(rainfallSums.data(), d_rainfallSums, binSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pixelCounts.data(), d_pixelCounts, countSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(tropicalSums.data(), d_tropicalSums, binSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(subtropicalSums.data(), d_subtropicalSums, binSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(temperateSums.data(), d_temperateSums, binSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(tropicalCounts.data(), d_tropicalCounts, countSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(subtropicalCounts.data(), d_subtropicalCounts, countSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(temperateCounts.data(), d_temperateCounts, countSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(islandSums.data(), d_islandSums, binSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(continentalSums.data(), d_continentalSums, binSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(islandCounts.data(), d_islandCounts, countSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(continentalCounts.data(), d_continentalCounts, countSize, cudaMemcpyDeviceToHost));
    
    // Calculate means and fit decay curves
    std::cout << "\nCalculating decay parameters..." << std::endl;
    
    std::vector<double> globalMeans(DISTANCE_BINS);
    std::vector<double> tropicalMeans(DISTANCE_BINS);
    std::vector<double> subtropicalMeans(DISTANCE_BINS);
    std::vector<double> temperateMeans(DISTANCE_BINS);
    std::vector<double> islandMeans(DISTANCE_BINS);
    std::vector<double> continentalMeans(DISTANCE_BINS);
    
    for (int i = 0; i < DISTANCE_BINS; i++) {
        globalMeans[i] = (pixelCounts[i] > 0) ? rainfallSums[i] / (pixelCounts[i] * MONTHS) : 0.0;
        tropicalMeans[i] = (tropicalCounts[i] > 0) ? tropicalSums[i] / (tropicalCounts[i] * MONTHS) : 0.0;
        subtropicalMeans[i] = (subtropicalCounts[i] > 0) ? subtropicalSums[i] / (subtropicalCounts[i] * MONTHS) : 0.0;
        temperateMeans[i] = (temperateCounts[i] > 0) ? temperateSums[i] / (temperateCounts[i] * MONTHS) : 0.0;
        islandMeans[i] = (islandCounts[i] > 0) ? islandSums[i] / (islandCounts[i] * MONTHS) : 0.0;
        continentalMeans[i] = (continentalCounts[i] > 0) ? continentalSums[i] / (continentalCounts[i] * MONTHS) : 0.0;
    }
    
    DecayFit globalFit = fitExponentialDecay(globalMeans, pixelCounts);
    DecayFit tropicalFit = fitExponentialDecay(tropicalMeans, tropicalCounts);
    DecayFit subtropicalFit = fitExponentialDecay(subtropicalMeans, subtropicalCounts);
    DecayFit temperateFit = fitExponentialDecay(temperateMeans, temperateCounts);
    DecayFit islandFit = fitExponentialDecay(islandMeans, islandCounts);
    DecayFit continentalFit = fitExponentialDecay(continentalMeans, continentalCounts);
    
    // Print results
    std::cout << "\n========== RESULTS ==========" << std::endl;
    std::cout << "\nGlobal Decay:" << std::endl;
    std::cout << "  Decay length: " << globalFit.decayLength << " km" << std::endl;
    std::cout << "  Base rainfall: " << globalFit.baseRainfall << std::endl;
    std::cout << "  R²: " << globalFit.r_squared << std::endl;
    
    std::cout << "\nTropical (0-23.5°):" << std::endl;
    std::cout << "  Decay length: " << tropicalFit.decayLength << " km" << std::endl;
    std::cout << "  R²: " << tropicalFit.r_squared << std::endl;
    
    std::cout << "\nSubtropical (23.5-40°):" << std::endl;
    std::cout << "  Decay length: " << subtropicalFit.decayLength << " km" << std::endl;
    std::cout << "  R²: " << subtropicalFit.r_squared << std::endl;
    
    std::cout << "\nTemperate (40-90°):" << std::endl;
    std::cout << "  Decay length: " << temperateFit.decayLength << " km" << std::endl;
    std::cout << "  R²: " << temperateFit.r_squared << std::endl;
    
    std::cout << "\nIsland (<300km max):" << std::endl;
    std::cout << "  Decay length: " << islandFit.decayLength << " km" << std::endl;
    std::cout << "  R²: " << islandFit.r_squared << std::endl;
    
    std::cout << "\nContinental (>300km max):" << std::endl;
    std::cout << "  Decay length: " << continentalFit.decayLength << " km" << std::endl;
    std::cout << "  R²: " << continentalFit.r_squared << std::endl;
    
    // Save to file
    std::ofstream out(outputFile);
    out << "# Rainfall Decay Parameters (from Earth data)\n";
    out << "global_decay_km " << globalFit.decayLength << "\n";
    out << "tropical_decay_km " << tropicalFit.decayLength << "\n";
    out << "subtropical_decay_km " << subtropicalFit.decayLength << "\n";
    out << "temperate_decay_km " << temperateFit.decayLength << "\n";
    out << "island_decay_km " << islandFit.decayLength << "\n";
    out << "continental_decay_km " << continentalFit.decayLength << "\n";
    out.close();
    
    std::cout << "\n✓ Results saved to " << outputFile << std::endl;
    std::cout << "\nUse these values in your RainfallCalculator!" << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_rainfall));
    CUDA_CHECK(cudaFree(d_elevation));
    CUDA_CHECK(cudaFree(d_distance));
    CUDA_CHECK(cudaFree(d_isLand));
    CUDA_CHECK(cudaFree(d_rainfallSums));
    CUDA_CHECK(cudaFree(d_pixelCounts));
    CUDA_CHECK(cudaFree(d_tropicalSums));
    CUDA_CHECK(cudaFree(d_subtropicalSums));
    CUDA_CHECK(cudaFree(d_temperateSums));
    CUDA_CHECK(cudaFree(d_tropicalCounts));
    CUDA_CHECK(cudaFree(d_subtropicalCounts));
    CUDA_CHECK(cudaFree(d_temperateCounts));
    CUDA_CHECK(cudaFree(d_islandSums));
    CUDA_CHECK(cudaFree(d_continentalSums));
    CUDA_CHECK(cudaFree(d_islandCounts));
    CUDA_CHECK(cudaFree(d_continentalCounts));
    
    return 0;
}
