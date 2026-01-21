//============================================================
// BasinDetector.cu
// Automatic Ocean Basin Detection from Bathymetry
// 
// Analyzes bathymetry depth map and segments into discrete ocean basins
// using watershed algorithm with local depth minima as seeds.
// 
// Input: Bathymetry file (.bath)
// Output: Basin map file (.basin)
// 
// Author: Mark Devereux with Claude (CTO)
//============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdint>
#include <queue>
#include <algorithm>
#include "ClimateFileFormat.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

//============================================================
// Basin Detection Parameters
//============================================================
struct BasinParams {
    int minBasinSize;        // Minimum pixels to be a basin (discard tiny basins)
    double depthThreshold;   // Depth difference threshold for basin boundaries
    int searchRadius;        // Local minima search radius (regional scale)
    int smoothRadius;        // Pre-smoothing radius to reduce noise (0 = no smoothing)
};

//============================================================
// Host: Load Bathymetry
//============================================================
bool loadBathymetry(const std::string& filename, 
                    std::vector<float>& bathymetry,
                    int& width, int& height)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Cannot open " << filename << "\n";
        return false;
    }
    
    std::cout << "📖 Loading bathymetry: " << filename << "\n";
    
    // Read header
    ClimateFileHeader header;
    file.read(reinterpret_cast<char*>(&header.magic), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&header.width), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&header.height), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&header.channels), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&header.dtype), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&header.version), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&header.depth), sizeof(int32_t));
    
    if (header.magic != BATH_MAGIC) {
        std::cerr << "❌ Invalid BATH file magic: 0x" << std::hex << header.magic << "\n";
        return false;
    }
    
    width = header.width;
    height = header.height;
    
    std::cout << "   Resolution: " << width << " × " << height << "\n";
    
    // Read data
    size_t numPixels = width * height;
    bathymetry.resize(numPixels);
    file.read(reinterpret_cast<char*>(bathymetry.data()), numPixels * sizeof(float));
    
    file.close();
    
    // Stats
    float minDepth = *std::min_element(bathymetry.begin(), bathymetry.end());
    float maxDepth = *std::max_element(bathymetry.begin(), bathymetry.end());
    std::cout << "   Depth range: [" << (int)minDepth << "m, " << (int)maxDepth << "m]\n\n";
    
    return true;
}

//============================================================
// Kernel: Gaussian Smooth Bathymetry
//============================================================
__global__ void gaussianSmoothKernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int width, int height,
                                     int radius)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    
    int x = idx % width;
    int y = idx / width;
    
    float sum = 0.0f;
    float weightSum = 0.0f;
    float sigma = radius / 3.0f;
    
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = (x + dx + width) % width;  // Wrap longitude
            int ny = y + dy;
            
            if (ny < 0 || ny >= height) continue;
            
            int nIdx = ny * width + nx;
            
            float dist = sqrtf(dx*dx + dy*dy);
            float weight = expf(-(dist * dist) / (2.0f * sigma * sigma));
            
            sum += input[nIdx] * weight;
            weightSum += weight;
        }
    }
    
    output[idx] = sum / weightSum;
}

//============================================================
// Kernel: Find Local Minima (Basin Centers)
//============================================================
__global__ void findLocalMinimaKernel(const float* __restrict__ bathymetry,
                                      int* __restrict__ basinSeeds,
                                      int width, int height,
                                      int searchRadius)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    
    int x = idx % width;
    int y = idx / width;
    
    float centerDepth = bathymetry[idx];
    bool isMinimum = true;
    
    // Check if this is a local minimum (deepest point in neighborhood)
    for (int dy = -searchRadius; dy <= searchRadius; dy++) {
        for (int dx = -searchRadius; dx <= searchRadius; dx++) {
            if (dx == 0 && dy == 0) continue;
            
            int nx = (x + dx + width) % width;  // Wrap longitude
            int ny = y + dy;
            
            if (ny < 0 || ny >= height) continue;
            
            int nIdx = ny * width + nx;
            if (bathymetry[nIdx] < centerDepth) {
                isMinimum = false;
                break;
            }
        }
        if (!isMinimum) break;
    }
    
    // Mark as seed if it's a local minimum
    basinSeeds[idx] = isMinimum ? 1 : 0;
}

//============================================================
// Host: Watershed-Based Basin Segmentation
//============================================================
void segmentBasins(const std::vector<float>& bathymetry,
                   std::vector<int>& basinMap,
                   int width, int height,
                   const BasinParams& params)
{
    size_t totalPixels = width * height;
    
    // Upload bathymetry to GPU
    float *d_bathymetry, *d_smoothed;
    int* d_basinSeeds;
    
    CUDA_CHECK(cudaMalloc(&d_bathymetry, totalPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_smoothed, totalPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_basinSeeds, totalPixels * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_bathymetry, bathymetry.data(),
                         totalPixels * sizeof(float), cudaMemcpyHostToDevice));
    
    int threads = 256;
    int blocks = (totalPixels + threads - 1) / threads;
    
    // Pre-smooth bathymetry to reduce noise
    if (params.smoothRadius > 0) {
        std::cout << "🔄 Smoothing bathymetry (radius " << params.smoothRadius << " pixels)...\n";
        gaussianSmoothKernel<<<blocks, threads>>>(d_bathymetry, d_smoothed,
                                                  width, height, params.smoothRadius);
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        CUDA_CHECK(cudaMemcpy(d_smoothed, d_bathymetry, totalPixels * sizeof(float),
                             cudaMemcpyDeviceToDevice));
    }
    
    std::cout << "🔍 Detecting basin centers...\n";
    
    // Find local minima on smoothed bathymetry
    findLocalMinimaKernel<<<blocks, threads>>>(d_smoothed, d_basinSeeds,
                                               width, height, params.searchRadius);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Download seeds
    std::vector<int> basinSeeds(totalPixels);
    CUDA_CHECK(cudaMemcpy(basinSeeds.data(), d_basinSeeds,
                         totalPixels * sizeof(int), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_bathymetry));
    CUDA_CHECK(cudaFree(d_smoothed));
    CUDA_CHECK(cudaFree(d_basinSeeds));
    
    // Count basin centers
    std::vector<int> seedIndices;
    for (int i = 0; i < totalPixels; ++i) {
        if (basinSeeds[i] == 1) {
            seedIndices.push_back(i);
        }
    }
    
    std::cout << "   Found " << seedIndices.size() << " potential basin centers\n";
    
    // Watershed segmentation (CPU - easier for priority queue)
    std::cout << "🌊 Segmenting basins (watershed)...\n";
    
    basinMap.assign(totalPixels, -1);  // -1 = unassigned
    
    // Priority queue: (depth, pixel_index, basin_id)
    struct PixelDepth {
        float depth;
        int idx;
        int basinId;
        bool operator<(const PixelDepth& other) const {
            return depth > other.depth;  // Min-heap (deepest first)
        }
    };
    
    std::priority_queue<PixelDepth> queue;
    
    // Initialize with basin seeds
    for (int i = 0; i < seedIndices.size(); ++i) {
        int idx = seedIndices[i];
        basinMap[idx] = i;  // Assign basin ID
        queue.push({bathymetry[idx], idx, i});
    }
    
    // Expand basins from seeds
    int processed = 0;
    while (!queue.empty()) {
        PixelDepth current = queue.top();
        queue.pop();
        
        int x = current.idx % width;
        int y = current.idx / width;
        
        // Check 4-neighbors
        int dx[] = {-1, 1, 0, 0};
        int dy[] = {0, 0, -1, 1};
        
        for (int d = 0; d < 4; ++d) {
            int nx = (x + dx[d] + width) % width;  // Wrap longitude
            int ny = y + dy[d];
            
            if (ny < 0 || ny >= height) continue;
            
            int nIdx = ny * width + nx;
            
            if (basinMap[nIdx] == -1) {  // Unassigned
                // Check depth threshold
                float depthDiff = std::abs(bathymetry[nIdx] - current.depth);
                if (depthDiff < params.depthThreshold) {
                    basinMap[nIdx] = current.basinId;
                    queue.push({bathymetry[nIdx], nIdx, current.basinId});
                }
            }
        }
        
        processed++;
        if (processed % 100000 == 0) {
            std::cout << "   Progress: " << (processed * 100 / totalPixels) << "%\r" << std::flush;
        }
    }
    
    std::cout << "\n";
    
    // Merge small basins with neighbors
    std::vector<int> basinSizes(seedIndices.size(), 0);
    for (int id : basinMap) {
        if (id >= 0 && id < basinSizes.size()) {
            basinSizes[id]++;
        }
    }
    
    std::cout << "📊 Merging small basins...\n";
    int merged = 0;
    for (int i = 0; i < basinSizes.size(); ++i) {
        if (basinSizes[i] < params.minBasinSize && basinSizes[i] > 0) {
            merged++;
            // Find largest neighbor basin
            // (Simplified: assign to basin 0 for now)
            for (int& id : basinMap) {
                if (id == i) id = 0;
            }
        }
    }
    
    std::cout << "   Merged " << merged << " small basins\n";
    
    // Renumber basins consecutively
    std::vector<int> basinRemap(seedIndices.size(), -1);
    int nextId = 0;
    for (int& id : basinMap) {
        if (id >= 0 && id < seedIndices.size()) {
            if (basinRemap[id] == -1) {
                basinRemap[id] = nextId++;
            }
            id = basinRemap[id];
        }
    }
    
    std::cout << "   Final basin count: " << nextId << "\n\n";
}

//============================================================
// Host: Save Basin Map
//============================================================
bool saveBasinMap(const std::string& filename,
                  const std::vector<int>& basinMap,
                  int width, int height)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Cannot open " << filename << "\n";
        return false;
    }
    
    std::cout << "💾 Saving basin map: " << filename << "\n";
    
    // Write header
    ClimateFileHeader header;
    header.magic = BASIN_MAGIC;
    header.width = width;
    header.height = height;
    header.channels = 1;
    header.dtype = 2;  // Int32
    header.version = 1;
    header.depth = 1;
    
    file.write(reinterpret_cast<const char*>(&header.magic), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&header.width), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&header.height), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&header.channels), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&header.dtype), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&header.version), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&header.depth), sizeof(int32_t));
    
    // Write data
    file.write(reinterpret_cast<const char*>(basinMap.data()),
               basinMap.size() * sizeof(int));
    
    file.close();
    
    size_t fileSize = 28 + basinMap.size() * sizeof(int);
    std::cout << "   Size: " << (fileSize / (1024.0*1024.0)) << " MB\n";
    
    return true;
}

//============================================================
// Host: Compute Basin Statistics
//============================================================
void computeBasinStats(const std::vector<int>& basinMap,
                       const std::vector<float>& bathymetry,
                       int width, int height)
{
    int maxBasinId = *std::max_element(basinMap.begin(), basinMap.end());
    
    std::vector<int> basinCounts(maxBasinId + 1, 0);
    std::vector<double> basinAvgDepth(maxBasinId + 1, 0.0);
    
    for (size_t i = 0; i < basinMap.size(); ++i) {
        int id = basinMap[i];
        if (id >= 0 && id <= maxBasinId) {
            basinCounts[id]++;
            basinAvgDepth[id] += bathymetry[i];
        }
    }
    
    std::cout << "\n📍 Basin Statistics:\n";
    for (int i = 0; i <= maxBasinId; ++i) {
        if (basinCounts[i] > 0) {
            double pct = (basinCounts[i] * 100.0) / basinMap.size();
            double avgDepth = basinAvgDepth[i] / basinCounts[i];
            std::cout << "   Basin " << i << ": " 
                      << pct << "% coverage, avg depth " 
                      << (int)avgDepth << "m\n";
        }
    }
    std::cout << "\n";
}

//============================================================
// Main
//============================================================
int main(int argc, char** argv)
{
    std::string bathFile = "Bathymetry.bath";
    std::string basinFile = "OceanBasins.basin";
    
    BasinParams params;
    params.minBasinSize = 10000;       // ~200km² minimum basin
    params.depthThreshold = 2000.0;    // 2000m depth change = boundary
    params.searchRadius = 50;          // Regional-scale search (80km at equator)
    params.smoothRadius = 10;          // Smooth bathymetry (16km) before detection
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0 && i+1 < argc)
            bathFile = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc)
            basinFile = argv[++i];
        else if (strcmp(argv[i], "--min-size") == 0 && i+1 < argc)
            params.minBasinSize = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threshold") == 0 && i+1 < argc)
            params.depthThreshold = atof(argv[++i]);
        else if (strcmp(argv[i], "--radius") == 0 && i+1 < argc)
            params.searchRadius = atoi(argv[++i]);
        else if (strcmp(argv[i], "--smooth") == 0 && i+1 < argc)
            params.smoothRadius = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -i <file>         Input bathymetry file (.bath)\n"
                      << "  -o <file>         Output basin file (.basin)\n"
                      << "  --min-size <px>   Minimum basin size in pixels (default 10000)\n"
                      << "  --threshold <m>   Depth threshold for boundaries (default 2000)\n"
                      << "  --radius <px>     Search radius for basin centers (default 50)\n"
                      << "  --smooth <px>     Pre-smoothing radius (default 10, 0=none)\n";
            return 0;
        }
    }
    
    std::cout << "🌊 Basin Detector\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "Input:  " << bathFile << "\n";
    std::cout << "Output: " << basinFile << "\n\n";
    
    // Load bathymetry
    std::vector<float> bathymetry;
    int width, height;
    
    if (!loadBathymetry(bathFile, bathymetry, width, height)) {
        return EXIT_FAILURE;
    }
    
    // Segment basins
    std::vector<int> basinMap;
    segmentBasins(bathymetry, basinMap, width, height, params);
    
    // Statistics
    computeBasinStats(basinMap, bathymetry, width, height);
    
    // Save
    if (!saveBasinMap(basinFile, basinMap, width, height)) {
        return EXIT_FAILURE;
    }
    
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "✅ Basin detection complete!\n";
    std::cout << "\nNext steps:\n";
    std::cout << "  1. Visualize: ./BathymetryViewer " << bathFile << " " << basinFile << "\n";
    std::cout << "  2. Use for gyre placement\n";
    
    return 0;
}
