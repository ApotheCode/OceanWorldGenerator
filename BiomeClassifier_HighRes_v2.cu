//============================================================
// BiomeClassifier_HighResolution.cu
// Uses 20 biome zones to visualize PNPL circulation patterns
// UPDATED: Uses standard ClimateFileFormat.h
// Author: Mark Devereux (2025-11-13)
//============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <string>
#include <cstring>
#include <cstdint>
#include "ClimateFileFormat.h"  // Standard header format

constexpr size_t WIDTH      = 3600;
constexpr size_t HEIGHT     = 1800;
constexpr size_t MONTHS     = 12;
constexpr size_t VARIABLES  = 14;
constexpr size_t TOTAL_CELLS = WIDTH * HEIGHT;

constexpr int VAR_SST       = 1;
constexpr int VAR_AIRTEMP   = 13;

constexpr int OUTPUT_CLASSES = 20;  // 20 zones = 5% each (fine enough to see PNPL)

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

//============================================================
// Device: High-Resolution Classification
//============================================================
__device__ __forceinline__ unsigned char classifyByTemp(double meanTemp)
{
    meanTemp = fmax(0.0, fmin(1.0, meanTemp));
    
    // Map [0,1] to [0,19] (20 zones)
    unsigned char biome = (unsigned char)(meanTemp * 20.0);
    
    // Clamp to valid range
    if (biome >= OUTPUT_CLASSES) biome = OUTPUT_CLASSES - 1;
    
    return biome;
}

//============================================================
// Kernel: Classify with Terrain Awareness
//============================================================
__global__ void classifyBiomesKernel(const double* __restrict__ data,
                                      const unsigned char* __restrict__ landMask,
                                      unsigned char* __restrict__ biomeMap,
                                      size_t totalCells,
                                      size_t months,
                                      size_t variables)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalCells) return;

    bool isLand = (landMask != nullptr) ? (landMask[idx] == 1) : false;
    int varIndex = isLand ? VAR_AIRTEMP : VAR_SST;
    
    size_t cellBase = idx * variables * months;
    size_t varBase = cellBase + (varIndex * months);
    
    double meanTemp = 0.0;
    int validMonths = 0;
    
    for (size_t m = 0; m < months; ++m) {
        double temp = data[varBase + m];
        if (isfinite(temp)) {
            meanTemp += temp;
            validMonths++;
        }
    }
    
    if (validMonths == 0) {
        biomeMap[idx] = 255;
        return;
    }
    
    meanTemp /= validMonths;
    biomeMap[idx] = classifyByTemp(meanTemp);
}

//============================================================
// Load Functions
//============================================================
bool loadClimateDataStandard(const std::string& filename, std::vector<double>& data)
{
    ClimateFileHeader header;
    if (!loadClimateData(filename, data, header)) {
        return false;
    }
    
    // Verify dimensions match expected
    if (header.width != WIDTH || header.height != HEIGHT) {
        std::cerr << "❌ Dimension mismatch!\n";
        std::cerr << "   Expected: " << WIDTH << "×" << HEIGHT << "\n";
        std::cerr << "   Got: " << header.width << "×" << header.height << "\n";
        return false;
    }
    
    return true;
}

bool loadLandMask(const std::string& filename, std::vector<unsigned char>& mask)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "⚠️ No terrain - pure ocean mode\n";
        mask.resize(TOTAL_CELLS, 0);
        return false;
    }
    
    int width, height;
    file.read(reinterpret_cast<char*>(&width), sizeof(int));
    file.read(reinterpret_cast<char*>(&height), sizeof(int));
    
    file.seekg(sizeof(int) * 2 + TOTAL_CELLS * sizeof(float), std::ios::beg);
    
    mask.resize(TOTAL_CELLS);
    file.read(reinterpret_cast<char*>(mask.data()), TOTAL_CELLS);
    file.close();
    
    size_t landCount = 0;
    for (auto m : mask) if (m == 1) landCount++;
    std::cout << "✅ Land: " << (landCount * 100.0 / TOTAL_CELLS) << "%\n";
    return true;
}

bool saveBiomeMap(const std::string& filename, const std::vector<unsigned char>& map)
{
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "❌ Failed to create: " << filename << "\n";
        return false;
    }
    
    // Create standard header for biome output
    ClimateFileHeader header;
    header.magic = BIOME_MAGIC;
    header.width = WIDTH;
    header.height = HEIGHT;
    header.channels = 1;  // Single biome layer
    header.dtype = 2;     // uint8/unsigned char
    header.version = 1;
    header.depth = 1;     // Single time step
    
    // Write standard 28-byte header
    writeClimateHeader(out, header);
    
    // Write biome map data
    out.write(reinterpret_cast<const char*>(map.data()), map.size());
    
    out.close();
    
    std::cout << "✓ Biome map saved: " << filename << "\n";
    return true;
}

//============================================================
// Main
//============================================================
int main(int argc, char** argv)
{
    std::string climateFile = "output/Earth_Climate_PNPL.bin";
    std::string terrainFile = "output/TerrainData.bin";
    std::string outputFile = "output/BiomeMap_HighRes.bin";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0 && i+1 < argc)
            climateFile = argv[++i];
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc)
            terrainFile = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc)
            outputFile = argv[++i];
    }

    std::cout << "🗺️ High-Resolution Biome Classifier (20 zones)\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "Climate: " << climateFile << "\n";
    std::cout << "Terrain: " << terrainFile << "\n";
    std::cout << "Output:  " << outputFile << "\n\n";

    std::vector<double> climateData;
    if (!loadClimateDataStandard(climateFile, climateData))
        return EXIT_FAILURE;
    
    std::vector<unsigned char> landMask;
    loadLandMask(terrainFile, landMask);
    
    // GPU
    double* d_data;
    unsigned char* d_landMask;
    unsigned char* d_biomeMap;
    
    size_t dataBytes = climateData.size() * sizeof(double);
    size_t maskBytes = TOTAL_CELLS * sizeof(unsigned char);
    
    CUDA_CHECK(cudaMalloc(&d_data, dataBytes));
    CUDA_CHECK(cudaMalloc(&d_landMask, maskBytes));
    CUDA_CHECK(cudaMalloc(&d_biomeMap, maskBytes));
    
    CUDA_CHECK(cudaMemcpy(d_data, climateData.data(), dataBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_landMask, landMask.data(), maskBytes, cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    size_t blocks = (TOTAL_CELLS + blockSize - 1) / blockSize;
    
    std::cout << "⚙️ Classifying with 20 temperature zones...\n";
    
    classifyBiomesKernel<<<blocks, blockSize>>>(
        d_data, d_landMask, d_biomeMap, TOTAL_CELLS, MONTHS, VARIABLES);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<unsigned char> biomeMap(TOTAL_CELLS);
    CUDA_CHECK(cudaMemcpy(biomeMap.data(), d_biomeMap, maskBytes, cudaMemcpyDeviceToHost));
    
    saveBiomeMap(outputFile, biomeMap);
    
    // Count distribution
    std::vector<size_t> counts(OUTPUT_CLASSES + 1, 0);
    for (auto b : biomeMap) {
        if (b < OUTPUT_CLASSES) counts[b]++;
        else counts[OUTPUT_CLASSES]++;
    }
    
    std::cout << "\n📊 Distribution (20 zones):\n";
    for (int i = 0; i < OUTPUT_CLASSES; ++i) {
        if (counts[i] > 0) {
            double pct = (counts[i] * 100.0) / TOTAL_CELLS;
            std::cout << "   Zone " << i << ": " << pct << "%\n";
        }
    }
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_landMask));
    CUDA_CHECK(cudaFree(d_biomeMap));
    
    std::cout << "\n✅ Saved: " << outputFile << "\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    
    return 0;
}