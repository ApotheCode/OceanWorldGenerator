//============================================================
// HeightmapProcessor.cu
// Loads and processes elevation data for terrain integration
// Author: Mark Devereux (2025-10-18)
//============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cstring>

constexpr int WIDTH = 3600;
constexpr int HEIGHT = 1800;
constexpr double EARTH_RADIUS_KM = 6371.0;
constexpr double DEG_TO_RAD = 3.14159265358979323846 / 180.0;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

struct TerrainData {
    float* elevation;      // Meters (-11000 to +8848)
    unsigned char* isLand; // 1=land, 0=ocean
    float* oceanDepth;     // Positive values for ocean
    float* continentality; // Distance to coast (km)
};

//============================================================
// Kernel: Generate Land/Ocean Mask
//============================================================
__global__ void generateLandMaskKernel(const float* __restrict__ elevation,
                                       unsigned char* __restrict__ isLand,
                                       float* __restrict__ oceanDepth,
                                       int totalPixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalPixels) return;
    
    float elev = elevation[idx];
    
    if (elev > 0.0f) {
        isLand[idx] = 1;
        oceanDepth[idx] = 0.0f;
    } else {
        isLand[idx] = 0;
        oceanDepth[idx] = -elev;  // Positive depth
    }
}

//============================================================
// Kernel: Calculate Continentality (Distance to Coast)
// Uses iterative expansion similar to spatial fill
//============================================================
__global__ void calculateContinentalityKernel(const unsigned char* __restrict__ isLand,
                                              float* __restrict__ continentality,
                                              int width, int height,
                                              int searchRadius,
                                              int* __restrict__ changeCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    
    int x = idx % width;
    int y = idx / width;
    
    // If already calculated (non-infinite), skip
    if (continentality[idx] < 1e6f) return;
    
    float minDist = 1e9f;
    bool foundCoast = false;
    
    // Search for nearest opposite type (land finds ocean, ocean finds land)
    unsigned char myType = isLand[idx];
    
    for (int dy = -searchRadius; dy <= searchRadius; dy++) {
        for (int dx = -searchRadius; dx <= searchRadius; dx++) {
            if (dx == 0 && dy == 0) continue;
            
            int nx = (x + dx + width) % width;
            int ny = y + dy;
            
            if (ny < 0 || ny >= height) continue;
            
            int nIdx = ny * width + nx;
            
            // Found opposite type?
            if (isLand[nIdx] != myType) {
                // Calculate spherical distance
                double lat1 = (90.0 - y * 180.0 / height) * DEG_TO_RAD;
                double lon1 = (x * 360.0 / width) * DEG_TO_RAD;
                double lat2 = (90.0 - ny * 180.0 / height) * DEG_TO_RAD;
                double lon2 = (nx * 360.0 / width) * DEG_TO_RAD;
                
                // Haversine formula
                double dlat = lat2 - lat1;
                double dlon = lon2 - lon1;
                double a = sin(dlat/2) * sin(dlat/2) + 
                          cos(lat1) * cos(lat2) * 
                          sin(dlon/2) * sin(dlon/2);
                double c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a));
                float dist = EARTH_RADIUS_KM * c;
                
                if (dist < minDist) {
                    minDist = dist;
                    foundCoast = true;
                }
            }
        }
    }
    
    if (foundCoast) {
        continentality[idx] = minDist;
        atomicAdd(changeCount, 1);
    }
}

//============================================================
// Host: Load Heightmap (Multiple Formats)
//============================================================
bool loadHeightmap(const std::string& filename, std::vector<float>& elevation)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Failed to open heightmap: " << filename << "\n";
        return false;
    }
    
    // Check file extension
    bool isRaw = filename.find(".raw") != std::string::npos || 
                 filename.find(".bin") != std::string::npos;
    
    elevation.resize(WIDTH * HEIGHT);
    
    if (isRaw) {
        // Raw 16-bit signed integers (common DEM format)
        std::vector<int16_t> rawData(WIDTH * HEIGHT);
        file.read(reinterpret_cast<char*>(rawData.data()), WIDTH * HEIGHT * sizeof(int16_t));
        
        for (size_t i = 0; i < rawData.size(); ++i) {
            elevation[i] = static_cast<float>(rawData[i]);
        }
        
        std::cout << "📁 Loaded RAW heightmap (16-bit)\n";
    } else {
        // Assume 32-bit float
        file.read(reinterpret_cast<char*>(elevation.data()), 
                 WIDTH * HEIGHT * sizeof(float));
        
        std::cout << "📁 Loaded float heightmap (32-bit)\n";
    }
    
    file.close();
    
    // Statistics
    float minElev = 1e9f, maxElev = -1e9f;
    size_t landCount = 0;
    
    for (float e : elevation) {
        minElev = std::min(minElev, e);
        maxElev = std::max(maxElev, e);
        if (e > 0.0f) landCount++;
    }
    
    std::cout << "   Elevation range: " << minElev << " to " << maxElev << " m\n";
    std::cout << "   Land coverage: " << (landCount * 100.0 / elevation.size()) << "%\n";
    
    return true;
}

//============================================================
// Host: Generate Synthetic Test Terrain
//============================================================
void generateTestTerrain(std::vector<float>& elevation, const std::string& type)
{
    elevation.resize(WIDTH * HEIGHT);
    
    if (type == "single_continent") {
        // One large continent at 30°N, 45°E
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                double lat = 90.0 - (y * 180.0 / HEIGHT);
                double lon = (x * 360.0 / WIDTH) - 180.0;
                
                // Elliptical continent
                double dx = (lon - 45.0) / 40.0;
                double dy = (lat - 30.0) / 25.0;
                double dist = sqrt(dx*dx + dy*dy);
                
                if (dist < 1.0) {
                    // Mountain at center, decreasing to coast
                    float height = 5000.0f * (1.0f - dist) * (1.0f - dist);
                    elevation[y * WIDTH + x] = height;
                } else {
                    // Ocean depth based on distance from coast
                    elevation[y * WIDTH + x] = -4000.0f;
                }
            }
        }
        std::cout << "🗺️  Generated single continent test terrain\n";
        
    } else if (type == "archipelago") {
        // Multiple islands
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                elevation[y * WIDTH + x] = -3000.0f;  // Default ocean
                
                // Add several island chains
                double lat = 90.0 - (y * 180.0 / HEIGHT);
                double lon = (x * 360.0 / WIDTH) - 180.0;
                
                // Island 1: Equatorial
                double d1 = sqrt(pow((lon - 20)/8, 2) + pow(lat/5, 2));
                if (d1 < 1.0) elevation[y * WIDTH + x] = 800.0f * (1.0f - d1);
                
                // Island 2: Mid-latitude
                double d2 = sqrt(pow((lon + 60)/12, 2) + pow((lat-40)/8, 2));
                if (d2 < 1.0) elevation[y * WIDTH + x] = 1500.0f * (1.0f - d2);
                
                // Island 3: Polar
                double d3 = sqrt(pow((lon + 120)/10, 2) + pow((lat+70)/6, 2));
                if (d3 < 1.0) elevation[y * WIDTH + x] = 500.0f * (1.0f - d3);
            }
        }
        std::cout << "🏝️  Generated archipelago test terrain\n";
        
    } else if (type == "pure_ocean") {
        // Validation case: restore OceanWorld
        for (size_t i = 0; i < elevation.size(); ++i) {
            elevation[i] = -4000.0f;
        }
        std::cout << "🌊 Generated pure ocean (OceanWorld restore)\n";
    }
}

//============================================================
// Main Processing Function
//============================================================
TerrainData processHeightmap(const std::string& source)
{
    TerrainData terrain;
    size_t totalPixels = WIDTH * HEIGHT;
    terrain.elevation = new float[totalPixels];
    terrain.isLand = new unsigned char[totalPixels];
    terrain.oceanDepth = new float[totalPixels];
    terrain.continentality = new float[totalPixels];

    std::vector<float> h_elevation;
    
    // Load or generate terrain
    if (source == "single_continent" || source == "archipelago" || source == "pure_ocean") {
        generateTestTerrain(h_elevation, source);
    } else {
        if (!loadHeightmap(source, h_elevation)) {
            std::cerr << "❌ Failed to load heightmap, using pure ocean fallback\n";
            generateTestTerrain(h_elevation, "pure_ocean");
        }
    }
    
    // Copy to terrain struct
    memcpy(terrain.elevation, h_elevation.data(), h_elevation.size() * sizeof(float));
    
    // Allocate GPU memory
    float* d_elevation;
    unsigned char* d_isLand;
    float* d_oceanDepth;
    float* d_continentality;
    int* d_changeCount;
    
    
    
    CUDA_CHECK(cudaMalloc(&d_elevation, totalPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_isLand, totalPixels * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_oceanDepth, totalPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_continentality, totalPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_changeCount, sizeof(int)));
    
    // Upload elevation
    CUDA_CHECK(cudaMemcpy(d_elevation, terrain.elevation, 
                         totalPixels * sizeof(float), cudaMemcpyHostToDevice));
    
    // Generate land mask
    std::cout << "🗺️  Generating land/ocean mask...\n";
    int threads = 256;
    int blocks = (totalPixels + threads - 1) / threads;
    
    generateLandMaskKernel<<<blocks, threads>>>(d_elevation, d_isLand, d_oceanDepth, totalPixels);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Download mask for stats
    CUDA_CHECK(cudaMemcpy(terrain.isLand, d_isLand, 
                         totalPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(terrain.oceanDepth, d_oceanDepth, 
                         totalPixels * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Calculate continentality
    std::cout << "📏 Calculating continentality (distance to coast)...\n";
    
    // Initialize to infinity
    std::vector<float> h_continentality(totalPixels, 1e9f);
    
    // Coast pixels get distance = 0
    for (size_t i = 0; i < totalPixels; ++i) {
        int x = i % WIDTH;
        int y = i / WIDTH;
        
        // Check if any neighbor is opposite type
        bool isCoast = false;
        for (int dy = -1; dy <= 1 && !isCoast; dy++) {
            for (int dx = -1; dx <= 1 && !isCoast; dx++) {
                int nx = (x + dx + WIDTH) % WIDTH;
                int ny = y + dy;
                if (ny >= 0 && ny < HEIGHT) {
                    int nIdx = ny * WIDTH + nx;
                    if (terrain.isLand[nIdx] != terrain.isLand[i]) {
                        isCoast = true;
                    }
                }
            }
        }
        
        if (isCoast) h_continentality[i] = 0.0f;
    }
    
    CUDA_CHECK(cudaMemcpy(d_continentality, h_continentality.data(), 
                         totalPixels * sizeof(float), cudaMemcpyHostToDevice));
    
    // Iteratively expand distance calculation
    int maxRadius = 500;  // ~500 pixels = ~5000 km
    int radiusStep = 20;
    
    for (int radius = 1; radius <= maxRadius; radius += radiusStep) {
        int h_change = 0;
        CUDA_CHECK(cudaMemcpy(d_changeCount, &h_change, sizeof(int), cudaMemcpyHostToDevice));
        
        calculateContinentalityKernel<<<blocks, threads>>>(
            d_isLand, d_continentality, WIDTH, HEIGHT, radius, d_changeCount);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_change, d_changeCount, sizeof(int), cudaMemcpyDeviceToHost));
        
        if (h_change == 0 && radius > 40) break;
        
        if (radius % 100 == 0) {
            std::cout << "   Radius: " << radius << " pixels\r" << std::flush;
        }
    }
    
    std::cout << "\n   ✅ Continentality calculated\n";
    
    // Download results
    CUDA_CHECK(cudaMemcpy(terrain.continentality, d_continentality, 
                         totalPixels * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_elevation));
    CUDA_CHECK(cudaFree(d_isLand));
    CUDA_CHECK(cudaFree(d_oceanDepth));
    CUDA_CHECK(cudaFree(d_continentality));
    CUDA_CHECK(cudaFree(d_changeCount));
    
    std::cout << "✅ Terrain processing complete\n\n";
    
    return terrain;
}

//============================================================
// Export Terrain Data
//============================================================
void exportTerrainData(const std::string& outputPath, const TerrainData& terrain)
{
    std::ofstream file(outputPath, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Failed to create output file\n";
        return;
    }
    
    // Simple header
    int width = WIDTH;
    int height = HEIGHT;
    file.write(reinterpret_cast<const char*>(&width), sizeof(int));
    file.write(reinterpret_cast<const char*>(&height), sizeof(int));
    
    // Write all arrays
    size_t totalPixels = WIDTH * HEIGHT;
    file.write(reinterpret_cast<const char*>(terrain.elevation), totalPixels * sizeof(float));
    file.write(reinterpret_cast<const char*>(terrain.isLand), totalPixels * sizeof(unsigned char));
    file.write(reinterpret_cast<const char*>(terrain.oceanDepth), totalPixels * sizeof(float));
    file.write(reinterpret_cast<const char*>(terrain.continentality), totalPixels * sizeof(float));
    
    file.close();
    
    std::cout << "💾 Terrain data exported to: " << outputPath << "\n";

    delete[] terrain.elevation;
    delete[] terrain.isLand;
    delete[] terrain.oceanDepth;
    delete[] terrain.continentality;
}

//============================================================
// Main (Standalone Testing)
//============================================================
int main(int argc, char** argv)
{
    std::string source = "single_continent";  // Default test
    std::string output = "./output/TerrainData.bin";
    
    if (argc >= 2) source = argv[1];
    if (argc >= 3) output = argv[2];
    
    std::cout << "🏔️  Heightmap Processor\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "Source: " << source << "\n";
    std::cout << "Output: " << output << "\n\n";
    
    TerrainData terrain = processHeightmap(source);
    exportTerrainData(output, terrain);
    
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "✅ Processing complete!\n";
    
    return 0;
}