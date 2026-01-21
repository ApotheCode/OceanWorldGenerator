//============================================================
// EarthOceanWorldGenerator.cu
// Earth-Based Ocean World Bathymetry Generator
// 
// Takes real Earth data and creates a realistic ocean world:
//   - Preserves GEBCO bathymetry for ocean basins (untouched)
//   - Remaps SRTM land elevations to ocean depths
//   - Smooths transitions at old coastlines (Gaussian blur)
//   - Adds procedural detail on former continents
//   - Outputs unified bathymetry matching real ocean physics
// 
// Input CSV files (3600×1800, equirectangular):
//   - SRTM elevation (meters, land positive)
//   - GEBCO bathymetry (meters, ocean negative)
// 
// Output:
//   - Unified bathymetry (.BATH file)
//   - Ocean basin map (derived from real basins)
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
#include <sstream>
#include <algorithm>
#include "ClimateFileFormat.h"

constexpr int WIDTH = STANDARD_WIDTH;   // 3600
constexpr int HEIGHT = STANDARD_HEIGHT; // 1800
constexpr double M_PI = 3.14159265358979323846;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr uint32_t BATH_MAGIC = 0x42415448;  // "BATH"

//============================================================
// Processing Parameters
//============================================================
struct RemapParams {
    double shallowDepth;      // Depth for lowlands (e.g., -2500m)
    double deepDepth;         // Depth for mountains (e.g., -6000m)
    double maxElevation;      // Max land elevation (e.g., 8848m Everest)
    double remapExponent;     // Power curve for remapping (>1 = more deep areas)
    int coastalBlurRadius;    // Pixels to blur around old coastlines
    bool addProceduralDetail; // Add abyssal hills on former land
    double proceduralAmp;     // Amplitude of added detail
};

//============================================================
// Host: Load CSV File
//============================================================
bool loadCSV(const std::string& filename, std::vector<float>& data, 
             int expectedWidth, int expectedHeight)
{
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "❌ Cannot open " << filename << "\n";
        return false;
    }
    
    std::cout << "📖 Loading " << filename << "...\n";
    
    data.clear();
    data.reserve(expectedWidth * expectedHeight);
    
    std::string line;
    int row = 0;
    
    while (std::getline(file, line) && row < expectedHeight) {
        std::stringstream ss(line);
        std::string value;
        int col = 0;
        
        while (std::getline(ss, value, ',') && col < expectedWidth) {
            float val = std::stof(value);
            data.push_back(val);
            col++;
        }
        
        if (col != expectedWidth) {
            std::cerr << "⚠️ Row " << row << " has " << col 
                      << " columns, expected " << expectedWidth << "\n";
        }
        
        row++;
        
        if (row % 300 == 0) {
            std::cout << "   Progress: " << (row * 100 / expectedHeight) << "%\r" << std::flush;
        }
    }
    
    std::cout << "   ✅ Loaded " << data.size() << " values\n";
    
    file.close();
    
    if (data.size() != expectedWidth * expectedHeight) {
        std::cerr << "⚠️ Size mismatch: got " << data.size() 
                  << ", expected " << (expectedWidth * expectedHeight) << "\n";
        return false;
    }
    
    return true;
}

//============================================================
// Device: Simple Hash for Procedural Noise
//============================================================
__device__ __forceinline__ float simpleHash(int x, int y, int seed)
{
    unsigned int h = seed;
    h ^= x * 374761393U;
    h ^= y * 668265263U;
    h = (h ^ (h >> 16)) * 0x85ebca6bU;
    h = (h ^ (h >> 13)) * 0xc2b2ae35U;
    return ((h & 0xFFFFFF) / 16777216.0f) * 2.0f - 1.0f;  // [-1, 1]
}

//============================================================
// Device: Procedural Detail (Fast Noise)
//============================================================
__device__ float proceduralDetail(int x, int y, int width, int height, int seed)
{
    // Multi-scale hash-based noise (faster than Perlin on GPU)
    float detail = 0.0f;
    float amplitude = 1.0f;
    int freq = 1;
    
    // 3 octaves of detail
    for (int i = 0; i < 3; ++i) {
        int sx = (x * freq) % width;
        int sy = (y * freq) % height;
        detail += simpleHash(sx, sy, seed + i) * amplitude;
        amplitude *= 0.5f;
        freq *= 2;
    }
    
    return detail;
}

//============================================================
// Device: Smooth Lerp
//============================================================
__device__ __forceinline__ float smoothstep(float t)
{
    return t * t * (3.0f - 2.0f * t);
}

__device__ __forceinline__ float lerp(float a, float b, float t)
{
    return a + (b - a) * t;
}

//============================================================
// Kernel: Remap Land to Ocean Depths
// 
// Handles complementary datasets:
// - SRTM: Real elevation on land, 99999 in ocean
// - GEBCO: Real bathymetry in ocean, 99999 on land
// 
// Logic:
// 1. If GEBCO valid (< 9000) → Keep bathymetry (ocean)
// 2. If SRTM valid (< 10000) → Remap to ocean depth (former land)
// 3. If both 99999 → Use mid-depth default (rare edge case)
//============================================================
__global__ void remapLandKernel(const float* __restrict__ srtm,
                                const float* __restrict__ gebco,
                                float* __restrict__ output,
                                unsigned char* __restrict__ landMask,
                                RemapParams params,
                                int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    
    float elevation = srtm[idx];
    float bathymetry = gebco[idx];
    
    float depth;
    bool isLand = false;
    
    // Datasets are complementary:
    // - SRTM: Real elevation on land, 99999 in ocean
    // - GEBCO: Real bathymetry in ocean, 99999 on land
    
    bool gebcoValid = (bathymetry < 9000.0f);  // Real bathymetry data
    bool srtmValid = (elevation < 10000.0f);   // Real elevation data
    
    if (gebcoValid) {
        // Real ocean bathymetry - keep it unchanged
        depth = bathymetry;
        isLand = false;
        
    } else if (srtmValid) {
        // Real land elevation - remap to ocean depth
        isLand = true;
        
        // Normalize elevation to [0, 1]
        float t = fminf(elevation / params.maxElevation, 1.0f);
        
        // Apply power curve (exponent > 1 pushes more area toward deep)
        t = powf(t, params.remapExponent);
        
        // Remap to ocean depth range
        depth = lerp(params.shallowDepth, params.deepDepth, t);
        
        // Add procedural detail on former land
        if (params.addProceduralDetail) {
            int x = idx % width;
            int y = idx / width;
            float detail = proceduralDetail(x, y, width, height, 42);
            depth += detail * params.proceduralAmp;
        }
        
    } else {
        // Both datasets are 99999 (rare edge case, data gaps)
        // Default to mid-depth ocean
        depth = params.shallowDepth + (params.deepDepth - params.shallowDepth) * 0.5f;
        isLand = false;  // Treat as ocean since we have no data
    }
    
    output[idx] = depth;
    landMask[idx] = isLand ? 1 : 0;
}

//============================================================
// Kernel: Detect Coastline (land adjacent to ocean)
//============================================================
__global__ void detectCoastlineKernel(const unsigned char* __restrict__ landMask,
                                      unsigned char* __restrict__ coastMask,
                                      int width, int height, int dilateRadius)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    
    int x = idx % width;
    int y = idx / width;
    
    bool nearCoast = false;
    
    // Check if this land pixel is near ocean
    if (landMask[idx] == 1) {
        for (int dy = -dilateRadius; dy <= dilateRadius; ++dy) {
            for (int dx = -dilateRadius; dx <= dilateRadius; ++dx) {
                int nx = (x + dx + width) % width;  // Wrap longitude
                int ny = y + dy;
                
                if (ny < 0 || ny >= height) continue;
                
                int nIdx = ny * width + nx;
                if (landMask[nIdx] == 0) {
                    nearCoast = true;
                    break;
                }
            }
            if (nearCoast) break;
        }
    }
    
    coastMask[idx] = nearCoast ? 1 : 0;
}

//============================================================
// Kernel: Gaussian Blur on Coastline
//============================================================
__global__ void coastalBlurKernel(const float* __restrict__ input,
                                  const unsigned char* __restrict__ coastMask,
                                  float* __restrict__ output,
                                  int width, int height, int blurRadius)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    
    int x = idx % width;
    int y = idx / width;
    
    // Only blur coastal areas
    if (coastMask[idx] == 0) {
        output[idx] = input[idx];
        return;
    }
    
    // Gaussian blur with specified radius
    float sum = 0.0f;
    float weightSum = 0.0f;
    float sigma = blurRadius / 3.0f;
    
    for (int dy = -blurRadius; dy <= blurRadius; ++dy) {
        for (int dx = -blurRadius; dx <= blurRadius; ++dx) {
            int nx = (x + dx + width) % width;  // Wrap longitude
            int ny = y + dy;
            
            if (ny < 0 || ny >= height) continue;
            
            int nIdx = ny * width + nx;
            
            // Gaussian weight
            float dist = sqrtf(dx*dx + dy*dy);
            float weight = expf(-(dist * dist) / (2.0f * sigma * sigma));
            
            sum += input[nIdx] * weight;
            weightSum += weight;
        }
    }
    
    output[idx] = sum / weightSum;
}

//============================================================
// Host: Save Bathymetry
//============================================================
bool saveBathymetry(const std::string& filename,
                    const std::vector<float>& bathymetry)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Cannot open " << filename << "\n";
        return false;
    }
    
    // Create header
    ClimateFileHeader header;
    header.magic = BATH_MAGIC;
    header.width = WIDTH;
    header.height = HEIGHT;
    header.channels = 1;
    header.dtype = 1;  // Float32
    header.version = 1;
    header.depth = 1;
    
    // Write header
    file.write(reinterpret_cast<const char*>(&header.magic), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&header.width), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&header.height), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&header.channels), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&header.dtype), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&header.version), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&header.depth), sizeof(int32_t));
    
    // Write data
    file.write(reinterpret_cast<const char*>(bathymetry.data()),
               bathymetry.size() * sizeof(float));
    
    file.close();
    
    std::cout << "💾 Saved: " << filename << " ("
              << (28 + bathymetry.size() * sizeof(float)) / (1024.0*1024.0) << " MB)\n";
    
    return true;
}

//============================================================
// Host: Compute Statistics
//============================================================
void computeStatistics(const std::vector<float>& bathymetry,
                       const std::vector<unsigned char>& landMask)
{
    std::cout << "\n📊 Bathymetry Statistics:\n";
    
    float minDepth = 1e9f;
    float maxDepth = -1e9f;
    double avgDepth = 0.0;
    int oceanPixels = 0;
    int formerLandPixels = 0;
    
    for (size_t i = 0; i < bathymetry.size(); ++i) {
        float d = bathymetry[i];
        minDepth = fmin(minDepth, d);
        maxDepth = fmax(maxDepth, d);
        avgDepth += d;
        
        if (landMask[i] == 1) {
            formerLandPixels++;
        } else {
            oceanPixels++;
        }
    }
    avgDepth /= bathymetry.size();
    
    std::cout << "   Depth range: [" << (int)minDepth << "m, " << (int)maxDepth << "m]\n";
    std::cout << "   Average depth: " << (int)avgDepth << "m\n";
    std::cout << "   Real ocean basins: " << (oceanPixels * 100.0 / bathymetry.size()) << "%\n";
    std::cout << "   Remapped land: " << (formerLandPixels * 100.0 / bathymetry.size()) << "%\n\n";
}

//============================================================
// Main
//============================================================
int main(int argc, char** argv)
{
    std::string srtmFile = "SRTM_RAMP2_TOPO_2000-02-11_rgb_3600x1800.CSV";
    std::string gebcoFile = "GEBCO_BATHY_2002-01-01_gs_3600x1800.CSV";
    std::string outputFile = "EarthOceanWorld.bath";
    
    // Default remapping parameters
    RemapParams params;
    params.shallowDepth = -2500.0;       // Lowlands → continental shelf depth
    params.deepDepth = -6000.0;          // Mountains → abyssal depth
    params.maxElevation = 8848.0;        // Everest height
    params.remapExponent = 1.5;          // Power curve (>1 = more deep areas)
    params.coastalBlurRadius = 20;       // Smooth 20 pixels around coasts
    params.addProceduralDetail = true;   // Add abyssal hills
    params.proceduralAmp = 150.0;        // 150m amplitude
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-s") == 0 && i+1 < argc)
            srtmFile = argv[++i];
        else if (strcmp(argv[i], "-g") == 0 && i+1 < argc)
            gebcoFile = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc)
            outputFile = argv[++i];
        else if (strcmp(argv[i], "--shallow") == 0 && i+1 < argc)
            params.shallowDepth = atof(argv[++i]);
        else if (strcmp(argv[i], "--deep") == 0 && i+1 < argc)
            params.deepDepth = atof(argv[++i]);
        else if (strcmp(argv[i], "--blur") == 0 && i+1 < argc)
            params.coastalBlurRadius = atoi(argv[++i]);
        else if (strcmp(argv[i], "--no-detail") == 0)
            params.addProceduralDetail = false;
        else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -s <file>       SRTM elevation CSV\n"
                      << "  -g <file>       GEBCO bathymetry CSV\n"
                      << "  -o <file>       Output bathymetry file\n"
                      << "  --shallow <m>   Depth for lowlands (default -2500)\n"
                      << "  --deep <m>      Depth for mountains (default -6000)\n"
                      << "  --blur <px>     Coastal blur radius (default 20)\n"
                      << "  --no-detail     Skip procedural detail on former land\n";
            return 0;
        }
    }
    
    std::cout << "🌊 Earth Ocean World Generator\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "SRTM elevation: " << srtmFile << "\n";
    std::cout << "GEBCO bathymetry: " << gebcoFile << "\n";
    std::cout << "Output: " << outputFile << "\n\n";
    
    // Load CSV files
    std::vector<float> h_srtm, h_gebco;
    
    if (!loadCSV(srtmFile, h_srtm, WIDTH, HEIGHT)) {
        return EXIT_FAILURE;
    }
    
    if (!loadCSV(gebcoFile, h_gebco, WIDTH, HEIGHT)) {
        return EXIT_FAILURE;
    }
    
    std::cout << "\n";
    
    // Allocate device memory
    size_t totalPixels = WIDTH * HEIGHT;
    float *d_srtm, *d_gebco, *d_output, *d_temp;
    unsigned char *d_landMask, *d_coastMask;
    
    CUDA_CHECK(cudaMalloc(&d_srtm, totalPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gebco, totalPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, totalPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp, totalPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_landMask, totalPixels * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_coastMask, totalPixels * sizeof(unsigned char)));
    
    // Upload data
    std::cout << "📤 Uploading to GPU...\n";
    CUDA_CHECK(cudaMemcpy(d_srtm, h_srtm.data(), totalPixels * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gebco, h_gebco.data(), totalPixels * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    int threads = 256;
    int blocks = (totalPixels + threads - 1) / threads;
    
    // Step 1: Remap land to ocean depths
    std::cout << "🔄 Remapping land elevations to ocean depths...\n";
    remapLandKernel<<<blocks, threads>>>(d_srtm, d_gebco, d_output, d_landMask,
                                         params, WIDTH, HEIGHT);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Step 2: Detect coastline
    std::cout << "🏖️  Detecting old coastlines...\n";
    detectCoastlineKernel<<<blocks, threads>>>(d_landMask, d_coastMask,
                                               WIDTH, HEIGHT, params.coastalBlurRadius);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Step 3: Blur coastlines
    std::cout << "🌊 Smoothing coastal transitions...\n";
    coastalBlurKernel<<<blocks, threads>>>(d_output, d_coastMask, d_temp,
                                           WIDTH, HEIGHT, params.coastalBlurRadius);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Download results
    std::cout << "📥 Downloading results...\n";
    std::vector<float> h_output(totalPixels);
    std::vector<unsigned char> h_landMask(totalPixels);
    
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_temp, totalPixels * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_landMask.data(), d_landMask, totalPixels * sizeof(unsigned char),
                         cudaMemcpyDeviceToHost));
    
    // Cleanup GPU memory
    CUDA_CHECK(cudaFree(d_srtm));
    CUDA_CHECK(cudaFree(d_gebco));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_landMask));
    CUDA_CHECK(cudaFree(d_coastMask));
    
    // Statistics
    computeStatistics(h_output, h_landMask);
    
    // Save output
    if (!saveBathymetry(outputFile, h_output)) {
        return EXIT_FAILURE;
    }
    
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "✅ Earth Ocean World generation complete!\n";
    std::cout << "\nNext steps:\n";
    std::cout << "  1. Visualize: ./BathymetryViewer " << outputFile << "\n";
    std::cout << "  2. Use for gyre placement (real Atlantic/Pacific basins)\n";
    std::cout << "  3. Validate against Earth's actual gyres\n";
    
    return 0;
}
