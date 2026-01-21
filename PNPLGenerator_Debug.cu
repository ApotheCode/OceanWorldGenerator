//============================================================
// PNPLGenerator_Debug.cu
// PNPL with extensive debugging output
//============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cstring>

constexpr int WIDTH = 3600;
constexpr int HEIGHT = 1800;
constexpr int MONTHS = 12;
constexpr int VARIABLES = 14;
constexpr double M_PI = 3.14159265358979323846;
constexpr double DEG_TO_RAD = M_PI / 180.0;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

struct PNPLParams {
    double intensity;
    double rotationPeriod;
    bool respectLandBoundaries;
};

//============================================================
// Device Functions
//============================================================
__device__ __forceinline__ double coriolisParameter(double lat, double rotationPeriod)
{
    double omega = 2.0 * M_PI / (rotationPeriod * 3600.0);
    return 2.0 * omega * sin(lat);
}

__device__ double planetaryWavePattern(double lon, double lat, int month, 
                                       double rotationPeriod)
{
    double seasonalPhase = (month * 2.0 * M_PI) / 12.0;
    double f = coriolisParameter(lat, rotationPeriod);
    double coriolisScale = fabs(f) * 1e4;
    
    double n = 0.0;
    n += 0.40 * sin(2.0 * lon + 1.3 * sin(3.0 * lat) + seasonalPhase * 0.2);
    n += 0.30 * sin(5.0 * lon - 2.2 * cos(2.0 * lat) + coriolisScale * 0.5);
    n += 0.20 * cos(8.0 * lon + 1.8 * sin(4.0 * lat) - seasonalPhase * 0.3);
    
    double jetStrength = exp(-pow((fabs(lat * 180.0/M_PI) - 50.0)/15.0, 2));
    n += 0.10 * jetStrength * sin(6.0 * lon + 2.5 * cos(lat) + seasonalPhase);
    
    return n;
}

__device__ double gyreConstraint(int x, int y, const unsigned char* __restrict__ isLand,
                                int width, int height, int searchRadius)
{
    if (isLand == nullptr) return 1.0;
    
    int idx = y * width + x;
    if (isLand[idx] == 1) return 0.0;
    
    for (int dy = -searchRadius; dy <= searchRadius; dy++) {
        for (int dx = -searchRadius; dx <= searchRadius; dx++) {
            int nx = (x + dx + width) % width;
            int ny = y + dy;
            
            if (ny < 0 || ny >= height) continue;
            
            int nIdx = ny * width + nx;
            if (isLand[nIdx] == 1) {
                double dist = sqrt((double)(dx*dx + dy*dy));
                double attenuation = dist / searchRadius;
                return fmin(1.0, attenuation);
            }
        }
    }
    
    return 1.0;
}

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
// Kernel with Debug Counters
//============================================================
__global__ void applyPNPLKernel(double* __restrict__ data,
                                const unsigned char* __restrict__ isLand,
                                int width, int height,
                                int varIdx,
                                PNPLParams params,
                                int* __restrict__ debugCounters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    
    int x = idx % width;
    int y = idx / width;
    
    double lat = (90.0 - (y + 0.5) * 180.0 / height) * DEG_TO_RAD;
    double lon = ((x + 0.5) * 360.0 / width) * DEG_TO_RAD;
    
    double boundaryFactor = 1.0;
    if (params.respectLandBoundaries && isLand != nullptr) {
        boundaryFactor = gyreConstraint(x, y, isLand, width, height, 10);
    }
    
    // Process all months for this pixel
    for (int month = 0; month < MONTHS; ++month) {
        double wavePattern = planetaryWavePattern(lon, lat, month, params.rotationPeriod);
        double latAmp = 0.5 + 0.5 * pow(fabs(sin(lat)), 1.2);
        double perturbation = params.intensity * latAmp * boundaryFactor * wavePattern;
        
        double value = getClimateValue(data, x, y, varIdx, month, width, height);
        
        if (isfinite(value)) {
            double oldValue = value;
            
            // Apply perturbation based on variable type
            if (varIdx == 1) {
                value += perturbation * 1.5;  // INCREASED from to 1.5
            }            
            
            value = fmax(0.0, fmin(1.0, value));
            setClimateValue(data, x, y, varIdx, month, width, height, value);
            
            // Debug: Count significant changes
            if (fabs(value - oldValue) > 0.01) {
                atomicAdd(&debugCounters[0], 1);  // Total changes
            }
            if (fabs(value - oldValue) > 0.05) {
                atomicAdd(&debugCounters[1], 1);  // Large changes
            }
            if (isLand && isLand[idx] == 0) {
                atomicAdd(&debugCounters[2], 1);  // Ocean pixel changes
            }
        }
    }
}

//============================================================
// Host Functions
//============================================================
void applyPNPLToClimate(std::vector<double>& climateData,
                        const unsigned char* d_landMask,
                        const PNPLParams& params)
{
    std::cout << "🌀 Applying PNPL with DEBUG...\n";
    std::cout << "   Intensity: " << params.intensity << "\n";
    std::cout << "   Rotation: " << params.rotationPeriod << " hrs\n\n";
    
    size_t totalSize = climateData.size();
    
    double* d_climateData;
    CUDA_CHECK(cudaMalloc(&d_climateData, totalSize * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_climateData, climateData.data(),
                         totalSize * sizeof(double), cudaMemcpyHostToDevice));
    
    int* d_debugCounters;
    CUDA_CHECK(cudaMalloc(&d_debugCounters, 3 * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_debugCounters, 0, 3 * sizeof(int)));
    
    int threads = 256;
    int blocks = (WIDTH * HEIGHT + threads - 1) / threads;
    
    int variablesToPerturb[] = {1};
    int numVars = sizeof(variablesToPerturb) / sizeof(int);
    const char* varNames[] = {
        "GPM", "SST", "LSTD", "LSTN", "ALBEDO", 
        "NETFLUX", "SWFLUX", "LWFLUX", "INSOL", 
        "LAI", "NISE", "SNOW", "NDVI", "AIRTEMP"
    };
    
    for (int i = 0; i < numVars; ++i) {
        int varIdx = variablesToPerturb[i];
        
        std::cout << "   Processing var " << varIdx << " (" << varNames[varIdx] << ")...\n";
        
        // Reset counters
        CUDA_CHECK(cudaMemset(d_debugCounters, 0, 3 * sizeof(int)));
        
        applyPNPLKernel<<<blocks, threads>>>(d_climateData, d_landMask,
                                             WIDTH, HEIGHT, varIdx, params,
                                             d_debugCounters);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Read debug counters
        int counters[3];
        CUDA_CHECK(cudaMemcpy(counters, d_debugCounters, 3 * sizeof(int), 
                             cudaMemcpyDeviceToHost));
        
        std::cout << "      ✓ Total changes >1%:  " << counters[0] << "\n";
        std::cout << "      ✓ Large changes >5%:  " << counters[1] << "\n";
        std::cout << "      ✓ Ocean pixels affected: " << counters[2] << "\n";
        
        if (counters[0] == 0) {
            std::cerr << "      ⚠️ WARNING: No changes detected!\n";
        }
    }
    
    CUDA_CHECK(cudaMemcpy(climateData.data(), d_climateData,
                         totalSize * sizeof(double), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_climateData));
    CUDA_CHECK(cudaFree(d_debugCounters));
    
    std::cout << "\n   ✅ PNPL complete\n\n";
}

unsigned char* loadLandMask(const std::string& terrainFile)
{
    std::ifstream file(terrainFile, std::ios::binary);
    if (!file) {
        std::cerr << "⚠️ No terrain file\n";
        return nullptr;
    }
    
    int width, height;
    file.read(reinterpret_cast<char*>(&width), sizeof(int));
    file.read(reinterpret_cast<char*>(&height), sizeof(int));
    
    if (width != WIDTH || height != HEIGHT) {
        std::cerr << "⚠️ Dimension mismatch\n";
        file.close();
        return nullptr;
    }
    
    file.seekg(sizeof(int) * 2 + WIDTH * HEIGHT * sizeof(float), std::ios::beg);
    
    std::vector<unsigned char> h_landMask(WIDTH * HEIGHT);
    file.read(reinterpret_cast<char*>(h_landMask.data()), WIDTH * HEIGHT);
    file.close();
    
    unsigned char* d_landMask;
    CUDA_CHECK(cudaMalloc(&d_landMask, WIDTH * HEIGHT * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_landMask, h_landMask.data(),
                         WIDTH * HEIGHT * sizeof(unsigned char),
                         cudaMemcpyHostToDevice));
    
    std::cout << "✅ Land mask loaded\n";
    return d_landMask;
}

bool validatePNPL(const std::vector<double>& data)
{
    std::cout << "🔍 Validating output...\n";
    
    size_t outOfBounds = 0, nanCount = 0;
    double minVal = 1e9, maxVal = -1e9;
    
    for (double val : data) {
        if (!isfinite(val)) nanCount++;
        else {
            if (val < 0.0 || val > 1.0) outOfBounds++;
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        }
    }
    
    std::cout << "   Range: [" << minVal << ", " << maxVal << "]\n";
    std::cout << "   Out of bounds: " << outOfBounds << "\n";
    std::cout << "   NaN: " << nanCount << "\n";
    
    if (outOfBounds > 0 || nanCount > data.size() * 0.01) {
        std::cerr << "   ⚠️ Validation issues detected\n";
        return false;
    }
    
    std::cout << "   ✅ Validation passed\n\n";
    return true;
}

//============================================================
// Main
//============================================================
int main(int argc, char** argv)
{
    std::string climateFile = "output/Earth_Climate.bin";
    std::string terrainFile = "output/TerrainData.bin";
    std::string outputFile = "output/Earth_Climate_PNPL_Debug.bin";
    
    double intensity = 0.25;
    double rotationPeriod = 24.0;
    bool useBoundaries = true;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0 && i+1 < argc)
            climateFile = argv[++i];
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc)
            terrainFile = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc)
            outputFile = argv[++i];
        else if (strcmp(argv[i], "--intensity") == 0 && i+1 < argc)
            intensity = atof(argv[++i]);
        else if (strcmp(argv[i], "--rotation") == 0 && i+1 < argc)
            rotationPeriod = atof(argv[++i]);
        else if (strcmp(argv[i], "--no-boundaries") == 0)
            useBoundaries = false;
    }
    
    std::cout << "🌀 PNPL Generator (DEBUG MODE)\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "Input:  " << climateFile << "\n";
    std::cout << "Terrain: " << terrainFile << "\n";
    std::cout << "Output: " << outputFile << "\n\n";
    
    // Load climate
    std::ifstream climFile(climateFile, std::ios::binary);
    if (!climFile) {
        std::cerr << "❌ Cannot open climate file\n";
        return EXIT_FAILURE;
    }
    
    int depth, height, width;
    climFile.read(reinterpret_cast<char*>(&depth), sizeof(int));
    climFile.read(reinterpret_cast<char*>(&height), sizeof(int));
    climFile.read(reinterpret_cast<char*>(&width), sizeof(int));
    
    size_t totalSize = width * height * VARIABLES * MONTHS;
    std::vector<double> climateData(totalSize);
    climFile.read(reinterpret_cast<char*>(climateData.data()),
                 totalSize * sizeof(double));
    climFile.close();
    
    std::cout << "📦 Loaded " << (totalSize * sizeof(double) / (1024.0*1024.0*1024.0)) 
              << " GB\n\n";
    
    // Load land mask
    unsigned char* d_landMask = nullptr;
    if (useBoundaries) {
        d_landMask = loadLandMask(terrainFile);
    }
    
    // Setup params
    PNPLParams params;
    params.intensity = intensity;
    params.rotationPeriod = rotationPeriod;
    params.respectLandBoundaries = (d_landMask != nullptr) && useBoundaries;
    
    // Apply PNPL
    applyPNPLToClimate(climateData, d_landMask, params);
    
    // Validate
    validatePNPL(climateData);
    
    // Save
    std::ofstream outFile(outputFile, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(&depth), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&height), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&width), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(climateData.data()),
                 totalSize * sizeof(double));
    outFile.close();
    
    if (d_landMask) CUDA_CHECK(cudaFree(d_landMask));
    
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "✅ Output: " << outputFile << "\n";
    std::cout << "   Check debug counters above to verify PNPL execution\n";
    
    return 0;
}