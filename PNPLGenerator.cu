//============================================================
// PNPLGenerator.cu
// Planetary Noise Perturbation Layer - Adds circulation dynamics
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
    double intensity;              // 0.03-0.10 typical
    double rotationPeriod;         // hours (affects Coriolis strength)
    bool respectLandBoundaries;    // Gyres deflect at coasts
};

//============================================================
// Device: Calculate Coriolis Parameter
//============================================================
__device__ __forceinline__ double coriolisParameter(double lat, double rotationPeriod)
{
    // f = 2 * Ω * sin(φ)
    // where Ω = 2π / T (rad/s)
    double omega = 2.0 * M_PI / (rotationPeriod * 3600.0);
    return 2.0 * omega * sin(lat);
}

//============================================================
// Device: Multi-Scale Planetary Wave Pattern
//============================================================
__device__ double planetaryWavePattern(double lon, double lat, int month, 
                                       double rotationPeriod)
{
    // Seasonal phase shift
    double seasonalPhase = (month * 2.0 * M_PI) / 12.0;
    
    // Calculate local Coriolis strength
    double f = coriolisParameter(lat, rotationPeriod);
    double coriolisScale = fabs(f) * 1e4;  // Normalize
    
    // Multi-scale wave components
    double n = 0.0;
    
    // 1. Planetary-scale Rossby waves (10,000+ km wavelength)
    n += 0.40 * sin(2.0 * lon + 1.3 * sin(3.0 * lat) + seasonalPhase * 0.2);
    
    // 2. Basin-scale gyres (3000-5000 km)
    n += 0.30 * sin(5.0 * lon - 2.2 * cos(2.0 * lat) + coriolisScale * 0.5);
    
    // 3. Mesoscale eddies (500-1000 km)
    n += 0.20 * cos(8.0 * lon + 1.8 * sin(4.0 * lat) - seasonalPhase * 0.3);
    
    // 4. Jet stream meanders (strongest 40-60° latitudes)
    double jetStrength = exp(-pow((fabs(lat * 180.0/M_PI) - 50.0)/15.0, 2));
    n += 0.10 * jetStrength * sin(6.0 * lon + 2.5 * cos(lat) + seasonalPhase);
    
    return n;
}

//============================================================
// Device: Gyre Boundary Constraint
//============================================================
__device__ double gyreConstraint(int x, int y, const unsigned char* __restrict__ isLand,
                                int width, int height, int searchRadius)
{
    if (isLand == nullptr) return 1.0;
    
    int idx = y * width + x;
    if (isLand[idx] == 1) return 0.0;  // On land
    
    // Check if near coast
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
    
    return 1.0;  // Open ocean
}

//============================================================
// Device: Get/Set Climate Value [cell][var][month]
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
// Kernel: Apply PNPL to Climate Data
//============================================================
__global__ void applyPNPLKernel(double* __restrict__ data,
                                const unsigned char* __restrict__ isLand,
                                int width, int height,
                                int varIdx,
                                PNPLParams params)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    
    int x = idx % width;
    int y = idx / width;
    
    // Geographic coordinates
    double lat = (90.0 - (y + 0.5) * 180.0 / height) * DEG_TO_RAD;
    double lon = ((x + 0.5) * 360.0 / width) * DEG_TO_RAD;
    
    // Gyre boundary constraint
    double boundaryFactor = 1.0;
    if (params.respectLandBoundaries && isLand != nullptr) {
        boundaryFactor = gyreConstraint(x, y, isLand, width, height, 10);
    }
    
    // Apply to all months
    for (int month = 0; month < MONTHS; ++month) {
        // Generate wave pattern for this month
        double wavePattern = planetaryWavePattern(lon, lat, month, params.rotationPeriod);
        
        // Latitude-based amplitude (stronger at poles)
        double latAmp = 0.5 + 0.5 * pow(fabs(sin(lat)), 1.2);
        
        // Calculate perturbation
        double perturbation = params.intensity * latAmp * boundaryFactor * wavePattern;
        
        // Get current value
        double value = getClimateValue(data, x, y, varIdx, month, width, height);
        
        if (isfinite(value)) {
            // Variable-specific perturbation
            // Temperature variables (1,2,3,13): additive
            if (varIdx == 1 || varIdx == 2 || varIdx == 3 || varIdx == 13) {
                value += perturbation * 0.50;  // ±5% of range
            }
            // Flux variables (5,6,7,8): multiplicative
            else if (varIdx >= 5 && varIdx <= 8) {
                value *= (1.0 + perturbation * 0.15);
            }
            // Others: gentle multiplicative
            else {
                value *= (1.0 + perturbation * 0.05);
            }
            
            // Clamp to [0,1]
            value = fmax(0.0, fmin(1.0, value));
            setClimateValue(data, x, y, varIdx, month, width, height, value);
        }
    }
}

//============================================================
// Host: Apply PNPL to All Variables
//============================================================
void applyPNPLToClimate(std::vector<double>& climateData,
                        const unsigned char* d_landMask,
                        const PNPLParams& params)
{
    std::cout << "🌀 Applying PNPL (Planetary Noise Perturbation Layer)...\n";
    std::cout << "   Intensity: " << params.intensity << "\n";
    std::cout << "   Rotation period: " << params.rotationPeriod << " hours\n";
    std::cout << "   Land boundaries: " << (params.respectLandBoundaries ? "Yes" : "No") << "\n";
    
    size_t totalSize = climateData.size();
    
    // Upload climate data to GPU
    double* d_climateData;
    CUDA_CHECK(cudaMalloc(&d_climateData, totalSize * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_climateData, climateData.data(),
                         totalSize * sizeof(double), cudaMemcpyHostToDevice));
    
    // Launch configuration
    int threads = 256;
    int blocks = (WIDTH * HEIGHT + threads - 1) / threads;
    
    // Apply PNPL SST
    int variablesToPerturb[] = {1};  // SST only
    int numVars = sizeof(variablesToPerturb) / sizeof(int);
    
    for (int i = 0; i < numVars; ++i) {
        int varIdx = variablesToPerturb[i];
        
        const char* varNames[] = {
            "GPM_3IMERGM", "MYD28M", "MOD_LSTD_M", "MOD_LSTN_M",
            "MCD43C3_M_BSA", "CERES_NETFLUX_M", "CERES_SWFLUX_M",
            "CERES_LWFLUX_M", "CERES_INSOL_M", "MOD15A2_M_LAI",
            "NISE_D", "MOD10C1_M_SNOW", "MOD_NDVI_M", "AirTemp"
        };
        
        std::cout << "   Perturbing var " << varIdx << " (" << varNames[varIdx] << ")...\n";
        
        applyPNPLKernel<<<blocks, threads>>>(d_climateData, d_landMask,
                                             WIDTH, HEIGHT, varIdx, params);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Download results
    CUDA_CHECK(cudaMemcpy(climateData.data(), d_climateData,
                         totalSize * sizeof(double), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_climateData));
    
    std::cout << "   ✅ PNPL applied to " << numVars << " variables\n\n";
}

//============================================================
// Host: Load Land Mask
//============================================================
unsigned char* loadLandMask(const std::string& terrainFile)
{
    std::ifstream file(terrainFile, std::ios::binary);
    if (!file) {
        std::cerr << "⚠️ No terrain file found, PNPL will ignore land boundaries\n";
        return nullptr;
    }
    
    int width, height;
    file.read(reinterpret_cast<char*>(&width), sizeof(int));
    file.read(reinterpret_cast<char*>(&height), sizeof(int));
    
    if (width != WIDTH || height != HEIGHT) {
        std::cerr << "⚠️ Terrain size mismatch\n";
        file.close();
        return nullptr;
    }
    
    size_t totalPixels = WIDTH * HEIGHT;
    
    // Skip elevation data
    file.seekg(sizeof(int) * 2 + totalPixels * sizeof(float), std::ios::beg);
    
    // Read land mask
    std::vector<unsigned char> h_landMask(totalPixels);
    file.read(reinterpret_cast<char*>(h_landMask.data()), totalPixels);
    file.close();
    
    // Upload to GPU
    unsigned char* d_landMask;
    CUDA_CHECK(cudaMalloc(&d_landMask, totalPixels * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_landMask, h_landMask.data(),
                         totalPixels * sizeof(unsigned char),
                         cudaMemcpyHostToDevice));
    
    std::cout << "✅ Land mask loaded for boundary constraints\n";
    return d_landMask;
}

//============================================================
// Host: Validate Post-PNPL
//============================================================
bool validatePNPL(const std::vector<double>& data)
{
    std::cout << "🔍 Validating PNPL output...\n";
    
    size_t outOfBounds = 0;
    size_t nanCount = 0;
    
    for (double val : data) {
        if (!isfinite(val)) nanCount++;
        else if (val < 0.0 || val > 1.0) outOfBounds++;
    }
    
    double oobPct = (outOfBounds * 100.0) / data.size();
    double nanPct = (nanCount * 100.0) / data.size();
    
    std::cout << "   Out of bounds: " << outOfBounds << " (" << oobPct << "%)\n";
    std::cout << "   NaN values: " << nanCount << " (" << nanPct << "%)\n";
    
    if (oobPct > 1.0) {
        std::cerr << "⚠️ WARNING: High out-of-bounds rate!\n";
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
    std::string climateFile = "Earth_Climate.bin";
    std::string terrainFile = "TerrainData.bin";
    std::string outputFile = "Earth_Climate_PNPL.bin";
    
    double intensity = 0.05;
    double rotationPeriod = 24.0;
    bool useBoundaries = true;
    
    // Parse arguments
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
        else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -i <file>       Input climate file\n"
                      << "  -t <file>       Terrain data file\n"
                      << "  -o <file>       Output file\n"
                      << "  --intensity <val> PNPL intensity (0.03-0.10)\n"
                      << "  --rotation <hrs>  Rotation period in hours\n"
                      << "  --no-boundaries   Ignore land boundaries\n";
            return 0;
        }
    }
    
    std::cout << "🌀 PNPL Generator\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "Input:  " << climateFile << "\n";
    std::cout << "Terrain: " << terrainFile << "\n";
    std::cout << "Output: " << outputFile << "\n\n";
    
    // Load climate data
    std::ifstream climFile(climateFile, std::ios::binary);
    if (!climFile) {
        std::cerr << "❌ Failed to open climate file\n";
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
    
    std::cout << "📦 Loaded climate data: "
              << (totalSize * sizeof(double) / (1024.0*1024.0*1024.0)) << " GB\n\n";
    
    // Load land mask
    unsigned char* d_landMask = nullptr;
    if (useBoundaries) {
        d_landMask = loadLandMask(terrainFile);
    }
    
    // Setup PNPL parameters
    PNPLParams params;
    params.intensity = intensity;
    params.rotationPeriod = rotationPeriod;
    params.respectLandBoundaries = (d_landMask != nullptr) && useBoundaries;
    
    // Apply PNPL
    applyPNPLToClimate(climateData, d_landMask, params);
    
    // Validate
    if (!validatePNPL(climateData)) {
        std::cerr << "⚠️ Validation warnings detected\n";
    }
    
    // Save output
    std::ofstream outFile(outputFile, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(&depth), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&height), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&width), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(climateData.data()),
                 totalSize * sizeof(double));
    outFile.close();
    
    // Cleanup
    if (d_landMask) CUDA_CHECK(cudaFree(d_landMask));
    
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "✅ PNPL generation complete!\n";
    std::cout << "   Output: " << outputFile << "\n";
    
    return 0;
}