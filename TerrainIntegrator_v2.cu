//============================================================
// TerrainIntegrator.cu
// Modifies climate data based on terrain elevation and geography
// Author: Mark Devereux (2025-10-18)
//============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdint>
#include "ClimateFileFormat.h"

constexpr int WIDTH = STANDARD_WIDTH;
constexpr int HEIGHT = STANDARD_HEIGHT;
constexpr int MONTHS = STANDARD_MONTHS;
constexpr int VARIABLES = STANDARD_VARIABLES;
constexpr double M_PI = 3.14159265358979323846;
constexpr double DEG_TO_RAD = M_PI / 180.0;

// Physical constants
constexpr double TEMP_LAPSE_RATE = 0.0065;      // °C per meter (troposphere)
constexpr double PRECIP_OROGRAPHIC_GAIN = 0.15; // 15% increase per 1000m windward
constexpr double PRECIP_RAINSHADOW_LOSS = 0.40; // 40% decrease leeward
constexpr double CONTINENTALITY_FACTOR = 0.08;  // Temperature range increase inland

// Variable indices
constexpr int VAR_RAINFALL   = 0;
constexpr int VAR_SST        = 1;
constexpr int VAR_LSTD       = 2;  // Land Surface Temp Day
constexpr int VAR_LSTN       = 3;  // Land Surface Temp Night
constexpr int VAR_ALBEDO     = 4;
constexpr int VAR_NETFLUX    = 5;
constexpr int VAR_SWFLUX     = 6;
constexpr int VAR_LWFLUX     = 7;
constexpr int VAR_INSOL      = 8;
constexpr int VAR_LAI        = 9;
constexpr int VAR_NISE       = 10;
constexpr int VAR_SNOW       = 11;
constexpr int VAR_NDVI       = 12;
constexpr int VAR_AIRTEMP    = 13;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

struct TerrainData {
    float* elevation;
    unsigned char* isLand;
    float* oceanDepth;
    float* continentality;
};

//============================================================
// Device: Get Climate Variable at [cell][var][month]
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
// Device: Denormalize Temperature [0,1] → °C
//============================================================
__device__ __forceinline__ double denormTemp(double normVal, double minC, double maxC)
{
    return normVal * (maxC - minC) + minC;
}

__device__ __forceinline__ double normTemp(double celsius, double minC, double maxC)
{
    return fmax(0.0, fmin(1.0, (celsius - minC) / (maxC - minC)));
}

//============================================================
// Device: Calculate Prevailing Wind Direction (Latitude-Based)
//============================================================
__device__ int getPrevailingWindDirection(double lat)
{
    // Simplified wind bands:
    // 0-30°: Trade winds (Easterlies) → wind from EAST
    // 30-60°: Westerlies → wind from WEST
    // 60-90°: Polar Easterlies → wind from EAST
    
    double absLat = fabs(lat * 180.0 / M_PI);
    
    if (absLat < 30.0) return 1;       // East (positive dx)
    else if (absLat < 60.0) return -1; // West (negative dx)
    else return 1;                     // East
}

//============================================================
// Kernel: Apply Elevation-Based Temperature Modification
//============================================================
__global__ void applyElevationTempKernel(double* __restrict__ climateData,
                                         const float* __restrict__ elevation,
                                         const unsigned char* __restrict__ isLand,
                                         int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    
    int x = idx % width;
    int y = idx / width;
    
    // Only modify land pixels
    if (isLand[idx] == 0) return;
    
    float elev = elevation[idx];
    if (elev <= 0.0f) return;  // Sea level or below
    
    // Temperature lapse rate: -6.5°C per 1000m
    double tempDrop = (elev / 1000.0) * TEMP_LAPSE_RATE;
    
    // Apply to all temperature variables for all months
    for (int month = 0; month < MONTHS; ++month) {
        // LSTD (Land Surface Temp Day): -50 to +60°C range
        double lstd = getClimateValue(climateData, x, y, VAR_LSTD, month, width, height);
        if (isfinite(lstd)) {
            double lstdC = denormTemp(lstd, -50.0, 60.0);
            lstdC -= tempDrop;
            lstd = normTemp(lstdC, -50.0, 60.0);
            setClimateValue(climateData, x, y, VAR_LSTD, month, width, height, lstd);
        }
        
        // LSTN (Land Surface Temp Night): -50 to +60°C range
        double lstn = getClimateValue(climateData, x, y, VAR_LSTN, month, width, height);
        if (isfinite(lstn)) {
            double lstnC = denormTemp(lstn, -50.0, 60.0);
            lstnC -= tempDrop;
            lstn = normTemp(lstnC, -50.0, 60.0);
            setClimateValue(climateData, x, y, VAR_LSTN, month, width, height, lstn);
        }
        
        // AirTemp: -4 to +33°C range
        double airTemp = getClimateValue(climateData, x, y, VAR_AIRTEMP, month, width, height);
        if (isfinite(airTemp)) {
            double airTempC = denormTemp(airTemp, -4.0, 33.0);
            airTempC -= tempDrop;
            airTemp = normTemp(airTempC, -4.0, 33.0);
            setClimateValue(climateData, x, y, VAR_AIRTEMP, month, width, height, airTemp);
        }
    }
}

//============================================================
// Kernel: Apply Orographic Precipitation
//============================================================
__global__ void applyOrographicPrecipKernel(double* __restrict__ climateData,
                                            const float* __restrict__ elevation,
                                            const unsigned char* __restrict__ isLand,
                                            int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    if (isLand[idx] == 0) return;  // Ocean only
    
    int x = idx % width;
    int y = idx / width;
    
    float elev = elevation[idx];
    if (elev <= 0.0f) return;
    
    // Calculate latitude for wind direction
    double lat = (90.0 - (y + 0.5) * 180.0 / height) * DEG_TO_RAD;
    int windDir = getPrevailingWindDirection(lat);
    
    // Check upwind elevation (is this a windward or leeward slope?)
    int upwindX = (x - windDir + width) % width;
    float upwindElev = elevation[y * width + upwindX];
    
    bool isWindward = (elev > upwindElev);  // Rising terrain in wind direction
    
    // Elevation effect on precipitation
    double elevFactor;
    if (isWindward) {
        // Windward: orographic lift increases precipitation
        elevFactor = 1.0 + (elev / 1000.0) * PRECIP_OROGRAPHIC_GAIN;
    } else {
        // Leeward: rain shadow decreases precipitation
        double shadowIntensity = fmin(1.0, (elev - upwindElev) / 2000.0);
        elevFactor = 1.0 - shadowIntensity * PRECIP_RAINSHADOW_LOSS;
    }
    
    elevFactor = fmax(0.1, fmin(3.0, elevFactor));  // Clamp to reasonable range
    
    // Apply to all months
    for (int month = 0; month < MONTHS; ++month) {
        double precip = getClimateValue(climateData, x, y, VAR_RAINFALL, month, width, height);
        if (isfinite(precip)) {
            precip *= elevFactor;
            precip = fmax(0.0, fmin(1.0, precip));
            setClimateValue(climateData, x, y, VAR_RAINFALL, month, width, height, precip);
        }
    }
}

//============================================================
// Kernel: Apply Continentality (Distance to Coast Effect)
//============================================================
__global__ void applyContinentalityKernel(double* __restrict__ climateData,
                                          const float* __restrict__ continentality,
                                          const unsigned char* __restrict__ isLand,
                                          int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    if (isLand[idx] == 0) return;  // Only land
    
    int x = idx % width;
    int y = idx / width;
    
    float distToCoast = continentality[idx];  // km
    
    // Continental effect: increase temperature range inland
    // Max effect at ~2000 km inland
    double continentalFactor = fmin(1.0, distToCoast / 2000.0);
    double tempRangeIncrease = continentalFactor * CONTINENTALITY_FACTOR;
    
    // Calculate annual mean and range for this location
    double sumTemp = 0.0;
    int validMonths = 0;
    
    for (int m = 0; m < MONTHS; ++m) {
        double temp = getClimateValue(climateData, x, y, VAR_AIRTEMP, m, width, height);
        if (isfinite(temp)) {
            sumTemp += temp;
            validMonths++;
        }
    }
    
    if (validMonths == 0) return;
    
    double meanTemp = sumTemp / validMonths;
    
    // Amplify seasonal variation around mean
    for (int month = 0; month < MONTHS; ++month) {
        double temp = getClimateValue(climateData, x, y, VAR_AIRTEMP, month, width, height);
        if (isfinite(temp)) {
            double deviation = temp - meanTemp;
            double newTemp = meanTemp + deviation * (1.0 + tempRangeIncrease);
            newTemp = fmax(0.0, fmin(1.0, newTemp));
            setClimateValue(climateData, x, y, VAR_AIRTEMP, month, width, height, newTemp);
        }
    }
    
    // Also reduce precipitation slightly inland (continental drying)
    double precipReduction = 1.0 - continentalFactor * 0.15;  // Up to 15% reduction
    
    for (int month = 0; month < MONTHS; ++month) {
        double precip = getClimateValue(climateData, x, y, VAR_RAINFALL, month, width, height);
        if (isfinite(precip)) {
            precip *= precipReduction;
            setClimateValue(climateData, x, y, VAR_RAINFALL, month, width, height, precip);
        }
    }
}

//============================================================
// Kernel: Adjust Ocean Temperature by Depth
//============================================================
__global__ void applyOceanDepthEffectKernel(double* __restrict__ climateData,
                                            const float* __restrict__ oceanDepth,
                                            const unsigned char* __restrict__ isLand,
                                            int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    if (isLand[idx] == 1) return;  // Only ocean
    
    int x = idx % width;
    int y = idx / width;
    
    float depth = oceanDepth[idx];  // meters
    
    // Shallow water (<200m): Higher temperature variability, warms faster
    // Deep water (>4000m): More stable, cooler on average
    
    double depthFactor;
    if (depth < 200.0f) {
        // Shallow: amplify seasonal variation slightly (continental shelf effect)
        depthFactor = 1.05;
    } else if (depth > 4000.0f) {
        // Deep: dampen variation, slight cooling
        depthFactor = 0.95;
    } else {
        // Mid-depth: proportional interpolation
        depthFactor = 1.05 - (depth - 200.0f) / 3800.0f * 0.10;
    }
    
    // Calculate mean SST
    double sumSST = 0.0;
    int validMonths = 0;
    
    for (int m = 0; m < MONTHS; ++m) {
        double sst = getClimateValue(climateData, x, y, VAR_SST, m, width, height);
        if (isfinite(sst)) {
            sumSST += sst;
            validMonths++;
        }
    }
    
    if (validMonths == 0) return;
    
    double meanSST = sumSST / validMonths;
    
    // Modulate seasonal variation
    for (int month = 0; month < MONTHS; ++month) {
        double sst = getClimateValue(climateData, x, y, VAR_SST, month, width, height);
        if (isfinite(sst)) {
            double deviation = sst - meanSST;
            double newSST = meanSST + deviation * depthFactor;
            newSST = fmax(0.0, fmin(1.0, newSST));
            setClimateValue(climateData, x, y, VAR_SST, month, width, height, newSST);
        }
    }
}

//============================================================
// Host: Load Terrain Data
//============================================================
TerrainData loadTerrainData(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Failed to open terrain file: " << filename << "\n";
        exit(EXIT_FAILURE);
    }
    
    // Read 28-byte standard header
    ClimateFileHeader header;
    if (!readClimateHeader(file, header)) {
        std::cerr << "❌ Failed to read terrain header\n";
        exit(EXIT_FAILURE);
    }
    
    int width = header.width;
    int height = header.height;
    
    if (width != WIDTH || height != HEIGHT) {
        std::cerr << "❌ Terrain dimensions mismatch! Expected " << WIDTH << "x" << HEIGHT 
                  << ", got " << width << "x" << height << "\n";
        exit(EXIT_FAILURE);
    }
    
    TerrainData terrain;
    size_t totalPixels = WIDTH * HEIGHT;
    
    CUDA_CHECK(cudaMalloc(&terrain.elevation, totalPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&terrain.isLand, totalPixels * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&terrain.oceanDepth, totalPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&terrain.continentality, totalPixels * sizeof(float)));
    
    std::vector<float> h_elevation(totalPixels);
    std::vector<unsigned char> h_isLand(totalPixels);
    std::vector<float> h_oceanDepth(totalPixels, 0.0f);  // Default to 0
    std::vector<float> h_continentality(totalPixels, 0.0f);  // Default to 0
    
    // Read elevation and land mask (only 2 arrays that exist in file)
    file.read(reinterpret_cast<char*>(h_elevation.data()), totalPixels * sizeof(float));
    file.read(reinterpret_cast<char*>(h_isLand.data()), totalPixels * sizeof(unsigned char));
    
    // Compute oceanDepth and continentality from elevation/landMask
    // oceanDepth: negative elevation for ocean pixels
    // continentality: distance from coast (simplified: 0 for now)
    for (size_t i = 0; i < totalPixels; ++i) {
        if (h_isLand[i] == 0) {
            // Ocean pixel - depth is negative elevation
            h_oceanDepth[i] = h_elevation[i] < 0 ? -h_elevation[i] : 0.0f;
        }
        // continentality stays 0 for now (could compute distance from coast)
    }
    
    CUDA_CHECK(cudaMemcpy(terrain.elevation, h_elevation.data(), 
                         totalPixels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(terrain.isLand, h_isLand.data(), 
                         totalPixels * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(terrain.oceanDepth, h_oceanDepth.data(), 
                         totalPixels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(terrain.continentality, h_continentality.data(), 
                         totalPixels * sizeof(float), cudaMemcpyHostToDevice));
    
    file.close();
    
    std::cout << "✅ Terrain data loaded (elevation + landMask)\n";
    return terrain;
}

//============================================================
// Host: Validate Energy Balance
//============================================================
bool validateClimateData(const std::vector<double>& data, const std::string& stage)
{
    std::cout << "🔍 Validating climate data (" << stage << ")...\n";
    
    size_t outOfBounds = 0;
    size_t nanCount = 0;
    double minVal = 1e9, maxVal = -1e9;
    
    for (double val : data) {
        if (!isfinite(val)) {
            nanCount++;
        } else {
            if (val < 0.0 || val > 1.0) outOfBounds++;
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        }
    }
    
    std::cout << "   Range: [" << minVal << ", " << maxVal << "]\n";
    std::cout << "   Out of bounds: " << outOfBounds << " (" 
              << (outOfBounds * 100.0 / data.size()) << "%)\n";
    std::cout << "   NaN values: " << nanCount << " ("
              << (nanCount * 100.0 / data.size()) << "%)\n";
    
    if (outOfBounds > data.size() * 0.01) {  // >1% out of bounds
        std::cerr << "⚠️ WARNING: High out-of-bounds percentage!\n";
        return false;
    }
    
    std::cout << "   ✅ Validation passed\n\n";
    return true;
}

//============================================================
// Main Integration Function
//============================================================
int main(int argc, char** argv)
{
    std::string climateFile = "output/OceanWorld_Climate.bin";
    std::string terrainFile = "output/TerrainData.bin";
    std::string outputFile = "output/Earth_Climate.bin";
    
    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            climateFile = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            terrainFile = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            outputFile = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  -i <file>  Input climate file (default: output/OceanWorld_Climate.bin)\n";
            std::cout << "  -t <file>  Input terrain file (default: output/TerrainData.bin)\n";
            std::cout << "  -o <file>  Output climate file (default: output/Earth_Climate.bin)\n";
            std::cout << "  -h, --help Show this help message\n";
            return 0;
        }
    }
    
    std::cout << "🏔️ Terrain Integrator\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "Climate: " << climateFile << "\n";
    std::cout << "Terrain: " << terrainFile << "\n";
    std::cout << "Output:  " << outputFile << "\n\n";
    
    // Load climate data
    std::vector<double> h_climateData;
    ClimateFileHeader header;
    
    if (!loadClimateData(climateFile, h_climateData, header)) {
        std::cerr << "❌ Failed to load climate file\n";
        return EXIT_FAILURE;
    }
    
    std::cout << "📦 Loaded climate data: " 
              << (h_climateData.size() * sizeof(double) / (1024.0*1024.0*1024.0)) << " GB\n\n";
    
    // Validate input
    validateClimateData(h_climateData, "Pre-Terrain");
    
    // Load terrain
    TerrainData terrain = loadTerrainData(terrainFile);
    
    // Upload climate to GPU
    double* d_climateData;
    size_t totalDataSize = WIDTH * HEIGHT * VARIABLES * MONTHS;
    CUDA_CHECK(cudaMalloc(&d_climateData, totalDataSize * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_climateData, h_climateData.data(), 
                         totalDataSize * sizeof(double), cudaMemcpyHostToDevice));
    
    // Launch configuration
    int threads = 256;
    int blocks = (WIDTH * HEIGHT + threads - 1) / threads;
    
    // Apply terrain effects sequentially
    std::cout << "🌡️  Applying elevation temperature effects...\n";
    applyElevationTempKernel<<<blocks, threads>>>(d_climateData, terrain.elevation, 
                                                   terrain.isLand, WIDTH, HEIGHT);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::cout << "🌧️  Applying orographic precipitation...\n";
    applyOrographicPrecipKernel<<<blocks, threads>>>(d_climateData, terrain.elevation,
                                                      terrain.isLand, WIDTH, HEIGHT);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::cout << "🏜️  Applying continentality effects...\n";
    applyContinentalityKernel<<<blocks, threads>>>(d_climateData, terrain.continentality,
                                                    terrain.isLand, WIDTH, HEIGHT);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::cout << "🌊 Applying ocean depth modulation...\n";
    applyOceanDepthEffectKernel<<<blocks, threads>>>(d_climateData, terrain.oceanDepth,
                                                      terrain.isLand, WIDTH, HEIGHT);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Download results
    CUDA_CHECK(cudaMemcpy(h_climateData.data(), d_climateData, 
                         totalDataSize * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Validate output
    if (!validateClimateData(h_climateData, "Post-Terrain")) {
        std::cerr << "❌ Validation failed - output may be unstable\n";
    }
    
    // Save output with standard TERRAIN header
    ClimateFileHeader outputHeader = createStandardHeader(TERRAIN_MAGIC);
    
    if (!saveClimateData(outputFile, h_climateData, outputHeader)) {
        std::cerr << "❌ Failed to save output file\n";
        return EXIT_FAILURE;
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_climateData));
    CUDA_CHECK(cudaFree(terrain.elevation));
    CUDA_CHECK(cudaFree(terrain.isLand));
    CUDA_CHECK(cudaFree(terrain.oceanDepth));
    CUDA_CHECK(cudaFree(terrain.continentality));
    
    std::cout << "\n══════════════════════════════════════════════════════════\n";
    std::cout << "✅ Terrain integration complete!\n";
    std::cout << "   Output: " << outputFile << "\n";
    
    return 0;
}