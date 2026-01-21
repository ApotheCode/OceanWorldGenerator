//============================================================
// RainfallCalculator.cu
// Physics-based rainfall with atmospheric moisture advection
// Author: Claude & Mark Devereux (2025-11-13)
// 
// IMPROVED MODEL (v2.0):
// - Oceanic evaporation (Clausius-Clapeyron)
// - Moisture advection along wind direction
// - Progressive rainout with distance (Dare et al. 2012)
// - Orographic enhancement (Roe 2005)
// - Rain shadow effects
// - ITCZ and storm track convergence
//
// Key Physics:
// 1. Ocean pixels: Local evaporation
// 2. Land pixels: Sample upwind ocean moisture
// 3. Distance decay: exp(-dist/DECAY_LENGTH_KM)
// 4. Orographic: Uplift on windward slopes
// 5. Rain shadow: Suppression on leeward slopes
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

// Physical constants (Earth-calibrated)
constexpr double EARTH_RADIUS_KM = 6371.0;
constexpr double ALPHA_EVAP = 1.2;           // Evaporation weight
constexpr double BETA_CONVERGENCE = 0.8;     // Convergence weight
constexpr double GAMMA_OROGRAPHIC = 0.25;    // 25% gain per 1000m
constexpr double DECAY_LENGTH_KM = 350.0;    // Inland moisture decay
constexpr double RAINSHADOW_FACTOR = 0.40;   // 40% leeward reduction
constexpr double ITCZ_PEAK_LAT = 5.0;        // ITCZ center (degrees N)
constexpr double ITCZ_MIGRATION = 10.0;      // Seasonal shift (degrees)
constexpr double ITCZ_WIDTH = 15.0;          // ITCZ width (degrees)
constexpr double STORM_TRACK_LAT = 45.0;     // Mid-latitude storms
constexpr double STORM_TRACK_WIDTH = 15.0;   // Storm track width

// Variable indices
constexpr int VAR_RAINFALL = 0;
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
// Device: Clausius-Clapeyron Evaporation (Ocean Only)
//============================================================
__device__ double computeOceanEvaporation(double sst_normalized)
{
    // Denormalize SST: [-2, +35]°C → [0, 1]
    double sstC = sst_normalized * 37.0 - 2.0;
    
    // Clausius-Clapeyron equation (simplified)
    // e_s = 6.112 * exp(17.67 * T / (T + 243.5))  [hPa]
    double es = 6.112 * exp(17.67 * sstC / (sstC + 243.5));
    
    // Bulk evaporation (proportional to saturation vapor pressure)
    // Normalized to [0, 1] range where warm tropical ocean = ~1.0
    // Cold polar ocean = ~0.1
    double evap = es / 60.0; // Scale factor (60 hPa = warm tropical)
    
    return fmin(1.0, fmax(0.0, evap));
}

//============================================================
// Device: Sample Upwind Ocean Moisture
// Traces back along wind direction to find moisture source
// Based on Dare et al. (2012) and Ogino et al. (2016)
//============================================================
__device__ double sampleUpwindMoisture(int x, int y,
                                       const double* climateData,
                                       const unsigned char* isLand,
                                       const float* elevation,
                                       int windDir,
                                       int month,
                                       int width, int height,
                                       float startDistKm)
{
    // Maximum trace distance (km) - atmospheric moisture depletion range
    constexpr float MAX_TRACE_KM = 1000.0f;
    constexpr float STEP_SIZE_KM = 20.0f; // 20 km steps (~2 pixels)
    
    double lat = 90.0 - (y + 0.5) * 180.0 / height;
    double cosLat = cos(lat * DEG_TO_RAD);
    float pixelKm = 11.1f * cosLat; // km per pixel at this latitude
    
    // Start from current position, trace upwind
    float traceX = (float)x;
    float traceY = (float)y;
    float tracedDistKm = 0.0f;
    float elevationGain = 0.0f; // Track orographic rainout en route
    float prevElev = elevation[y * width + x];
    
    int traceSteps = (int)(MAX_TRACE_KM / STEP_SIZE_KM);
    
    for (int step = 0; step < traceSteps; step++) {
        // Move upwind (opposite of wind direction)
        traceX -= windDir * (STEP_SIZE_KM / pixelKm);
        
        // Wrap longitude
        while (traceX < 0) traceX += width;
        while (traceX >= width) traceX -= width;
        
        // Check latitude bounds
        if (traceY < 0 || traceY >= height) break;
        
        int tx = (int)traceX;
        int ty = (int)traceY;
        int tidx = ty * width + tx;
        
        tracedDistKm += STEP_SIZE_KM;
        
        // Found ocean source?
        if (isLand[tidx] == 0) {
            // Sample ocean evaporation
            double sst = getClimateValue(climateData, tx, ty, VAR_SST, month, width, height);
            double oceanEvap = computeOceanEvaporation(sst);
            
            // Total distance from ocean to target pixel
            float totalDistKm = startDistKm + tracedDistKm;
            
            // Distance decay (Ogino et al. 2016: sharp drop within 300 km)
            // Dare et al. (2012): island decay ~88-160 km, continental ~890 km
            double distanceDecay = exp(-totalDistKm / DECAY_LENGTH_KM);
            
            // Orographic depletion (moisture removed by mountains en route)
            // Based on Roe (2005) orographic precipitation theory
            double orographicDepletion = exp(-elevationGain / 2000.0); // e-folding at 2000m
            
            return oceanEvap * distanceDecay * orographicDepletion;
        }
        
        // Still over land - accumulate orographic moisture loss
        float currElev = elevation[tidx];
        if (currElev > prevElev) {
            // Ascending slope: moisture precipitates out
            elevationGain += (currElev - prevElev);
        }
        prevElev = currElev;
    }
    
    // No ocean found within search radius
    // Deep continental interior: minimal moisture
    return 0.05; // 5% of maximum (dry continental climate)
}

//============================================================
// Device: Convergence Zones (ITCZ + Storm Tracks)
//============================================================
__device__ double computeConvergence(double lat, int month)
{
    // Seasonal ITCZ migration (follows sun)
    double seasonalShift = ITCZ_MIGRATION * sin(2.0 * M_PI * month / 12.0);
    double itczLat = ITCZ_PEAK_LAT + seasonalShift;
    
    // ITCZ contribution (Gaussian centered on itczLat)
    double itczDist = fabs(lat - itczLat);
    double itczContrib = exp(-0.5 * pow(itczDist / ITCZ_WIDTH, 2.0));
    
    // Mid-latitude storm tracks (both hemispheres)
    double stormNorth = exp(-0.5 * pow((lat - STORM_TRACK_LAT) / STORM_TRACK_WIDTH, 2.0));
    double stormSouth = exp(-0.5 * pow((lat + STORM_TRACK_LAT) / STORM_TRACK_WIDTH, 2.0));
    double stormContrib = 0.6 * (stormNorth + stormSouth);
    
    // Combine
    double convergence = itczContrib + stormContrib;
    
    // ✨ NEW: Seasonal intensity variation (summer = stronger)
    double monthAngle = 2.0 * M_PI * month / 12.0;
    double seasonalIntensity = 1.0 + 0.2 * sin(monthAngle); // ±20% variation
    convergence *= seasonalIntensity;
    
    return fmin(1.0, convergence);
}

//============================================================
// Device: Prevailing Wind Direction
//============================================================
__device__ int getWindDirection(double lat)
{
    double absLat = fabs(lat);
    
    if (absLat < 30.0) return 1;       // Trade winds (Easterlies) → +x
    else if (absLat < 60.0) return -1; // Westerlies → -x
    else return 1;                     // Polar Easterlies → +x
}

//============================================================
// Device: Distance to Coast (Approximate)
//============================================================
__device__ float approximateDistanceToCoast(int x, int y, 
                                             const unsigned char* isLand,
                                             int width, int height)
{
    // For land pixels: search for nearest ocean
    // For ocean pixels: search for nearest land
    unsigned char myType = isLand[y * width + x];
    
    // Quick local search (16-pixel radius = ~160 km)
    int searchRadius = 16;
    float minDistKm = 1e9f;
    
    for (int dy = -searchRadius; dy <= searchRadius; dy++) {
        for (int dx = -searchRadius; dx <= searchRadius; dx++) {
            if (dx == 0 && dy == 0) continue;
            
            int nx = (x + dx + width) % width; // Wrap longitude
            int ny = y + dy;
            
            if (ny < 0 || ny >= height) continue;
            
            int nIdx = ny * width + nx;
            
            // Found opposite type?
            if (isLand[nIdx] != myType) {
                // Approximate distance (km) using grid spacing
                // 0.1° grid ≈ 11.1 km at equator, less at poles
                double lat = (90.0 - (y + 0.5) * 180.0 / height) * DEG_TO_RAD;
                double cosLat = cos(lat);
                float distKm = sqrt(dx * dx * cosLat * cosLat + dy * dy) * 11.1f;
                
                minDistKm = fmin(minDistKm, distKm);
            }
        }
    }
    
    // If no opposite type found nearby, return large distance
    return (minDistKm < 1e8f) ? minDistKm : 1000.0f;
}

//============================================================
// Device: Orographic Slope
//============================================================
__device__ float computeWindwardSlope(int x, int y,
                                      const float* elevation,
                                      int windDir,
                                      int width, int height)
{
    float elevHere = elevation[y * width + x];
    
    // Sample upwind elevation
    int upwindX = (x - windDir + width) % width;
    float elevUpwind = elevation[y * width + upwindX];
    
    // Slope in wind direction (positive = upslope)
    float slope = (elevHere - elevUpwind) / 11.1f; // meters per km
    
    return fmax(0.0f, slope / 1000.0f); // Normalize to "km of elevation per km"
}

//============================================================
// Device: Rain Shadow Effect
//============================================================
__device__ float computeRainShadow(int x, int y,
                                   const float* elevation,
                                   int windDir,
                                   int width, int height)
{
    float elevHere = elevation[y * width + x];
    
    // Check if we're downwind of a mountain (leeward side)
    int downwindX = (x + windDir + width) % width;
    float elevDownwind = elevation[y * width + downwindX];
    
    // Are we on descending slope?
    if (elevDownwind < elevHere) {
        // Compute maximum elevation upwind (within 10 pixels = ~100 km)
        float maxUpwindElev = elevHere;
        for (int i = 1; i <= 10; i++) {
            int upwindX = (x - windDir * i + width) % width;
            float elevUp = elevation[y * width + upwindX];
            maxUpwindElev = fmax(maxUpwindElev, elevUp);
        }
        
        // Rain shadow strength proportional to elevation drop
        float elevDrop = maxUpwindElev - elevHere;
        if (elevDrop > 500.0f) { // Significant mountain (500m+)
            return RAINSHADOW_FACTOR * fmin(1.0f, elevDrop / 2000.0f);
        }
    }
    
    return 0.0f; // No rain shadow
}

//============================================================
// Kernel: Calculate Monthly Rainfall
// Physics-based approach with moisture advection
//============================================================
__global__ void calculateRainfallKernel(const double* __restrict__ climateData,
                                        const float* __restrict__ elevation,
                                        const unsigned char* __restrict__ isLand,
                                        double* __restrict__ rainfallRaw,
                                        int month,
                                        int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    
    int x = idx % width;
    int y = idx / width;
    
    // Latitude (degrees, -90 to +90)
    double lat = 90.0 - (y + 0.5) * 180.0 / height;
    bool land = (isLand[idx] == 1);
    
    // Get SST for this cell
    double sst = getClimateValue(climateData, x, y, VAR_SST, month, width, height);
    
    // ================================================================
    // STEP 1: MOISTURE SOURCE (with transport)
    // ================================================================
    double moistureAvailable;
    
    if (!land) {
        // Over ocean: use local evaporation
        moistureAvailable = computeOceanEvaporation(sst);
    } else {
        // Over land: sample moisture from upwind ocean source
        // This represents atmospheric moisture advection
        float distToCoast = approximateDistanceToCoast(x, y, isLand, width, height);
        int windDir = getWindDirection(lat);
        
        moistureAvailable = sampleUpwindMoisture(
            x, y, climateData, isLand, elevation,
            windDir, month, width, height, distToCoast
        );
    }
    
    // ================================================================
    // STEP 2: CONVERGENCE ZONES (ITCZ, Storm Tracks)
    // ================================================================
    // Large-scale atmospheric convergence creates lifting
    double convergence = computeConvergence(lat, month);
    
    // ================================================================
    // STEP 3: OROGRAPHIC ENHANCEMENT
    // ================================================================
    // Windward slopes force uplift → precipitation
    int windDir = getWindDirection(lat);
    float slope = computeWindwardSlope(x, y, elevation, windDir, width, height);
    
    // Orographic precipitation (Roe 2005)
    // P_oro = γ * slope * moisture
    // Enhanced on windward slopes, suppressed on leeward
    double orographicFactor = 1.0 + GAMMA_OROGRAPHIC * slope * slope * moistureAvailable;
    
    // ================================================================
    // STEP 4: RAIN SHADOW
    // ================================================================
    float rainShadow = computeRainShadow(x, y, elevation, windDir, width, height);
    double rainShadowFactor = 1.0 - rainShadow;
    
    // ================================================================
    // STEP 5: COMBINE PHYSICAL PROCESSES
    // ================================================================
    // Base precipitation from moisture + convergence
    double basePrecip = ALPHA_EVAP * moistureAvailable 
                      + BETA_CONVERGENCE * convergence;
    
    // Apply orographic enhancement
    basePrecip *= orographicFactor;
    
    // Apply rain shadow suppression
    basePrecip *= rainShadowFactor;
    
    // Ensure non-negative
    double rainfall = fmax(0.0, basePrecip);
    
    // Store raw value (normalized later)
    rainfallRaw[idx] = rainfall;
}

//============================================================
// Kernel: Normalize Rainfall to [0, 1]
//============================================================
__global__ void normalizeRainfallKernel(double* __restrict__ climateData,
                                        const double* __restrict__ rainfallRaw,
                                        double minRain, double maxRain,
                                        int month, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    
    int x = idx % width;
    int y = idx / width;
    
    // Normalize to [0, 1]
    double raw = rainfallRaw[idx];
    double normalized = 0.0;
    
    if (maxRain > minRain) {
        normalized = (raw - minRain) / (maxRain - minRain);
        normalized = fmin(1.0, fmax(0.0, normalized));
    }
    
    // Write to Variable 0 (rainfall)
    setClimateValue(climateData, x, y, VAR_RAINFALL, month, width, height, normalized);
}

//============================================================
// Host: Load Climate Data
//============================================================
bool loadClimateData(const std::string& filename, std::vector<double>& data)
{
    ClimateFileHeader header;
    return ::loadClimateData(filename, data, header);
}

//============================================================
// Host: Load Terrain Data
//============================================================
bool loadTerrainData(const std::string& filename,
                    std::vector<float>& elevation,
                    std::vector<unsigned char>& isLand)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Failed to open: " << filename << "\n";
        return false;
    }
    
    // Read 28-byte standard header
    ClimateFileHeader header;
    if (!readClimateHeader(file, header)) {
        std::cerr << "❌ Failed to read terrain header\n";
        return false;
    }
    
    int width = header.width;
    int height = header.height;
    
    if (width != WIDTH || height != HEIGHT) {
        std::cerr << "❌ Terrain dimension mismatch: Expected " << WIDTH << "x" << HEIGHT 
                  << ", got " << width << "x" << height << "\n";
        return false;
    }
    
    size_t totalPixels = WIDTH * HEIGHT;
    elevation.resize(totalPixels);
    isLand.resize(totalPixels);
    
    // Read elevation (float array)
    file.read(reinterpret_cast<char*>(elevation.data()), totalPixels * sizeof(float));
    
    // Read land mask (uchar array)
    file.read(reinterpret_cast<char*>(isLand.data()), totalPixels * sizeof(unsigned char));
    
    file.close();
    
    std::cout << "✅ Terrain loaded\n";
    return true;
}

//============================================================
// Host: Save Climate Data
//============================================================
bool saveClimateData(const std::string& filename, const std::vector<double>& data)
{
    ClimateFileHeader header = createStandardHeader(RAINFALL_MAGIC);
    return ::saveClimateData(filename, data, header);
}

//============================================================
// Host: Validation
//============================================================
void validateRainfall(const std::vector<double>& data)
{
    std::cout << "🔍 Validating rainfall data...\n";
    
    size_t nanCount = 0;
    size_t outOfBounds = 0;
    double minVal = 1e9, maxVal = -1e9;
    
    size_t totalPixels = WIDTH * HEIGHT;
    for (size_t i = 0; i < totalPixels; ++i) {
        for (int month = 0; month < MONTHS; ++month) {
            size_t idx = i * VARIABLES * MONTHS + VAR_RAINFALL * MONTHS + month;
            double val = data[idx];
            
            if (!isfinite(val)) {
                nanCount++;
            } else {
                if (val < 0.0 || val > 1.0) outOfBounds++;
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }
        }
    }
    
    std::cout << "   Range: [" << minVal << ", " << maxVal << "]\n";
    std::cout << "   Out of bounds: " << outOfBounds << " ("
              << (outOfBounds * 100.0 / (totalPixels * MONTHS)) << "%)\n";
    std::cout << "   NaN values: " << nanCount << " ("
              << (nanCount * 100.0 / (totalPixels * MONTHS)) << "%)\n";
    
    if (outOfBounds > 0 || nanCount > 0) {
        std::cerr << "⚠️ WARNING: Validation issues detected!\n";
    } else {
        std::cout << "   ✅ Validation passed\n";
    }
}

//============================================================
// Main
//============================================================
int main(int argc, char** argv)
{
    std::string climateFile = "output/Kyushu_Climate_PNPL.bin";
    std::string terrainFile = "output/KyushuTerrainData.bin";
    std::string outputFile = "output/Kyushu_Climate_Rainfall.bin";
    
    // Parse arguments
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
            std::cout << "  -i <file>  Input climate file (default: output/Kyushu_Climate_PNPL.bin)\n";
            std::cout << "  -t <file>  Input terrain file (default: output/KyushuTerrainData.bin)\n";
            std::cout << "  -o <file>  Output file (default: output/Kyushu_Climate_Rainfall.bin)\n";
            std::cout << "  -h, --help Show this help\n";
            return 0;
        }
    }
    
    std::cout << "🌧️ Rainfall Calculator\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "Climate: " << climateFile << "\n";
    std::cout << "Terrain: " << terrainFile << "\n";
    std::cout << "Output:  " << outputFile << "\n\n";
    
    // Load data
    std::vector<double> h_climateData;
    if (!loadClimateData(climateFile, h_climateData)) return EXIT_FAILURE;
    
    std::vector<float> h_elevation;
    std::vector<unsigned char> h_isLand;
    if (!loadTerrainData(terrainFile, h_elevation, h_isLand)) return EXIT_FAILURE;
    
    // Allocate GPU memory
    size_t totalPixels = WIDTH * HEIGHT;
    size_t climateBytes = h_climateData.size() * sizeof(double);
    size_t elevBytes = totalPixels * sizeof(float);
    size_t maskBytes = totalPixels * sizeof(unsigned char);
    
    double* d_climateData;
    float* d_elevation;
    unsigned char* d_isLand;
    double* d_rainfallRaw;
    
    CUDA_CHECK(cudaMalloc(&d_climateData, climateBytes));
    CUDA_CHECK(cudaMalloc(&d_elevation, elevBytes));
    CUDA_CHECK(cudaMalloc(&d_isLand, maskBytes));
    CUDA_CHECK(cudaMalloc(&d_rainfallRaw, totalPixels * sizeof(double)));
    
    // Upload data
    CUDA_CHECK(cudaMemcpy(d_climateData, h_climateData.data(), 
                         climateBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_elevation, h_elevation.data(), 
                         elevBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_isLand, h_isLand.data(), 
                         maskBytes, cudaMemcpyHostToDevice));
    
    // Launch configuration
    int threads = 256;
    int blocks = (totalPixels + threads - 1) / threads;
    
    std::cout << "🌊 Generating rainfall (12 months)...\n";
    
    // Process each month
    for (int month = 0; month < MONTHS; ++month) {
        std::cout << "   Month " << (month + 1) << "/12\r" << std::flush;
        
        // Calculate raw rainfall
        calculateRainfallKernel<<<blocks, threads>>>(
            d_climateData, d_elevation, d_isLand, d_rainfallRaw,
            month, WIDTH, HEIGHT);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Find min/max for normalization
        std::vector<double> h_rainfallRaw(totalPixels);
        CUDA_CHECK(cudaMemcpy(h_rainfallRaw.data(), d_rainfallRaw,
                             totalPixels * sizeof(double), cudaMemcpyDeviceToHost));
        
        double minRain = 1e9, maxRain = -1e9;
        for (double val : h_rainfallRaw) {
            if (isfinite(val)) {
                minRain = std::min(minRain, val);
                maxRain = std::max(maxRain, val);
            }
        }
        
        // Normalize to [0, 1]
        normalizeRainfallKernel<<<blocks, threads>>>(
            d_climateData, d_rainfallRaw, minRain, maxRain,
            month, WIDTH, HEIGHT);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    std::cout << "\n   ✅ All months processed\n\n";
    
    // Download results
    CUDA_CHECK(cudaMemcpy(h_climateData.data(), d_climateData,
                         climateBytes, cudaMemcpyDeviceToHost));
    
    // Validate
    validateRainfall(h_climateData);
    
    // Save output
    if (!saveClimateData(outputFile, h_climateData)) return EXIT_FAILURE;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_climateData));
    CUDA_CHECK(cudaFree(d_elevation));
    CUDA_CHECK(cudaFree(d_isLand));
    CUDA_CHECK(cudaFree(d_rainfallRaw));
    
    std::cout << "\n══════════════════════════════════════════════════════════\n";
    std::cout << "✅ Rainfall generation complete!\n";
    std::cout << "   Output: " << outputFile << "\n";
    
    return 0;
}
