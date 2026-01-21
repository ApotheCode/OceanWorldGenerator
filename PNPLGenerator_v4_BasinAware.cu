//============================================================
// PNPLGenerator_v4_BasinAware.cu
// Planetary Noise Perturbation Layer - Basin-Specific Ocean Gyres
// 
// v4.0 (2025-11-19): Basin-Aware Gyre Physics
//   - Integrates bathymetry and basin detection
//   - Places gyres within individual ocean basins
//   - Subtropical gyres in basins spanning 20-40°
//   - Subpolar gyres in basins spanning 40-60°
//   - Gyres confined to basin boundaries
//   - Supports multiple gyres per basin (North Pacific subtropical + subpolar)
//   - Western intensification per basin
//   - Proper Coriolis effects per hemisphere
// 
// Author: Mark Devereux (CTO)
//============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <unordered_map>
#include "ClimateFileFormat.h"

constexpr int WIDTH = STANDARD_WIDTH;
constexpr int HEIGHT = STANDARD_HEIGHT;
constexpr int MONTHS = STANDARD_MONTHS;
constexpr int VARIABLES = STANDARD_VARIABLES;
constexpr double M_PI = 3.14159265358979323846;
constexpr double DEG_TO_RAD = M_PI / 180.0;
constexpr int MAX_BASINS = 32;

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
    bool respectBasinBoundaries;   // Gyres confined to basins
};

// Basin metadata computed on CPU, uploaded to GPU
struct BasinInfo {
    int basinID;                   // Basin identifier (1-31, 0=land)
    double centroidLat;            // Basin center latitude (degrees)
    double centroidLon;            // Basin center longitude (degrees)
    double minLat;                 // Basin extent
    double maxLat;
    double minLon;
    double maxLon;
    int pixelCount;                // Basin size
    bool hasSubtropicalGyre;       // Spans 20-40°?
    bool hasSubpolarGyre;          // Spans 40-60°?
    double gyreCenterLat;          // Where to place gyre center
    double gyreCenterLon;
};

//============================================================
// Device: Calculate Coriolis Parameter
//============================================================
__device__ __forceinline__ double coriolisParameter(double lat, double rotationPeriod)
{
    double omega = 2.0 * M_PI / (rotationPeriod * 3600.0);
    return 2.0 * omega * sin(lat);
}

//============================================================
// Device: Check if pixel is in a specific basin
//============================================================
__device__ bool inBasin(int x, int y, int targetBasinID, 
                       const unsigned char* __restrict__ basinMap,
                       int width, int height)
{
    if (basinMap == nullptr) return false;
    int idx = y * width + x;
    return (basinMap[idx] == targetBasinID);
}

//============================================================
// Device: Basin-Specific Gyre Pattern
// Generates gyres constrained to individual ocean basins
//============================================================
__device__ double basinGyrePattern(double lon, double lat, int month,
                                   double rotationPeriod,
                                   const BasinInfo* __restrict__ basins,
                                   int numBasins,
                                   const unsigned char* __restrict__ basinMap,
                                   int x, int y, int width, int height)
{
    double latDeg = lat * 180.0 / M_PI;
    double lonDeg = lon * 180.0 / M_PI;
    
    // Find which basin this pixel belongs to
    int pixelBasin = (basinMap != nullptr) ? basinMap[y * width + x] : 0;
    
    // Land or no basin
    if (pixelBasin == 0) return 0.0;
    
    // Find basin info
    const BasinInfo* myBasin = nullptr;
    for (int i = 0; i < numBasins; ++i) {
        if (basins[i].basinID == pixelBasin) {
            myBasin = &basins[i];
            break;
        }
    }
    
    if (myBasin == nullptr) return 0.0;
    
    double n = 0.0;
    double hemisphereSign = (lat >= 0.0) ? 1.0 : -1.0;
    double seasonalPhase = (month * 2.0 * M_PI) / 12.0;
    double f = coriolisParameter(lat, rotationPeriod);
    
    // ================================================================
    // SUBTROPICAL GYRE (if basin spans 20-40°)
    // ================================================================
    if (myBasin->hasSubtropicalGyre) {
        // Distance from gyre center (in degrees)
        double dLat = latDeg - myBasin->gyreCenterLat;
        double dLon = lonDeg - myBasin->gyreCenterLon;
        
        // Handle longitude wraparound
        if (dLon > 180.0) dLon -= 360.0;
        if (dLon < -180.0) dLon += 360.0;
        
        // Gyre extent (larger basins = larger gyres)
        double latExtent = myBasin->maxLat - myBasin->minLat;
        double lonExtent = myBasin->maxLon - myBasin->minLon;
        if (lonExtent < 0.0) lonExtent += 360.0;
        
        double gyreRadiusLat = latExtent * 0.4;  // 40% of basin extent
        double gyreRadiusLon = lonExtent * 0.4;
        
        // Radial distance (elliptical)
        double r = sqrt(pow(dLat / gyreRadiusLat, 2) + pow(dLon / gyreRadiusLon, 2));
        
        // Gyre strength (Gaussian falloff from center)
        double gyreStrength = exp(-pow(r, 2));
        
        // Western intensification (faster currents on western boundary)
        // Reference longitude to basin's western edge
        double lonFromWest = lonDeg - myBasin->minLon;
        if (lonFromWest < 0.0) lonFromWest += 360.0;
        double westIntensification = 1.5 - 0.5 * (lonFromWest / lonExtent);
        westIntensification = fmax(0.5, fmin(2.0, westIntensification));
        
        // Angular position for rotation
        double theta = atan2(dLon, dLat);
        
        // Clockwise (NH) or counterclockwise (SH) rotation
        // Use cos and sin for proper vorticity
        n += 0.50 * gyreStrength * westIntensification *
             cos(theta + hemisphereSign * M_PI/2 + seasonalPhase * 0.1);
    }
    
    // ================================================================
    // SUBPOLAR GYRE (if basin spans 40-60°)
    // ================================================================
    if (myBasin->hasSubpolarGyre) {
        // Subpolar gyres are at higher latitudes, weaker
        double subpolarCenterLat = (fabs(myBasin->gyreCenterLat) > 50.0) 
                                   ? myBasin->gyreCenterLat 
                                   : hemisphereSign * 50.0;
        
        double dLat = latDeg - subpolarCenterLat;
        double dLon = lonDeg - myBasin->gyreCenterLon;
        
        if (dLon > 180.0) dLon -= 360.0;
        if (dLon < -180.0) dLon += 360.0;
        
        double latExtent = myBasin->maxLat - myBasin->minLat;
        double lonExtent = myBasin->maxLon - myBasin->minLon;
        if (lonExtent < 0.0) lonExtent += 360.0;
        
        double gyreRadiusLat = latExtent * 0.3;
        double gyreRadiusLon = lonExtent * 0.3;
        
        double r = sqrt(pow(dLat / gyreRadiusLat, 2) + pow(dLon / gyreRadiusLon, 2));
        double gyreStrength = exp(-pow(r, 2));
        
        double theta = atan2(dLon, dLat);
        
        // Weaker subpolar gyres (0.25 vs 0.50)
        // Opposite rotation to subtropical gyres
        n += 0.25 * gyreStrength *
             cos(theta - hemisphereSign * M_PI/2 + seasonalPhase * 0.1);
    }
    
    // ================================================================
    // EQUATORIAL DYNAMICS (between ±10°, if basin spans equator)
    // ================================================================
    if (fabs(latDeg) < 10.0 && myBasin->minLat < 10.0 && myBasin->maxLat > -10.0) {
        double eqStrength = exp(-pow(latDeg / 5.0, 2));  // Peak at equator
        
        // Eastward equatorial current (simplified)
        // Add zonal Kelvin wave-like pattern
        double kelvinWave = 0.3 * eqStrength * sin(lon * 2.0 + seasonalPhase);
        n += kelvinWave;
    }
    
    // ================================================================
    // PLANETARY ROSSBY WAVES (basin-wide, slower propagation)
    // ================================================================
    double basinWidth = myBasin->maxLon - myBasin->minLon;
    if (basinWidth < 0.0) basinWidth += 360.0;
    
    double wavelength = basinWidth / 3.0;  // 3 waves across basin
    double waveNumber = 2.0 * M_PI / (wavelength * DEG_TO_RAD);
    
    double rossby = 0.15 * sin(waveNumber * lon - 0.5 * seasonalPhase) *
                    cos(2.0 * lat);
    n += rossby;
    
    return n;
}

//============================================================
// Device: Get/Set Climate Value
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
// Kernel: Apply Basin-Aware PNPL
//============================================================
__global__ void applyBasinPNPLKernel(double* __restrict__ data,
                                     const unsigned char* __restrict__ basinMap,
                                     const BasinInfo* __restrict__ basins,
                                     int numBasins,
                                     int width, int height,
                                     int varIdx,
                                     PNPLParams params)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    
    int x = idx % width;
    int y = idx / width;
    
    // Skip land pixels
    if (basinMap != nullptr && basinMap[idx] == 0) return;
    
    // Geographic coordinates
    double lat = (90.0 - (y + 0.5) * 180.0 / height) * DEG_TO_RAD;
    double lon = ((x + 0.5) * 360.0 / width - 180.0) * DEG_TO_RAD;
    
    // Apply to all months
    for (int month = 0; month < MONTHS; ++month) {
        // Generate basin-specific gyre pattern
        double wavePattern = basinGyrePattern(lon, lat, month, params.rotationPeriod,
                                             basins, numBasins, basinMap,
                                             x, y, width, height);
        
        // Latitude-based amplitude (stronger at mid-latitudes)
        double latDeg = lat * 180.0 / M_PI;
        double midLatBoost = exp(-pow((fabs(latDeg) - 45.0) / 30.0, 2));
        double latAmp = 0.3 + 0.7 * midLatBoost;
        
        // Normalized perturbation (target ±2-3°C for SST)
        const double GYRE_AMPLITUDE_BOOST = 2.0;
        double perturbation = params.intensity * GYRE_AMPLITUDE_BOOST * 
                             latAmp * wavePattern;
        
        // Get current value
        double value = getClimateValue(data, x, y, varIdx, month, width, height);
        
        if (isfinite(value)) {
            // Temperature variables: additive
            if (varIdx == 1 || varIdx == 2 || varIdx == 3 || varIdx == 13) {
                value += perturbation;
            }
            // Flux variables: multiplicative
            else if (varIdx >= 5 && varIdx <= 8) {
                value *= (1.0 + perturbation * 0.15);
            }
            // Others: gentle multiplicative
            else {
                value *= (1.0 + perturbation * 0.05);
            }
            
            value = fmax(0.0, fmin(1.0, value));
            setClimateValue(data, x, y, varIdx, month, width, height, value);
        }
    }
}

//============================================================
// Host: Load Bathymetry (28-byte header + float data)
//============================================================
std::vector<float> loadBathymetry(const std::string& filename)
{
    std::cout << "📊 Loading bathymetry: " << filename << "\n";
    
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Cannot open bathymetry file\n";
        return {};
    }
    
    ClimateFileHeader header;
    if (!readClimateHeader(file, header)) {
        std::cerr << "❌ Failed to read bathymetry header\n";
        return {};
    }
    
    if (header.width != WIDTH || header.height != HEIGHT) {
        std::cerr << "❌ Bathymetry size mismatch: " 
                  << header.width << "x" << header.height << "\n";
        return {};
    }
    
    size_t totalPixels = WIDTH * HEIGHT;
    std::vector<float> bathymetry(totalPixels);
    
    file.read(reinterpret_cast<char*>(bathymetry.data()), 
              totalPixels * sizeof(float));
    file.close();
    
    std::cout << "   ✅ Bathymetry loaded: " << WIDTH << "x" << HEIGHT << "\n";
    return bathymetry;
}

//============================================================
// Host: Load Basin Map (reads int32, converts to uint8)
//============================================================
std::vector<unsigned char> loadBasinMap(const std::string& filename)
{
    std::cout << "🗺️  Loading basin map: " << filename << "\n";
    
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Cannot open basin file\n";
        return {};
    }
    
    ClimateFileHeader header;
    if (!readClimateHeader(file, header)) {
        std::cerr << "❌ Failed to read basin header\n";
        return {};
    }
    
    if (header.width != WIDTH || header.height != HEIGHT) {
        std::cerr << "❌ Basin map size mismatch: " 
                  << header.width << "x" << header.height << "\n";
        return {};
    }
    
    size_t totalPixels = WIDTH * HEIGHT;
    std::vector<unsigned char> basinMap(totalPixels);
    
    // Check dtype and read appropriately
    if (header.dtype == 2) {
        // int32 format (from BasinDetector)
        std::vector<int> basinMap32(totalPixels);
        file.read(reinterpret_cast<char*>(basinMap32.data()), 
                  totalPixels * sizeof(int));
        
        // Convert to uint8
        for (size_t i = 0; i < totalPixels; ++i) {
            basinMap[i] = static_cast<unsigned char>(basinMap32[i]);
        }
        
        std::cout << "   Converted int32 → uint8 basin map\n";
    }
    else if (header.dtype == 1) {
        // uint8 format (legacy)
        file.read(reinterpret_cast<char*>(basinMap.data()), totalPixels);
    }
    else {
        std::cerr << "❌ Unsupported basin dtype: " << header.dtype << "\n";
        return {};
    }
    
    file.close();
    
    // Count unique basins
    std::unordered_map<int, int> basinCounts;
    for (unsigned char b : basinMap) {
        basinCounts[b]++;
    }
    
    int numBasins = basinCounts.size();
    if (basinCounts.count(0) > 0) numBasins--;  // Exclude land (ID 0)
    
    std::cout << "   ✅ Basin map loaded: " << numBasins 
              << " ocean basins detected\n";
    
    return basinMap;
}

//============================================================
// Host: Analyze Basins and Compute Gyre Placement
//============================================================
std::vector<BasinInfo> analyzeBasins(const std::vector<unsigned char>& basinMap,
                                     const std::vector<float>& bathymetry)
{
    std::cout << "🔍 Analyzing ocean basins for gyre placement...\n";
    
    std::unordered_map<int, std::vector<std::pair<int, int>>> basinPixels;
    
    // Group pixels by basin
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            int idx = y * WIDTH + x;
            unsigned char basinID = basinMap[idx];
            
            if (basinID > 0) {  // Skip land (0)
                basinPixels[basinID].push_back({x, y});
            }
        }
    }
    
    std::vector<BasinInfo> basins;
    
    for (auto& [basinID, pixels] : basinPixels) {
        // Ignore tiny basins (< 1000 pixels = too small for gyres)
        if (pixels.size() < 75000) continue;
        
        BasinInfo info;
        info.basinID = basinID;
        info.pixelCount = pixels.size();
        
        // Compute bounds and centroid
        double sumLat = 0.0, sumLon = 0.0;
        info.minLat = 90.0;
        info.maxLat = -90.0;
        info.minLon = 180.0;
        info.maxLon = -180.0;
        
        for (auto [x, y] : pixels) {
            double lat = 90.0 - (y + 0.5) * 180.0 / HEIGHT;
            double lon = (x + 0.5) * 360.0 / WIDTH - 180.0;
            
            sumLat += lat;
            sumLon += lon;
            
            info.minLat = std::min(info.minLat, lat);
            info.maxLat = std::max(info.maxLat, lat);
            info.minLon = std::min(info.minLon, lon);
            info.maxLon = std::max(info.maxLon, lon);
        }
        
        info.centroidLat = sumLat / pixels.size();
        info.centroidLon = sumLon / pixels.size();
        
        // Determine gyre types based on latitude range
        info.hasSubtropicalGyre = (info.minLat < 40.0 && info.maxLat > 20.0) ||
                                  (info.minLat < -20.0 && info.maxLat > -40.0);
        
        info.hasSubpolarGyre = (info.minLat < 60.0 && info.maxLat > 40.0) ||
                               (info.minLat < -40.0 && info.maxLat > -60.0);
        
        // Place gyre center
        if (info.hasSubtropicalGyre) {
            // Place at 30° (NH) or -30° (SH)
            if (info.centroidLat > 0) {
                info.gyreCenterLat = 30.0;
            } else {
                info.gyreCenterLat = -30.0;
            }
        } else if (info.hasSubpolarGyre) {
            // Place at 50° (NH) or -50° (SH)
            if (info.centroidLat > 0) {
                info.gyreCenterLat = 50.0;
            } else {
                info.gyreCenterLat = -50.0;
            }
        } else {
            info.gyreCenterLat = info.centroidLat;
        }
        
        info.gyreCenterLon = info.centroidLon;
        
        basins.push_back(info);
        
        std::cout << "   Basin " << basinID << ": "
                  << "Lat [" << info.minLat << "° to " << info.maxLat << "°], "
                  << "Lon [" << info.minLon << "° to " << info.maxLon << "°]\n"
                  << "      Centroid: (" << info.centroidLat << "°, " 
                  << info.centroidLon << "°), "
                  << pixels.size() << " pixels\n";
        
        if (info.hasSubtropicalGyre) {
            std::cout << "      → Subtropical gyre at " 
                      << info.gyreCenterLat << "°\n";
        }
        if (info.hasSubpolarGyre) {
            std::cout << "      → Subpolar gyre at " 
                      << (info.centroidLat > 0 ? 50.0 : -50.0) << "°\n";
        }
    }
    
    std::cout << "   ✅ Analyzed " << basins.size() << " ocean basins\n\n";
    return basins;
}

//============================================================
// Host: Apply Basin-Aware PNPL
//============================================================
void applyBasinPNPL(std::vector<double>& climateData,
                   const unsigned char* d_basinMap,
                   const BasinInfo* d_basins,
                   int numBasins,
                   const PNPLParams& params)
{
    std::cout << "🌀 Applying Basin-Aware PNPL...\n";
    std::cout << "   Intensity: " << params.intensity << "\n";
    std::cout << "   Rotation period: " << params.rotationPeriod << " hours\n";
    std::cout << "   Number of basins: " << numBasins << "\n";
    
    size_t totalSize = climateData.size();
    
    // Upload climate data to GPU
    double* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, totalSize * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_data, climateData.data(), 
                         totalSize * sizeof(double), 
                         cudaMemcpyHostToDevice));
    
    // Process each variable
    int threadsPerBlock = 256;
    int blocksPerGrid = (WIDTH * HEIGHT + threadsPerBlock - 1) / threadsPerBlock;
    
    for (int varIdx = 0; varIdx < VARIABLES; ++varIdx) {
        applyBasinPNPLKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_data, d_basinMap, d_basins, numBasins,
            WIDTH, HEIGHT, varIdx, params
        );
        CUDA_CHECK(cudaGetLastError());
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Download results
    CUDA_CHECK(cudaMemcpy(climateData.data(), d_data,
                         totalSize * sizeof(double),
                         cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_data));
    
    std::cout << "   ✅ PNPL applied to all variables\n\n";
}

//============================================================
// Host: Load Climate Data (wrapper using ClimateFileFormat helpers)
//============================================================
std::vector<double> loadClimateDataWrapper(const std::string& filename)
{
    std::cout << "📦 Loading climate data: " << filename << "\n";
    
    std::vector<double> data;
    ClimateFileHeader header;
    
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Cannot open climate file\n";
        return {};
    }
    
    if (!readClimateHeader(file, header)) {
        std::cerr << "❌ Failed to read climate header\n";
        return {};
    }
    
    if (header.width != WIDTH || header.height != HEIGHT) {
        std::cerr << "❌ Climate data size mismatch\n";
        return {};
    }
    
    size_t totalElements = (size_t)WIDTH * HEIGHT * VARIABLES * MONTHS;
    data.resize(totalElements);
    
    file.read(reinterpret_cast<char*>(data.data()), totalElements * sizeof(double));
    file.close();
    
    std::cout << "   ✅ Climate data loaded: " 
              << (totalElements * sizeof(double) / (1024.0*1024.0)) << " MB\n";
    
    return data;
}

//============================================================
// Host: Save Climate Data (same as v3)
//============================================================
// Host: Save Climate Data (uses ClimateFileFormat helper)
//============================================================
bool saveClimateDataOutput(const std::string& filename, 
                          const std::vector<double>& data,
                          const ClimateFileHeader& header)
{
    std::cout << "💾 Saving PNPL output: " << filename << "\n";
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Cannot create output file\n";
        return false;
    }
    
    writeClimateHeader(file, header);
    file.write(reinterpret_cast<const char*>(data.data()), 
               data.size() * sizeof(double));
    file.close();
    
    std::cout << "   ✅ Saved: " << (data.size() * sizeof(double) / (1024.0*1024.0)) 
              << " MB\n";
    
    return true;
}

//============================================================
// Host: Validate Output (same as v3)
//============================================================
bool validatePNPL(const std::vector<double>& data)
{
    std::cout << "🔍 Validating PNPL output...\n";
    
    size_t outOfBounds = 0;
    size_t nanCount = 0;
    double minVal = 1e9, maxVal = -1e9;
    
    for (double val : data) {
        if (!isfinite(val)) {
            nanCount++;
        } else {
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
            if (val < 0.0 || val > 1.0) outOfBounds++;
        }
    }
    
    double oobPct = (outOfBounds * 100.0) / data.size();
    double nanPct = (nanCount * 100.0) / data.size();
    
    std::cout << "   Value range: [" << minVal << ", " << maxVal << "]\n";
    std::cout << "   Out of [0,1]: " << outOfBounds << " (" << oobPct << "%)\n";
    std::cout << "   NaN values: " << nanCount << " (" << nanPct << "%)\n";
    
    if (oobPct > 1.0) {
        std::cerr << "⚠️ WARNING: High out-of-bounds rate!\n";
        return false;
    }
    
    if (nanPct > 1.0) {
        std::cerr << "⚠️ WARNING: High NaN rate!\n";
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
    std::string bathymetryFile = "Bathymetry.bath";
    std::string basinFile = "Basins.basin";
    std::string outputFile = "Earth_Climate_PNPL_v4.bin";
    
    double intensity = 0.05;
    double rotationPeriod = 24.0;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0 && i+1 < argc)
            climateFile = argv[++i];
        else if (strcmp(argv[i], "-b") == 0 && i+1 < argc)
            bathymetryFile = argv[++i];
        else if (strcmp(argv[i], "--basins") == 0 && i+1 < argc)
            basinFile = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc)
            outputFile = argv[++i];
        else if (strcmp(argv[i], "--intensity") == 0 && i+1 < argc)
            intensity = atof(argv[++i]);
        else if (strcmp(argv[i], "--rotation") == 0 && i+1 < argc)
            rotationPeriod = atof(argv[++i]);
    }
    
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "🌊 PNPL Generator v4 - Basin-Aware Ocean Gyres\n";
    std::cout << "══════════════════════════════════════════════════════════\n\n";
    
    // Load bathymetry
    auto bathymetry = loadBathymetry(bathymetryFile);
    if (bathymetry.empty()) {
        std::cerr << "❌ Failed to load bathymetry\n";
        return EXIT_FAILURE;
    }
    
    // Load basin map
    auto basinMap = loadBasinMap(basinFile);
    if (basinMap.empty()) {
        std::cerr << "❌ Failed to load basin map\n";
        return EXIT_FAILURE;
    }
    
    // Analyze basins and compute gyre placement
    auto basins = analyzeBasins(basinMap, bathymetry);
    if (basins.empty()) {
        std::cerr << "❌ No valid ocean basins found\n";
        return EXIT_FAILURE;
    }
    
    // Upload basin data to GPU
    unsigned char* d_basinMap;
    CUDA_CHECK(cudaMalloc(&d_basinMap, basinMap.size()));
    CUDA_CHECK(cudaMemcpy(d_basinMap, basinMap.data(), 
                         basinMap.size(), cudaMemcpyHostToDevice));
    
    BasinInfo* d_basins;
    CUDA_CHECK(cudaMalloc(&d_basins, basins.size() * sizeof(BasinInfo)));
    CUDA_CHECK(cudaMemcpy(d_basins, basins.data(),
                         basins.size() * sizeof(BasinInfo),
                         cudaMemcpyHostToDevice));
    
    // Load climate data
    auto climateData = loadClimateDataWrapper(climateFile);
    if (climateData.empty()) {
        std::cerr << "❌ Failed to load climate data\n";
        return EXIT_FAILURE;
    }
    
    // Setup PNPL parameters
    PNPLParams params;
    params.intensity = intensity;
    params.rotationPeriod = rotationPeriod;
    params.respectBasinBoundaries = true;
    
    // Apply basin-aware PNPL
    applyBasinPNPL(climateData, d_basinMap, d_basins, basins.size(), params);
    
    // Validate
    if (!validatePNPL(climateData)) {
        std::cerr << "⚠️ Validation warnings detected\n";
    }
    
    // Save output
    ClimateFileHeader outputHeader = createStandardHeader(PNPL_MAGIC);
    
    if (!saveClimateDataOutput(outputFile, climateData, outputHeader)) {
        std::cerr << "❌ Failed to save output file\n";
        return EXIT_FAILURE;
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_basinMap));
    CUDA_CHECK(cudaFree(d_basins));
    
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "✅ Basin-aware PNPL generation complete!\n";
    std::cout << "   Output: " << outputFile << "\n";
    std::cout << "   Basins processed: " << basins.size() << "\n";
    
    return 0;
}
