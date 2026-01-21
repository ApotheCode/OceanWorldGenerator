//============================================================
// PNPLGenerator_v3.cu
// Planetary Noise Perturbation Layer - Hemispherical Ocean Gyres
// 
// v3.0 (2025-11-16): Hemispherical Gyre Physics
//   - Separate subtropical gyres at ±30° (strongest)
//   - Separate subpolar gyres at ±60° (weaker)
//   - Proper Coriolis rotation: clockwise NH, counterclockwise SH
//   - Western intensification (Gulf Stream-like currents)
//   - Distinct equatorial dynamics (±10°)
//   - Planetary-scale Rossby waves
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
#include "ClimateFileFormat.h"

constexpr int WIDTH = STANDARD_WIDTH;
constexpr int HEIGHT = STANDARD_HEIGHT;
constexpr int MONTHS = STANDARD_MONTHS;
constexpr int VARIABLES = STANDARD_VARIABLES;
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
// Device: Hemispherical Ocean Gyre Pattern
// Implements realistic subtropical and subpolar gyres with:
// - Clockwise rotation in Northern Hemisphere
// - Counterclockwise rotation in Southern Hemisphere
// - Peak intensity at ±30° (subtropical gyres)
// - Western intensification
// - Weaker subpolar gyres at ±60°
//============================================================
__device__ double planetaryWavePattern(double lon, double lat, int month, 
                                       double rotationPeriod)
{
    // Convert lat to degrees for easier interpretation
    double latDeg = lat * 180.0 / M_PI;
    
    // Hemisphere sign: +1 for Northern, -1 for Southern
    double hemisphereSign = (lat >= 0.0) ? 1.0 : -1.0;
    
    // Seasonal phase shift
    double seasonalPhase = (month * 2.0 * M_PI) / 12.0;
    
    // Calculate local Coriolis parameter (includes sign)
    double f = coriolisParameter(lat, rotationPeriod);
    
    double n = 0.0;
    
    // ================================================================
    // 1. SUBTROPICAL GYRES (±20° to ±40°, peak at ±30°)
    // ================================================================
    // These are the DOMINANT features in ocean circulation
    
    // Subtropical gyre strength (Gaussian centered at ±30°)
    double subtropicalCenter = 30.0;
    double subtropicalWidth = 15.0;
    double subtropicalStrength = exp(-pow((fabs(latDeg) - subtropicalCenter) / subtropicalWidth, 2));
    
    // Western intensification: gyres are asymmetric, with faster currents on western boundary
    // This creates features like the Gulf Stream (NH) and Brazil Current (SH)
    double westIntensification = 0.5 + 0.5 * cos(lon - M_PI/2);  // Peak at west (lon ≈ -π/2)
    
    // Main subtropical gyre circulation
    // Rotation direction reverses with hemisphere (Coriolis effect)
    n += 0.50 * subtropicalStrength * westIntensification * 
         sin(4.0 * lon * hemisphereSign + 2.0 * lat + seasonalPhase * 0.3);
    
    // Subtropical mesoscale eddies (100-500 km)
    n += 0.15 * subtropicalStrength * 
         cos(9.0 * lon * hemisphereSign - 3.5 * lat + seasonalPhase * 0.5);
    
    // ================================================================
    // 2. SUBPOLAR GYRES (±50° to ±70°, peak at ±60°)
    // ================================================================
    // Weaker than subtropical, opposite rotation in same hemisphere
    
    double subpolarCenter = 60.0;
    double subpolarWidth = 12.0;
    double subpolarStrength = 0.4 * exp(-pow((fabs(latDeg) - subpolarCenter) / subpolarWidth, 2));
    
    // Subpolar gyres rotate opposite to subtropical in same hemisphere
    n += 0.25 * subpolarStrength * 
         sin(-3.5 * lon * hemisphereSign + 1.8 * lat - seasonalPhase * 0.2);
    
    // ================================================================
    // 3. EQUATORIAL REGION (±10°)
    // ================================================================
    // Different dynamics: weak Coriolis, upwelling, countercurrents
    
    double equatorialStrength = exp(-pow(latDeg / 10.0, 2));
    
    // Equatorial undercurrent (eastward, opposite to trade winds)
    n += 0.10 * equatorialStrength * sin(6.0 * lon + seasonalPhase);
    
    // Tropical instability waves
    n += 0.08 * equatorialStrength * cos(12.0 * lon - 4.0 * lat + seasonalPhase * 2.0);
    
    // ================================================================
    // 4. PLANETARY-SCALE ROSSBY WAVES
    // ================================================================
    // Long-wavelength patterns that propagate across ocean basins
    
    double coriolisScale = fabs(f) * 1e4;
    
    // Rossby waves (stronger at mid-latitudes)
    double rossbyStrength = exp(-pow(latDeg / 45.0, 2));
    n += 0.15 * rossbyStrength * 
         sin(2.0 * lon + 1.2 * sin(lat) * hemisphereSign + coriolisScale * 0.3);
    
    // ================================================================
    // 5. BOUNDARY CURRENTS
    // ================================================================
    // Enhanced circulation along western and eastern boundaries
    
    // Western boundary current enhancement (Gulf Stream-like)
    double westernBoundary = exp(-pow((lon + M_PI/2) / 0.3, 2));
    n += 0.12 * westernBoundary * subtropicalStrength * 
         sin(8.0 * lat * hemisphereSign + seasonalPhase);
    
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
        
        // Latitude-based amplitude (stronger at mid-latitudes where gyres form)
        // Reduced at poles (ice) and equator (different dynamics)
        double latDeg = lat * 180.0 / M_PI;
        double midLatBoost = exp(-pow((fabs(latDeg) - 45.0) / 30.0, 2));  // Peak at ±45°
        double latAmp = 0.3 + 0.7 * midLatBoost;
        
        // Calculate perturbation in NORMALIZED [0,1] space
        // SST is normalized: -2°C = 0.0, 35°C = 1.0 (37°C range)
        // Target: ±2-3°C gyres → ±0.054-0.081 in normalized space
        // With intensity=0.05, wavePattern~1, latAmp~0.75: base perturbation = 0.0375
        // Multiply by 2.0 to get ~0.075 normalized (2.8°C in real units)
        const double GYRE_AMPLITUDE_BOOST = 2.0;
        
        double perturbation = params.intensity * GYRE_AMPLITUDE_BOOST * latAmp * boundaryFactor * wavePattern;
        
        // Get current value
        double value = getClimateValue(data, x, y, varIdx, month, width, height);
        
        if (isfinite(value)) {
            // Variable-specific perturbation
            // Temperature variables (1,2,3,13): additive in normalized [0,1] space
            if (varIdx == 1 || varIdx == 2 || varIdx == 3 || varIdx == 13) {
                // Direct addition (no extra 0.50 factor)
                // intensity=0.05 with boost → ~0.075 normalized units → ~2.8°C for SST
                value += perturbation;
            }
            // Flux variables (5,6,7,8): multiplicative
            else if (varIdx >= 5 && varIdx <= 8) {
                value *= (1.0 + perturbation * 0.15);
            }
            // Others: gentle multiplicative
            else {
                value *= (1.0 + perturbation * 0.05);
            }
            
            // Clamp to [0,1] - data IS normalized!
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
    std::vector<double> climateData;
    ClimateFileHeader header;
    
    if (!loadClimateData(climateFile, climateData, header)) {
        std::cerr << "❌ Failed to load climate file\n";
        return EXIT_FAILURE;
    }
    
    std::cout << "📦 Loaded climate data: "
              << (climateData.size() * sizeof(double) / (1024.0*1024.0*1024.0)) << " GB\n\n";
    
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
    
    // Save output with standard PNPL header
    ClimateFileHeader outputHeader = createStandardHeader(PNPL_MAGIC);
    
    if (!saveClimateData(outputFile, climateData, outputHeader)) {
        std::cerr << "❌ Failed to save output file\n";
        return EXIT_FAILURE;
    }
    
    // Cleanup
    if (d_landMask) CUDA_CHECK(cudaFree(d_landMask));
    
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "✅ PNPL generation complete!\n";
    std::cout << "   Output: " << outputFile << "\n";
    
    return 0;
}