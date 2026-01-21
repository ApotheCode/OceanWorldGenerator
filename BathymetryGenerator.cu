//============================================================
// BathymetryGenerator.cu
// Procedural Ocean Floor Generation with Basin Detection
// 
// Generates realistic bathymetry using:
// 
// PLATE TECTONICS:
//   - Voronoi tessellation for tectonic plates/ocean basins
//   - Spherical uniform distribution (proper pole coverage)
//   - Plate motion vectors for realistic boundary classification:
//     * Divergent boundaries → Mid-ocean ridges (shallow)
//     * Convergent boundaries → Oceanic trenches (deep)
//     * Transform boundaries → Fracture zones (neutral)
//   - Crustal aging: depth increases quadratically away from ridges
// 
// MULTI-SCALE ROUGHNESS (Earth-like detail):
//   - Large-scale: Seamount chains, fracture zones (50-100km)
//   - Abyssal hill provinces (10-20km wavelength, covers ~80% of floor)
//   - Seamount clusters (volcanic peaks, 300-800m rise)
//   - Fine-scale texture (1-5km rocky features)
//   - Ridge segmentation (fractured, irregular ridges)
//   - 5-octave fractal Perlin noise for micro-bathymetry
// 
// SPHERICAL WRAPPING:
//   - Seamless longitude wrapping (toroidal at dateline)
//   - Great-circle distance calculations
//   - Equirectangular projection ready
// 
// Outputs:
//   - Bathymetry depth map (.BATH file)
//   - Ocean basin ID map (.BASN file)
// 
// Author: Mark Devereux with Claude (CTO)
//============================================================
// Enhanced with ChatGPT's recommendations for plate tectonics realism
//============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstring>
#include <cstdint>
#include <random>
#include "ClimateFileFormat.h"

constexpr int WIDTH = STANDARD_WIDTH;
constexpr int HEIGHT = STANDARD_HEIGHT;
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

//============================================================
// Magic Numbers for File Types
//============================================================
constexpr uint32_t BATH_MAGIC = 0x42415448;  // "BATH"
constexpr uint32_t BASN_MAGIC = 0x4241534E;  // "BASN"

//============================================================
// Bathymetry Parameters
//============================================================
struct BathymetryParams {
    int numTectonicPlates;       // Number of Voronoi seed points (6-12 typical)
    double ridgeDepth;           // Mid-ocean ridge depth in meters (-2500m typical)
    double abyssalDepth;         // Abyssal plain depth (-4000m to -5000m)
    double trenchDepth;          // Oceanic trench depth (-8000m to -11000m)
    double continentalShelf;     // Shelf depth (-200m)
    double noiseAmplitude;       // Perlin noise strength (500-1000m typical)
    int perlinOctaves;           // Number of octave layers (4-6 typical)
    double perlinLacunarity;     // Frequency multiplier per octave (2.0 typical)
    double perlinPersistence;    // Amplitude multiplier per octave (0.5 typical)
    unsigned long long seed;     // Random seed for reproducibility
};

//============================================================
// Voronoi Plate Seed Point with Motion Vector
//============================================================
struct PlateSeed {
    double lon;        // Radians
    double lat;        // Radians
    int id;            // Basin ID
    double motionX;    // Motion vector X (longitude direction)
    double motionY;    // Motion vector Y (latitude direction)
};

//============================================================
// Device: Perlin Noise Hash Function
//============================================================
__device__ __forceinline__ int perlinHash(int x, int y, int z, unsigned long long seed)
{
    // Simple hash combining coordinates with seed
    unsigned long long h = seed;
    h ^= x * 374761393ULL;
    h ^= y * 668265263ULL;
    h ^= z * 1274126177ULL;
    h = (h ^ (h >> 32)) * 0xd6e8feb86659fd93ULL;
    h = (h ^ (h >> 32)) * 0xd6e8feb86659fd93ULL;
    return (int)(h & 0xFF);
}

//============================================================
// Device: Smoothstep Interpolation
//============================================================
__device__ __forceinline__ double smoothstep(double t)
{
    return t * t * (3.0 - 2.0 * t);
}

//============================================================
// Device: 3D Perlin Noise (wraps in X for seamless longitude)
//============================================================
__device__ double perlinNoise3D(double x, double y, double z, unsigned long long seed)
{
    // For seamless wrapping, treat x as periodic with period 2.0 (maps to full longitude)
    // This ensures left edge matches right edge perfectly
    
    int xi = (int)floor(x);
    int yi = (int)floor(y);
    int zi = (int)floor(z);
    
    double xf = x - xi;
    double yf = y - yi;
    double zf = z - zi;
    
    // Smoothstep interpolation weights
    double u = smoothstep(xf);
    double v = smoothstep(yf);
    double w = smoothstep(zf);
    
    // For longitude wrapping, ensure xi wraps with period 2
    // (since x is normalized to [-1, 1] for full planet)
    auto wrapX = [](int x) -> int {
        // Wrap to period covering -1 to 1 (width = 2)
        // Scale up to avoid floating point issues
        int period = 1000; // Arbitrary large period for hash stability
        return ((x % period) + period) % period;
    };
    
    // Hash gradients at cube corners (with X wrapping)
    int g000 = perlinHash(wrapX(xi),   yi,   zi,   seed);
    int g001 = perlinHash(wrapX(xi),   yi,   zi+1, seed);
    int g010 = perlinHash(wrapX(xi),   yi+1, zi,   seed);
    int g011 = perlinHash(wrapX(xi),   yi+1, zi+1, seed);
    int g100 = perlinHash(wrapX(xi+1), yi,   zi,   seed);
    int g101 = perlinHash(wrapX(xi+1), yi,   zi+1, seed);
    int g110 = perlinHash(wrapX(xi+1), yi+1, zi,   seed);
    int g111 = perlinHash(wrapX(xi+1), yi+1, zi+1, seed);
    
    // Convert hash to gradient direction (-1 to 1)
    auto toGrad = [](int h) -> double { return (h & 0x80) ? -1.0 : 1.0; };
    
    // Trilinear interpolation
    double x00 = (1.0 - u) * toGrad(g000) + u * toGrad(g100);
    double x01 = (1.0 - u) * toGrad(g001) + u * toGrad(g101);
    double x10 = (1.0 - u) * toGrad(g010) + u * toGrad(g110);
    double x11 = (1.0 - u) * toGrad(g011) + u * toGrad(g111);
    
    double y0 = (1.0 - v) * x00 + v * x10;
    double y1 = (1.0 - v) * x01 + v * x11;
    
    return (1.0 - w) * y0 + w * y1;
}

//============================================================
// Device: Multi-Octave Perlin (Fractal Brownian Motion)
//============================================================
__device__ double fractalPerlin(double x, double y, const BathymetryParams& params)
{
    double total = 0.0;
    double amplitude = 1.0;
    double frequency = 1.0;
    double maxValue = 0.0;
    
    for (int octave = 0; octave < params.perlinOctaves; ++octave) {
        total += perlinNoise3D(x * frequency, y * frequency, 
                               (double)octave, params.seed) * amplitude;
        
        maxValue += amplitude;
        amplitude *= params.perlinPersistence;
        frequency *= params.perlinLacunarity;
    }
    
    return total / maxValue;  // Normalize to [-1, 1]
}

//============================================================
// Device: Great Circle Distance with Toroidal Longitude Wrapping
// Handles seamless wrapping at dateline for cylindrical projection
//============================================================
__device__ __forceinline__ double greatCircleDistance(double lon1, double lat1,
                                                       double lon2, double lat2)
{
    // Haversine formula with longitude wrapping
    double dlat = lat2 - lat1;
    double dlon = lon2 - lon1;
    
    // Normalize dlon to [-π, π] for toroidal wrapping (shortest path around sphere)
    while (dlon > M_PI) dlon -= 2.0 * M_PI;
    while (dlon < -M_PI) dlon += 2.0 * M_PI;
    
    double a = sin(dlat/2) * sin(dlat/2) +
               cos(lat1) * cos(lat2) * sin(dlon/2) * sin(dlon/2);
    
    return 2.0 * atan2(sqrt(a), sqrt(1.0 - a));  // Angular distance in radians
}

//============================================================
// Device: Classify Plate Boundary Type
// Returns: -1 = convergent (trench), 0 = transform, +1 = divergent (ridge)
//============================================================
__device__ int classifyBoundary(const PlateSeed& plate1, const PlateSeed& plate2,
                                 double boundaryLon, double boundaryLat)
{
    // Calculate vector from boundary to each plate center (with longitude wrapping)
    double dx1 = plate1.lon - boundaryLon;
    double dy1 = plate1.lat - boundaryLat;
    double dx2 = plate2.lon - boundaryLon;
    double dy2 = plate2.lat - boundaryLat;
    
    // Wrap longitude differences to [-π, π]
    while (dx1 > M_PI) dx1 -= 2.0 * M_PI;
    while (dx1 < -M_PI) dx1 += 2.0 * M_PI;
    while (dx2 > M_PI) dx2 -= 2.0 * M_PI;
    while (dx2 < -M_PI) dx2 += 2.0 * M_PI;
    
    // Normalize to unit vectors
    double len1 = sqrt(dx1*dx1 + dy1*dy1);
    double len2 = sqrt(dx2*dx2 + dy2*dy2);
    if (len1 > 1e-9) { dx1 /= len1; dy1 /= len1; }
    if (len2 > 1e-9) { dx2 /= len2; dy2 /= len2; }
    
    // Project plate motion onto boundary normal (vector from plate1 to plate2)
    double normalX = dx2 - dx1;
    double normalY = dy2 - dy1;
    double normalLen = sqrt(normalX*normalX + normalY*normalY);
    if (normalLen > 1e-9) {
        normalX /= normalLen;
        normalY /= normalLen;
    }
    
    // Relative velocity along boundary normal
    double relativeMotion = (plate1.motionX - plate2.motionX) * normalX +
                           (plate1.motionY - plate2.motionY) * normalY;
    
    // Classify based on relative motion
    if (relativeMotion > 0.1) {
        return 1;   // Divergent (plates moving apart → ridge)
    } else if (relativeMotion < -0.1) {
        return -1;  // Convergent (plates colliding → trench)
    } else {
        return 0;   // Transform (sliding past each other → neutral)
    }
}

//============================================================
// Device: Find Nearest and Second Nearest Plates
//============================================================
__device__ void findNearestTwoPlates(double lon, double lat,
                                     const PlateSeed* __restrict__ plates,
                                     int numPlates,
                                     int& nearestId,
                                     int& secondNearestId,
                                     double& distToNearest,
                                     double& distToSecondNearest)
{
    double minDist = 1e9;
    double secondMinDist = 1e9;
    nearestId = 0;
    secondNearestId = 0;
    
    for (int i = 0; i < numPlates; ++i) {
        double dist = greatCircleDistance(lon, lat, plates[i].lon, plates[i].lat);
        
        if (dist < minDist) {
            secondMinDist = minDist;
            secondNearestId = nearestId;
            minDist = dist;
            nearestId = plates[i].id;
        } else if (dist < secondMinDist) {
            secondMinDist = dist;
            secondNearestId = plates[i].id;
        }
    }
    
    distToNearest = minDist;
    distToSecondNearest = secondMinDist;
}

//============================================================
// Device: Calculate Bathymetry Depth
//============================================================
__device__ double calculateBathymetry(double lon, double lat,
                                       const PlateSeed* __restrict__ plates,
                                       int numPlates,
                                       const BathymetryParams& params,
                                       int& basinId)
{
    // Find nearest two tectonic plates (Voronoi cell and neighbor)
    int nearestId, secondNearestId;
    double distToNearest, distToSecondNearest;
    findNearestTwoPlates(lon, lat, plates, numPlates,
                        nearestId, secondNearestId,
                        distToNearest, distToSecondNearest);
    
    basinId = nearestId;
    
    // Distance to plate boundary (Voronoi edge)
    double boundaryDist = distToSecondNearest - distToNearest;
    double distToBoundary = fabs(boundaryDist);
    
    // Classify boundary type based on plate motion vectors
    int boundaryType = classifyBoundary(plates[nearestId], plates[secondNearestId],
                                        lon, lat);
    
    // Normalize to world coordinates for Perlin
    double x = lon / M_PI;  // [-1, 1]
    double y = lat / (M_PI/2);  // [-1, 1]
    
    // Multi-octave Perlin noise for seafloor roughness
    double noise = fractalPerlin(x * 4.0, y * 4.0, params);
    
    // ================================================================
    // DEPTH CALCULATION - Enhanced Plate Tectonics
    // ================================================================
    
    double depth = params.abyssalDepth;  // Base depth (abyssal plain)
    
    // 1. CRUSTAL AGING - Depth increases with distance from nearest ridge
    // This is the dominant effect in real ocean basins
    double crustalAge = fmin(1.0, distToNearest / 0.8);  // 0-1 over ~5000 km
    double agingDepth = crustalAge * crustalAge * 1500.0;  // Quadratic deepening
    depth -= agingDepth;
    
    // 2. PLATE BOUNDARIES - Ridge/Trench/Transform based on motion vectors
    double boundaryWidth = (boundaryType == -1) ? 0.08 : 0.15;  // Trenches narrower than ridges
    double boundaryStrength = 1.0 - smoothstep(fmin(1.0, distToBoundary / boundaryWidth));
    
    if (boundaryType == 1) {
        // DIVERGENT BOUNDARY → MID-OCEAN RIDGE (shallow)
        depth += boundaryStrength * (params.ridgeDepth - params.abyssalDepth);
        
        // Add ridge roughness (segmented ridge)
        double ridgeSegmentation = sin(lon * 50.0 + lat * 30.0) * 200.0;
        depth += ridgeSegmentation * boundaryStrength;
        
    } else if (boundaryType == -1) {
        // CONVERGENT BOUNDARY → OCEANIC TRENCH (deep)
        depth += boundaryStrength * (params.trenchDepth - params.abyssalDepth);
        
    } else {
        // TRANSFORM BOUNDARY → Neutral depth, but add fracture zone roughness
        double fractureZoneRoughness = sin(lon * 80.0) * sin(lat * 60.0) * 300.0;
        depth += fractureZoneRoughness * boundaryStrength * 0.5;
    }
    
    // 3. MULTI-SCALE ROUGHNESS (Earth-like bathymetric detail)
    // Lower amplitude in deep basins, higher near ridges
    double noiseScale = 1.0 - crustalAge * 0.5;  // More variation near young crust
    
    // 3a. Base Perlin (large features: seamount chains, fracture zones)
    depth += noise * params.noiseAmplitude * noiseScale;
    
    // 3b. ABYSSAL HILL PROVINCES (10-20km wavelength, 50-200m height)
    // These cover ~80% of ocean floor on Earth
    double abyssalHills = fractalPerlin(x * 15.0, y * 15.0, params) * 150.0;
    depth += abyssalHills * (1.0 - boundaryStrength * 0.8);  // Reduce near boundaries
    
    // 3c. SEAMOUNT CLUSTERS (50-100km spacing, 300-800m height)
    // Volcanic features scattered across plates
    double seamounts = fractalPerlin(x * 8.0, y * 8.0, params);
    seamounts = fmax(0.0, seamounts);  // Only positive peaks
    depth += seamounts * seamounts * 500.0;  // Squared for sharp peaks
    
    // 3d. FINE-SCALE TEXTURE (1-5km features for rocky appearance)
    double fineTexture = fractalPerlin(x * 25.0, y * 25.0, params) * 80.0;
    depth += fineTexture;
    
    // 3e. RIDGE SEGMENTATION (makes ridges look fractured, not smooth)
    if (boundaryType == 1 && boundaryStrength > 0.3) {
        double ridgeRoughness = fractalPerlin(x * 18.0, y * 18.0, params) * 400.0;
        depth += ridgeRoughness * boundaryStrength;
    }
    
    // 4. ABYSSAL PLAIN SMOOTHING
    // Very old crust becomes smoother (sediment burial)
    if (crustalAge > 0.7) {
        double smoothing = (crustalAge - 0.7) / 0.3;  // 0-1 for old crust
        depth += smoothing * 200.0;  // Slight rise due to sediment accumulation
    }
    
    return depth;
}

//============================================================
// Kernel: Generate Bathymetry and Basin Map
//============================================================
__global__ void generateBathymetryKernel(float* __restrict__ bathymetry,
                                         int* __restrict__ basinMap,
                                         const PlateSeed* __restrict__ plates,
                                         int numPlates,
                                         BathymetryParams params,
                                         int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    
    int x = idx % width;
    int y = idx / width;
    
    // Convert pixel to lat/lon
    double lon = (x / (double)width) * 2.0 * M_PI - M_PI;  // [-π, π]
    double lat = (y / (double)height) * M_PI - M_PI/2;     // [-π/2, π/2]
    
    int basinId;
    double depth = calculateBathymetry(lon, lat, plates, numPlates, params, basinId);
    
    bathymetry[idx] = (float)depth;
    basinMap[idx] = basinId;
}

//============================================================
// Host: Generate Random Plate Seeds with Motion Vectors
//============================================================
std::vector<PlateSeed> generatePlateSeedsHost(int numPlates, unsigned long long seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> lonDist(-M_PI, M_PI);
    std::uniform_real_distribution<double> uniformDist(-1.0, 1.0);
    std::uniform_real_distribution<double> motionDist(-1.0, 1.0);  // Motion vector components
    
    std::vector<PlateSeed> plates(numPlates);
    
    std::cout << "🌍 Generating tectonic plates with motion vectors:\n";
    
    for (int i = 0; i < numPlates; ++i) {
        plates[i].lon = lonDist(rng);
        
        // Spherical uniform distribution (area-weighted sampling)
        // asin(uniform(-1,1)) gives equal area probability on sphere surface
        plates[i].lat = asin(uniformDist(rng));
        
        plates[i].id = i;
        
        // Random motion vector (simulates plate drift)
        plates[i].motionX = motionDist(rng);
        plates[i].motionY = motionDist(rng);
        
        // Normalize motion to unit vector
        double motionMag = sqrt(plates[i].motionX * plates[i].motionX +
                               plates[i].motionY * plates[i].motionY);
        if (motionMag > 1e-9) {
            plates[i].motionX /= motionMag;
            plates[i].motionY /= motionMag;
        }
        
        // Scale to realistic plate velocities (cm/year equivalent in normalized coords)
        double velocity = 0.5 + (rng() % 100) / 100.0;  // 0.5 to 1.5 relative units
        plates[i].motionX *= velocity;
        plates[i].motionY *= velocity;
        
        std::cout << "   Plate " << i << ": "
                  << "lon=" << (plates[i].lon * 180.0 / M_PI) << "°, "
                  << "lat=" << (plates[i].lat * 180.0 / M_PI) << "°, "
                  << "motion=(" << plates[i].motionX << ", " << plates[i].motionY << ")\n";
    }
    
    return plates;
}

//============================================================
// Host: Generate Bathymetry
//============================================================
void generateBathymetry(std::vector<float>& bathymetry,
                        std::vector<int>& basinMap,
                        const BathymetryParams& params)
{
    size_t totalPixels = WIDTH * HEIGHT;
    
    std::cout << "🌊 Generating bathymetry...\n";
    std::cout << "   Tectonic plates: " << params.numTectonicPlates << "\n";
    std::cout << "   Perlin octaves: " << params.perlinOctaves << "\n";
    std::cout << "   Seed: " << params.seed << "\n\n";
    
    // Generate plate seeds
    std::vector<PlateSeed> h_plates = generatePlateSeedsHost(params.numTectonicPlates, 
                                                             params.seed);
    
    // Allocate device memory
    float* d_bathymetry;
    int* d_basinMap;
    PlateSeed* d_plates;
    
    CUDA_CHECK(cudaMalloc(&d_bathymetry, totalPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_basinMap, totalPixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_plates, h_plates.size() * sizeof(PlateSeed)));
    
    CUDA_CHECK(cudaMemcpy(d_plates, h_plates.data(),
                         h_plates.size() * sizeof(PlateSeed),
                         cudaMemcpyHostToDevice));
    
    // Launch kernel
    int threads = 256;
    int blocks = (totalPixels + threads - 1) / threads;
    
    std::cout << "🚀 Launching CUDA kernel...\n";
    generateBathymetryKernel<<<blocks, threads>>>(d_bathymetry, d_basinMap, d_plates,
                                                  params.numTectonicPlates, params,
                                                  WIDTH, HEIGHT);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Download results
    bathymetry.resize(totalPixels);
    basinMap.resize(totalPixels);
    
    CUDA_CHECK(cudaMemcpy(bathymetry.data(), d_bathymetry,
                         totalPixels * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(basinMap.data(), d_basinMap,
                         totalPixels * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_bathymetry));
    CUDA_CHECK(cudaFree(d_basinMap));
    CUDA_CHECK(cudaFree(d_plates));
    
    std::cout << "   ✅ Bathymetry generated\n\n";
}

//============================================================
// Host: Save Bathymetry File (.BATH)
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
    header.channels = 1;  // Single channel: depth
    header.dtype = 1;     // Float32
    header.version = 1;
    header.depth = 1;     // No temporal dimension
    
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
// Host: Save Basin Map (.BASN)
//============================================================
bool saveBasinMap(const std::string& filename,
                  const std::vector<int>& basinMap)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Cannot open " << filename << "\n";
        return false;
    }
    
    // Create header
    ClimateFileHeader header;
    header.magic = BASN_MAGIC;
    header.width = WIDTH;
    header.height = HEIGHT;
    header.channels = 1;  // Single channel: basin ID
    header.dtype = 2;     // Int32
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
    file.write(reinterpret_cast<const char*>(basinMap.data()),
               basinMap.size() * sizeof(int));
    
    file.close();
    
    std::cout << "💾 Saved: " << filename << " ("
              << (28 + basinMap.size() * sizeof(int)) / (1024.0*1024.0) << " MB)\n";
    
    return true;
}

//============================================================
// Host: Compute Statistics
//============================================================
void computeStatistics(const std::vector<float>& bathymetry,
                       const std::vector<int>& basinMap,
                       int numPlates)
{
    std::cout << "📊 Bathymetry Statistics:\n";
    
    // Depth stats
    float minDepth = 1e9f;
    float maxDepth = -1e9f;
    double avgDepth = 0.0;
    
    // Depth distribution
    int ridgePixels = 0;
    int abyssalPixels = 0;
    int trenchPixels = 0;
    
    for (float d : bathymetry) {
        minDepth = fmin(minDepth, d);
        maxDepth = fmax(maxDepth, d);
        avgDepth += d;
        
        // Classify by depth
        if (d > -3000.0f) ridgePixels++;
        else if (d < -7000.0f) trenchPixels++;
        else abyssalPixels++;
    }
    avgDepth /= bathymetry.size();
    
    std::cout << "   Depth range: [" << (int)minDepth << "m, " << (int)maxDepth << "m]\n";
    std::cout << "   Average depth: " << (int)avgDepth << "m\n";
    std::cout << "   Ridge/shallow: " << (ridgePixels * 100.0 / bathymetry.size()) << "%\n";
    std::cout << "   Abyssal plains: " << (abyssalPixels * 100.0 / bathymetry.size()) << "%\n";
    std::cout << "   Trenches/deep: " << (trenchPixels * 100.0 / bathymetry.size()) << "%\n";
    
    // Basin coverage
    std::vector<int> basinCounts(numPlates, 0);
    for (int id : basinMap) {
        if (id >= 0 && id < numPlates) {
            basinCounts[id]++;
        }
    }
    
    std::cout << "\n📍 Ocean Basin Coverage:\n";
    for (int i = 0; i < numPlates; ++i) {
        double pct = (basinCounts[i] * 100.0) / basinMap.size();
        std::cout << "   Basin " << i << ": " << pct << "% of seafloor\n";
    }
    
    std::cout << "\n";
}

//============================================================
// Main
//============================================================
int main(int argc, char** argv)
{
    std::string bathyFile = "Bathymetry.bath";
    std::string basinFile = "OceanBasins.basn";
    
    // Default parameters (realistic Earth-like ocean world)
    BathymetryParams params;
    params.numTectonicPlates = 8;        // 8 major ocean basins
    params.ridgeDepth = -2500.0;         // Mid-ocean ridge depth
    params.abyssalDepth = -4500.0;       // Abyssal plain depth
    params.trenchDepth = -9000.0;        // Oceanic trench depth
    params.continentalShelf = -200.0;    // Shelf depth (unused in pure ocean world)
    params.noiseAmplitude = 800.0;       // Seamount/hill variation
    params.perlinOctaves = 5;            // Detail layers
    params.perlinLacunarity = 2.0;       // Frequency increase per octave
    params.perlinPersistence = 0.5;      // Amplitude decrease per octave
    params.seed = 42;                    // Random seed
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-o") == 0 && i+1 < argc)
            bathyFile = argv[++i];
        else if (strcmp(argv[i], "-b") == 0 && i+1 < argc)
            basinFile = argv[++i];
        else if (strcmp(argv[i], "--plates") == 0 && i+1 < argc)
            params.numTectonicPlates = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc)
            params.seed = atoll(argv[++i]);
        else if (strcmp(argv[i], "--octaves") == 0 && i+1 < argc)
            params.perlinOctaves = atoi(argv[++i]);
        else if (strcmp(argv[i], "--noise") == 0 && i+1 < argc)
            params.noiseAmplitude = atof(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -o <file>       Output bathymetry file (.bath)\n"
                      << "  -b <file>       Output basin map file (.basn)\n"
                      << "  --plates <n>    Number of tectonic plates (6-12)\n"
                      << "  --seed <n>      Random seed\n"
                      << "  --octaves <n>   Perlin octaves (4-6)\n"
                      << "  --noise <val>   Noise amplitude in meters\n";
            return 0;
        }
    }
    
    std::cout << "🌊 Bathymetry Generator\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "Resolution: " << WIDTH << " × " << HEIGHT << "\n";
    std::cout << "Output bathymetry: " << bathyFile << "\n";
    std::cout << "Output basins: " << basinFile << "\n\n";
    
    // Generate bathymetry
    std::vector<float> bathymetry;
    std::vector<int> basinMap;
    
    generateBathymetry(bathymetry, basinMap, params);
    
    // Statistics
    computeStatistics(bathymetry, basinMap, params.numTectonicPlates);
    
    // Save outputs
    if (!saveBathymetry(bathyFile, bathymetry)) {
        return EXIT_FAILURE;
    }
    
    if (!saveBasinMap(basinFile, basinMap)) {
        return EXIT_FAILURE;
    }
    
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "✅ Bathymetry generation complete!\n";
    
    return 0;
}
