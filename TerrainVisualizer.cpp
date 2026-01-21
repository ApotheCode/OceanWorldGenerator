//============================================================
// TerrainVisualizer.cpp
// Converts TerrainData.bin to PNG heightmap visualization
// Author: Mark Devereux (2025-10-18)
//============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

constexpr int WIDTH = 3600;
constexpr int HEIGHT = 1800;

//============================================================
// Color Palettes
//============================================================
struct RGB {
    unsigned char r, g, b;
};

// Terrain color ramp (bathymetry + topography)
RGB getTerrainColor(float elevation)
{
    RGB color;
    
    // Deep ocean: -11000m (dark blue) → -200m (light blue)
    if (elevation < -4000.0f) {
        color = {0, 0, 139};  // Dark blue
    }
    else if (elevation < -2000.0f) {
        float t = (elevation + 4000.0f) / 2000.0f;
        color.r = (unsigned char)(0 + t * 25);
        color.g = (unsigned char)(0 + t * 50);
        color.b = (unsigned char)(139 + t * 91);
    }
    else if (elevation < -200.0f) {
        float t = (elevation + 2000.0f) / 1800.0f;
        color.r = (unsigned char)(25 + t * 75);
        color.g = (unsigned char)(50 + t * 105);
        color.b = (unsigned char)(230 + t * 25);
    }
    // Shallow water: -200m → 0m (cyan to light cyan)
    else if (elevation < 0.0f) {
        float t = (elevation + 200.0f) / 200.0f;
        color.r = (unsigned char)(100 + t * 155);
        color.g = (unsigned char)(155 + t * 100);
        color.b = 255;
    }
    // Coast: 0m → 50m (sandy beach)
    else if (elevation < 50.0f) {
        float t = elevation / 50.0f;
        color.r = (unsigned char)(194 + t * 44);
        color.g = (unsigned char)(178 + t * 40);
        color.b = (unsigned char)(128 + t * 32);
    }
    // Lowlands: 50m → 500m (green)
    else if (elevation < 500.0f) {
        float t = (elevation - 50.0f) / 450.0f;
        color.r = (unsigned char)(34 + t * 100);
        color.g = (unsigned char)(139 + t * 80);
        color.b = (unsigned char)(34 - t * 20);
    }
    // Hills: 500m → 1500m (yellow-brown)
    else if (elevation < 1500.0f) {
        float t = (elevation - 500.0f) / 1000.0f;
        color.r = (unsigned char)(134 + t * 90);
        color.g = (unsigned char)(219 - t * 90);
        color.b = (unsigned char)(14 + t * 10);
    }
    // Mountains: 1500m → 3000m (brown to gray)
    else if (elevation < 3000.0f) {
        float t = (elevation - 1500.0f) / 1500.0f;
        color.r = (unsigned char)(160 - t * 50);
        color.g = (unsigned char)(82 + t * 28);
        color.b = (unsigned char)(45 + t * 45);
    }
    // High mountains: 3000m → 5000m (gray to white)
    else if (elevation < 5000.0f) {
        float t = (elevation - 3000.0f) / 2000.0f;
        color.r = (unsigned char)(110 + t * 100);
        color.g = (unsigned char)(110 + t * 100);
        color.b = (unsigned char)(90 + t * 120);
    }
    // Peaks: 5000m+ (white)
    else {
        float t = std::min(1.0f, (elevation - 5000.0f) / 3000.0f);
        color.r = (unsigned char)(210 + t * 45);
        color.g = (unsigned char)(210 + t * 45);
        color.b = (unsigned char)(210 + t * 45);
    }
    
    return color;
}

// Simple land/ocean mask (binary)
RGB getLandMaskColor(unsigned char isLand)
{
    if (isLand == 1) {
        return {139, 69, 19};  // Brown = land
    } else {
        return {30, 144, 255}; // Blue = ocean
    }
}

// Grayscale heightmap (for traditional DEM visualization)
RGB getGrayscaleColor(float elevation, float minElev, float maxElev)
{
    float normalized = (elevation - minElev) / (maxElev - minElev);
    normalized = std::max(0.0f, std::min(1.0f, normalized));
    unsigned char gray = (unsigned char)(normalized * 255);
    return {gray, gray, gray};
}

//============================================================
// Load Terrain Data
//============================================================
bool loadTerrainData(const std::string& filename,
                     std::vector<float>& elevation,
                     std::vector<unsigned char>& isLand,
                     std::vector<float>& oceanDepth,
                     std::vector<float>& continentality)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Failed to open: " << filename << "\n";
        return false;
    }
    
    int width, height;
    file.read(reinterpret_cast<char*>(&width), sizeof(int));
    file.read(reinterpret_cast<char*>(&height), sizeof(int));
    
    if (width != WIDTH || height != HEIGHT) {
        std::cerr << "❌ Dimension mismatch: " << width << "×" << height << "\n";
        return false;
    }
    
    size_t totalPixels = WIDTH * HEIGHT;
    
    elevation.resize(totalPixels);
    isLand.resize(totalPixels);
    oceanDepth.resize(totalPixels);
    continentality.resize(totalPixels);
    
    file.read(reinterpret_cast<char*>(elevation.data()), totalPixels * sizeof(float));
    file.read(reinterpret_cast<char*>(isLand.data()), totalPixels * sizeof(unsigned char));
    file.read(reinterpret_cast<char*>(oceanDepth.data()), totalPixels * sizeof(float));
    file.read(reinterpret_cast<char*>(continentality.data()), totalPixels * sizeof(float));
    
    file.close();
    
    // Statistics
    float minElev = *std::min_element(elevation.begin(), elevation.end());
    float maxElev = *std::max_element(elevation.begin(), elevation.end());
    size_t landCount = std::count(isLand.begin(), isLand.end(), 1);
    
    std::cout << "✅ Loaded terrain data:\n";
    std::cout << "   Dimensions: " << width << "×" << height << "\n";
    std::cout << "   Elevation range: " << minElev << "m to " << maxElev << "m\n";
    std::cout << "   Land coverage: " << (landCount * 100.0 / totalPixels) << "%\n\n";
    
    return true;
}

//============================================================
// Export PNG
//============================================================
bool exportPNG(const std::string& filename,
               const std::vector<RGB>& image)
{
    int result = stbi_write_png(filename.c_str(), WIDTH, HEIGHT, 3,
                                 image.data(), WIDTH * 3);
    
    if (result) {
        std::cout << "✅ Saved: " << filename << "\n";
        return true;
    } else {
        std::cerr << "❌ Failed to write: " << filename << "\n";
        return false;
    }
}

//============================================================
// Main
//============================================================
int main(int argc, char** argv)
{
    std::string inputFile = "output/TerrainData.bin";
    std::string outputPrefix = "output/Terrain";
    
    if (argc >= 2) inputFile = argv[1];
    if (argc >= 3) outputPrefix = argv[2];
    
    std::cout << "🗺️  Terrain Visualizer\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "Input:  " << inputFile << "\n";
    std::cout << "Output: " << outputPrefix << "_*.png\n\n";
    
    // Load data
    std::vector<float> elevation;
    std::vector<unsigned char> isLand;
    std::vector<float> oceanDepth;
    std::vector<float> continentality;
    
    if (!loadTerrainData(inputFile, elevation, isLand, oceanDepth, continentality)) {
        return EXIT_FAILURE;
    }
    
    size_t totalPixels = WIDTH * HEIGHT;
    
    // Find elevation range for grayscale
    float minElev = *std::min_element(elevation.begin(), elevation.end());
    float maxElev = *std::max_element(elevation.begin(), elevation.end());
    
    // Generate visualizations
    std::cout << "🎨 Generating visualizations...\n";
    
    // 1. Color heightmap (bathymetry + topography)
    std::vector<RGB> colorMap(totalPixels);
    for (size_t i = 0; i < totalPixels; ++i) {
        colorMap[i] = getTerrainColor(elevation[i]);
    }
    exportPNG(outputPrefix + "_Color.png", colorMap);
    
    // 2. Land/ocean mask
    std::vector<RGB> maskMap(totalPixels);
    for (size_t i = 0; i < totalPixels; ++i) {
        maskMap[i] = getLandMaskColor(isLand[i]);
    }
    exportPNG(outputPrefix + "_Mask.png", maskMap);
    
    // 3. Grayscale heightmap
    std::vector<RGB> grayMap(totalPixels);
    for (size_t i = 0; i < totalPixels; ++i) {
        grayMap[i] = getGrayscaleColor(elevation[i], minElev, maxElev);
    }
    exportPNG(outputPrefix + "_Grayscale.png", grayMap);
    
    // 4. Ocean depth visualization
    std::vector<RGB> depthMap(totalPixels);
    float maxDepth = *std::max_element(oceanDepth.begin(), oceanDepth.end());
    for (size_t i = 0; i < totalPixels; ++i) {
        if (isLand[i] == 0) {
            // Ocean: darker = deeper
            float t = 1.0f - std::min(1.0f, oceanDepth[i] / 6000.0f);
            depthMap[i].r = (unsigned char)(t * 100);
            depthMap[i].g = (unsigned char)(t * 150);
            depthMap[i].b = (unsigned char)(t * 255);
        } else {
            depthMap[i] = {50, 50, 50};  // Land = gray
        }
    }
    exportPNG(outputPrefix + "_OceanDepth.png", depthMap);
    
    // 5. Continentality visualization
    std::vector<RGB> contMap(totalPixels);
    float maxCont = 5000.0f;  // Max distance in km
    for (size_t i = 0; i < totalPixels; ++i) {
        float t = std::min(1.0f, continentality[i] / maxCont);
        if (isLand[i] == 1) {
            // Land: green (coast) to brown (interior)
            contMap[i].r = (unsigned char)(100 + t * 100);
            contMap[i].g = (unsigned char)(200 - t * 130);
            contMap[i].b = (unsigned char)(100 - t * 80);
        } else {
            // Ocean: light blue (coast) to dark blue (far)
            contMap[i].r = (unsigned char)(100 - t * 80);
            contMap[i].g = (unsigned char)(200 - t * 100);
            contMap[i].b = 255;
        }
    }
    exportPNG(outputPrefix + "_Continentality.png", contMap);
    
    std::cout << "\n══════════════════════════════════════════════════════════\n";
    std::cout << "✅ Generated 5 visualizations:\n";
    std::cout << "   1. " << outputPrefix << "_Color.png (full terrain)\n";
    std::cout << "   2. " << outputPrefix << "_Mask.png (land/ocean)\n";
    std::cout << "   3. " << outputPrefix << "_Grayscale.png (traditional DEM)\n";
    std::cout << "   4. " << outputPrefix << "_OceanDepth.png (bathymetry)\n";
    std::cout << "   5. " << outputPrefix << "_Continentality.png (distance to coast)\n";
    
    return 0;
}