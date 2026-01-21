//============================================================
// BathymetryViewer.cpp
// Reads .bath and .basn files and outputs PNG images
// 
// Requires: stb_image_write.h (already in your project)
// 
// Compile:
//   g++ -O3 BathymetryViewer.cpp -o BathymetryViewer
// 
// Usage:
//   ./BathymetryViewer Bathymetry.bath [OceanBasins.basn]
//============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstring>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//============================================================
// File Header Structure
//============================================================
struct FileHeader {
    uint32_t magic;
    int32_t width;
    int32_t height;
    int32_t channels;
    int32_t dtype;
    int32_t version;
    int32_t depth;
};

//============================================================
// Color Mapping Functions
//============================================================
struct RGB {
    uint8_t r, g, b;
};

// Bathymetric colormap: deep blue -> cyan for depth
RGB bathymetricColormap(float depth, float minDepth, float maxDepth)
{
    // Normalize depth to [0, 1]
    float t = (depth - minDepth) / (maxDepth - minDepth);
    t = std::max(0.0f, std::min(1.0f, t));
    
    RGB color;
    
    // Deep trench (0.0) -> Dark navy blue
    // Abyssal plain (0.3) -> Deep blue
    // Mid depth (0.5) -> Medium blue
    // Shallow (0.7) -> Light blue
    // Ridge (1.0) -> Cyan
    
    if (t < 0.25f) {
        // Deep trench -> Abyssal plain (dark navy -> deep blue)
        float local = t / 0.25f;
        color.r = (uint8_t)(0 + local * 0);
        color.g = (uint8_t)(0 + local * 51);
        color.b = (uint8_t)(51 + local * 51);
    } else if (t < 0.5f) {
        // Abyssal plain -> Mid depth (deep blue -> medium blue)
        float local = (t - 0.25f) / 0.25f;
        color.r = (uint8_t)(0 + local * 0);
        color.g = (uint8_t)(51 + local * 51);
        color.b = (uint8_t)(102 + local * 102);
    } else if (t < 0.75f) {
        // Mid depth -> Shallow (medium blue -> light blue)
        float local = (t - 0.5f) / 0.25f;
        color.r = (uint8_t)(0 + local * 51);
        color.g = (uint8_t)(102 + local * 51);
        color.b = (uint8_t)(204 + local * 51);
    } else {
        // Shallow -> Ridge (light blue -> cyan)
        float local = (t - 0.75f) / 0.25f;
        color.r = (uint8_t)(51 + local * 153);
        color.g = (uint8_t)(153 + local * 102);
        color.b = (uint8_t)(255);
    }
    
    return color;
}

// Basin colormap: distinct colors per basin
RGB basinColormap(int basinId, int numBasins)
{
    // Use HSV color wheel for distinct basin colors
    float hue = (basinId * 360.0f) / numBasins;
    float sat = 0.8f;
    float val = 0.9f;
    
    // HSV to RGB conversion
    float c = val * sat;
    float x = c * (1.0f - fabs(fmod(hue / 60.0f, 2.0f) - 1.0f));
    float m = val - c;
    
    float r, g, b;
    
    if (hue < 60) {
        r = c; g = x; b = 0;
    } else if (hue < 120) {
        r = x; g = c; b = 0;
    } else if (hue < 180) {
        r = 0; g = c; b = x;
    } else if (hue < 240) {
        r = 0; g = x; b = c;
    } else if (hue < 300) {
        r = x; g = 0; b = c;
    } else {
        r = c; g = 0; b = x;
    }
    
    RGB color;
    color.r = (uint8_t)((r + m) * 255);
    color.g = (uint8_t)((g + m) * 255);
    color.b = (uint8_t)((b + m) * 255);
    
    return color;
}

//============================================================
// Write PNG Image using stb_image_write
//============================================================
bool writePNG(const std::string& filename, 
              const std::vector<uint8_t>& pixels,
              int width, int height)
{
    // stbi_write_png returns 0 on failure, non-zero on success
    int result = stbi_write_png(filename.c_str(), width, height, 
                                3, pixels.data(), width * 3);
    
    if (!result) {
        std::cerr << "❌ Failed to write " << filename << "\n";
        return false;
    }
    
    return true;
}

//============================================================
// Read Binary File
//============================================================
bool readBathymetryFile(const std::string& filename, 
                        std::vector<float>& data,
                        FileHeader& header)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Cannot open " << filename << "\n";
        return false;
    }
    
    // Read header (28 bytes)
    file.read(reinterpret_cast<char*>(&header.magic), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&header.width), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&header.height), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&header.channels), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&header.dtype), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&header.version), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&header.depth), sizeof(int32_t));
    
    std::cout << "📖 Reading " << filename << "...\n";
    std::cout << "   Magic: 0x" << std::hex << header.magic << std::dec << "\n";
    std::cout << "   Resolution: " << header.width << " × " << header.height << "\n";
    
    // Read data
    size_t numPixels = header.width * header.height;
    data.resize(numPixels);
    file.read(reinterpret_cast<char*>(data.data()), numPixels * sizeof(float));
    
    file.close();
    return true;
}

bool readBasinFile(const std::string& filename,
                   std::vector<int>& data,
                   FileHeader& header)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Cannot open " << filename << "\n";
        return false;
    }
    
    // Read header (28 bytes)
    file.read(reinterpret_cast<char*>(&header.magic), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&header.width), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&header.height), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&header.channels), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&header.dtype), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&header.version), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&header.depth), sizeof(int32_t));
    
    std::cout << "📖 Reading " << filename << "...\n";
    std::cout << "   Magic: 0x" << std::hex << header.magic << std::dec << "\n";
    std::cout << "   Resolution: " << header.width << " × " << header.height << "\n";
    
    // Read data
    size_t numPixels = header.width * header.height;
    data.resize(numPixels);
    file.read(reinterpret_cast<char*>(data.data()), numPixels * sizeof(int));
    
    file.close();
    return true;
}

//============================================================
// Visualize Bathymetry as PPM
//============================================================
bool visualizeBathymetry(const std::string& bathFile,
                         const std::string& outputFile)
{
    std::vector<float> bathymetry;
    FileHeader header;
    
    if (!readBathymetryFile(bathFile, bathymetry, header)) {
        return false;
    }
    
    // Compute statistics
    float minDepth = 1e9f;
    float maxDepth = -1e9f;
    double avgDepth = 0.0;
    
    for (float d : bathymetry) {
        minDepth = std::min(minDepth, d);
        maxDepth = std::max(maxDepth, d);
        avgDepth += d;
    }
    avgDepth /= bathymetry.size();
    
    std::cout << "   Depth range: [" << (int)minDepth << "m, " << (int)maxDepth << "m]\n";
    std::cout << "   Average depth: " << (int)avgDepth << "m\n\n";
    
    // Create RGB image
    std::vector<uint8_t> image(header.width * header.height * 3);
    
    for (size_t i = 0; i < bathymetry.size(); ++i) {
        RGB color = bathymetricColormap(bathymetry[i], minDepth, maxDepth);
        image[i * 3 + 0] = color.r;
        image[i * 3 + 1] = color.g;
        image[i * 3 + 2] = color.b;
    }
    
    // Write PNG
    if (!writePNG(outputFile, image, header.width, header.height)) {
        return false;
    }
    
    std::cout << "💾 Saved: " << outputFile << " ("
              << (image.size() / (1024.0*1024.0)) << " MB uncompressed)\n";
    return true;
}

//============================================================
// Visualize Basin Map as PNG
//============================================================
bool visualizeBasins(const std::string& basinFile,
                     const std::string& outputFile)
{
    std::vector<int> basinMap;
    FileHeader header;
    
    if (!readBasinFile(basinFile, basinMap, header)) {
        return false;
    }
    
    // Find max basin ID
    int maxBasinId = 0;
    for (int id : basinMap) {
        maxBasinId = std::max(maxBasinId, id);
    }
    
    std::cout << "   Number of basins: " << (maxBasinId + 1) << "\n";
    
    // Count basin coverage
    std::vector<int> basinCounts(maxBasinId + 1, 0);
    for (int id : basinMap) {
        if (id >= 0 && id <= maxBasinId) {
            basinCounts[id]++;
        }
    }
    
    std::cout << "\n📊 Basin Coverage:\n";
    for (int i = 0; i <= maxBasinId; ++i) {
        double pct = (basinCounts[i] * 100.0) / basinMap.size();
        std::cout << "   Basin " << i << ": " << pct << "%\n";
    }
    std::cout << "\n";
    
    // Create RGB image
    std::vector<uint8_t> image(header.width * header.height * 3);
    
    for (size_t i = 0; i < basinMap.size(); ++i) {
        RGB color = basinColormap(basinMap[i], maxBasinId + 1);
        image[i * 3 + 0] = color.r;
        image[i * 3 + 1] = color.g;
        image[i * 3 + 2] = color.b;
    }
    
    // Write PNG
    if (!writePNG(outputFile, image, header.width, header.height)) {
        return false;
    }
    
    std::cout << "💾 Saved: " << outputFile << " ("
              << (image.size() / (1024.0*1024.0)) << " MB uncompressed)\n";
    return true;
}

//============================================================
// Main
//============================================================
int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <bathymetry.bath> [basins.basn]\n\n";
        std::cout << "Outputs PNG images\n\n";
        std::cout << "Example:\n";
        std::cout << "  " << argv[0] << " Bathymetry.bath OceanBasins.basn\n";
        return 1;
    }
    
    std::string bathFile = argv[1];
    std::string basinFile = (argc > 2) ? argv[2] : "";
    
    std::cout << "🌊 Bathymetry Viewer\n";
    std::cout << "══════════════════════════════════════════════════════════\n\n";
    
    // Visualize bathymetry
    std::string bathOutput = bathFile;
    size_t dotPos = bathOutput.find_last_of('.');
    if (dotPos != std::string::npos) {
        bathOutput = bathOutput.substr(0, dotPos);
    }
    bathOutput += "_depth.png";
    
    if (!visualizeBathymetry(bathFile, bathOutput)) {
        return EXIT_FAILURE;
    }
    
    // Visualize basins if provided
    if (!basinFile.empty()) {
        std::string basinOutput = basinFile;
        dotPos = basinOutput.find_last_of('.');
        if (dotPos != std::string::npos) {
            basinOutput = basinOutput.substr(0, dotPos);
        }
        basinOutput += "_basins.png";
        
        if (!visualizeBasins(basinFile, basinOutput)) {
            return EXIT_FAILURE;
        }
    }
    
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "✅ Visualization complete!\n";
    
    return 0;
}
