//============================================================
// PNGToBathymetry.cpp
// Converts PNG heightmap to .BATH bathymetry file
// 
// Gaea format: Grayscale PNG where:
//   - Black (0) = Deepest (e.g., -11000m)
//   - White (255 or 65535) = Shallowest (e.g., -2000m)
// 
// Supports 8-bit and 16-bit PNGs
// 
// Compile: g++ -O3 PNGToBathymetry.cpp -o PNGToBathymetry
// Usage: ./PNGToBathymetry input.png output.bath --deep -11000 --shallow -2000
//============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "ClimateFileFormat.h"

//============================================================
// Convert PNG to Bathymetry
//============================================================
bool convertPNGToBathymetry(const std::string& pngFile,
                            const std::string& bathFile,
                            double deepDepth,
                            double shallowDepth)
{
    std::cout << "📖 Loading PNG: " << pngFile << "\n";
    
    int width, height, channels;
    int bitDepth = 0;
    
    // Try loading as 16-bit first
    uint16_t* data16 = stbi_load_16(pngFile.c_str(), &width, &height, &channels, 1);
    uint8_t* data8 = nullptr;
    
    bool is16bit = false;
    
    if (data16) {
        is16bit = true;
        bitDepth = 16;
        std::cout << "   Format: 16-bit grayscale\n";
    } else {
        // Try 8-bit
        data8 = stbi_load(pngFile.c_str(), &width, &height, &channels, 1);
        if (!data8) {
            std::cerr << "❌ Failed to load PNG: " << stbi_failure_reason() << "\n";
            return false;
        }
        bitDepth = 8;
        std::cout << "   Format: 8-bit grayscale\n";
    }
    
    std::cout << "   Resolution: " << width << " × " << height << "\n";
    std::cout << "   Bit depth: " << bitDepth << "-bit\n\n";
    
    // Convert to bathymetry
    std::cout << "🔄 Converting to bathymetry...\n";
    std::cout << "   Depth range: [" << (int)deepDepth << "m, " << (int)shallowDepth << "m]\n";
    
    size_t totalPixels = width * height;
    std::vector<float> bathymetry(totalPixels);
    
    double depthRange = shallowDepth - deepDepth;
    
    if (is16bit) {
        // 16-bit: 0-65535
        for (size_t i = 0; i < totalPixels; ++i) {
            double t = data16[i] / 65535.0;  // Normalize to [0, 1]
            bathymetry[i] = (float)(deepDepth + t * depthRange);
        }
        stbi_image_free(data16);
    } else {
        // 8-bit: 0-255
        for (size_t i = 0; i < totalPixels; ++i) {
            double t = data8[i] / 255.0;  // Normalize to [0, 1]
            bathymetry[i] = (float)(deepDepth + t * depthRange);
        }
        stbi_image_free(data8);
    }
    
    // Compute statistics
    float minDepth = bathymetry[0];
    float maxDepth = bathymetry[0];
    double avgDepth = 0.0;
    
    for (float d : bathymetry) {
        minDepth = std::min(minDepth, d);
        maxDepth = std::max(maxDepth, d);
        avgDepth += d;
    }
    avgDepth /= bathymetry.size();
    
    std::cout << "\n📊 Bathymetry Statistics:\n";
    std::cout << "   Depth range: [" << (int)minDepth << "m, " << (int)maxDepth << "m]\n";
    std::cout << "   Average depth: " << (int)avgDepth << "m\n\n";
    
    // Save as .BATH file
    std::cout << "💾 Saving: " << bathFile << "\n";
    
    std::ofstream file(bathFile, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Cannot open output file\n";
        return false;
    }
    
    // Write header
    ClimateFileHeader header;
    header.magic = BATH_MAGIC;
    header.width = width;
    header.height = height;
    header.channels = 1;
    header.dtype = 1;  // Float32
    header.version = 1;
    header.depth = 1;
    
    writeClimateHeader(file, header);
    
    // Write data
    file.write(reinterpret_cast<const char*>(bathymetry.data()),
               bathymetry.size() * sizeof(float));
    
    file.close();
    
    size_t fileSize = 28 + bathymetry.size() * sizeof(float);
    std::cout << "   Size: " << (fileSize / (1024.0*1024.0)) << " MB\n";
    
    return true;
}

//============================================================
// Main
//============================================================
int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cout << "PNG to Bathymetry Converter\n";
        std::cout << "══════════════════════════════════════════════════════════\n\n";
        std::cout << "Usage: " << argv[0] << " <input.png> <output.bath> [options]\n\n";
        std::cout << "Options:\n";
        std::cout << "  --deep <m>      Depth for black/0 (default -11000)\n";
        std::cout << "  --shallow <m>   Depth for white/max (default -2000)\n\n";
        std::cout << "Examples:\n";
        std::cout << "  " << argv[0] << " bathymetry.png output.bath\n";
        std::cout << "  " << argv[0] << " terrain.png output.bath --deep -8000 --shallow -1000\n\n";
        std::cout << "PNG Format:\n";
        std::cout << "  - Grayscale (8-bit or 16-bit)\n";
        std::cout << "  - Black (0) = Deepest ocean\n";
        std::cout << "  - White (255/65535) = Shallowest\n";
        std::cout << "  - Any resolution (will preserve dimensions)\n";
        return 1;
    }
    
    std::string inputPng = argv[1];
    std::string outputBath = argv[2];
    
    double deepDepth = -11000.0;    // Black = deepest (Mariana Trench depth)
    double shallowDepth = -2000.0;  // White = shallow (continental shelf)
    
    // Parse options
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--deep") == 0 && i+1 < argc) {
            deepDepth = atof(argv[++i]);
        } else if (strcmp(argv[i], "--shallow") == 0 && i+1 < argc) {
            shallowDepth = atof(argv[++i]);
        }
    }
    
    std::cout << "🌊 PNG to Bathymetry Converter\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "Input:  " << inputPng << "\n";
    std::cout << "Output: " << outputBath << "\n\n";
    
    if (!convertPNGToBathymetry(inputPng, outputBath, deepDepth, shallowDepth)) {
        return EXIT_FAILURE;
    }
    
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "✅ Conversion complete!\n\n";
    std::cout << "Next steps:\n";
    std::cout << "  1. Visualize: ./BathymetryViewer " << outputBath << "\n";
    std::cout << "  2. Use in gyre generation pipeline\n";
    
    return 0;
}
