//============================================================
// BiomeMapExporter.cpp
// Converts binary biome maps to PNG with color visualization
// ------------------------------------------------------------
// Reads BiomeMap_Ocean.bin and exports to PNG format
// Uses stb_image_write for PNG encoding (header-only library)
//
// Author: Mark Devereux
// Date: 2025-10-17
//============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>

// stb_image_write: https://github.com/nothings/stb
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

constexpr size_t WIDTH  = 3600;
constexpr size_t HEIGHT = 1800;
constexpr int CHANNELS  = 3;  // RGB

//============================================================
// Color Palette for Ocean Biomes
//============================================================
struct RGB {
    uint8_t r, g, b;
};

// Beautiful ocean color scheme (inspired by bathymetric maps)
const RGB OCEAN_PALETTE[6] = {
    {230, 242, 255},  // 0: Frozen Ocean   (pale ice blue)
    {148, 197, 232},  // 1: Cold Ocean     (light blue)
    {74, 144, 217},   // 2: Temperate Ocean (medium blue)
    {21, 101, 192},   // 3: Warm Ocean     (deep blue)
    {13, 59, 102},    // 4: Tropical Ocean (navy)
    {128, 0, 128}     // 255: No-data/Invalid (magenta for debugging)
};

//============================================================
// Load Binary Biome Map
//============================================================
bool loadBiomeMap(const std::string& filename, std::vector<uint8_t>& map)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "❌ Failed to open file: " << filename << std::endl;
        return false;
    }

    // Validate file size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t expectedSize = WIDTH * HEIGHT;
    if (fileSize != expectedSize)
    {
        std::cerr << "❌ File size mismatch!\n";
        std::cerr << "   Expected: " << expectedSize << " bytes\n";
        std::cerr << "   Found:    " << fileSize << " bytes\n";
        return false;
    }

    map.resize(expectedSize);
    file.read(reinterpret_cast<char*>(map.data()), expectedSize);

    if (!file)
    {
        std::cerr << "❌ Failed to read complete file.\n";
        return false;
    }

    file.close();
    return true;
}

//============================================================
// Convert Biome Map to RGB Image
//============================================================
void biomeMapToRGB(const std::vector<uint8_t>& biomeMap, std::vector<uint8_t>& rgbImage)
{
    size_t totalPixels = WIDTH * HEIGHT;
    rgbImage.resize(totalPixels * CHANNELS);

    for (size_t i = 0; i < totalPixels; ++i)
    {
        uint8_t biomeID = biomeMap[i];
        
        // Select color from palette
        RGB color;
        if (biomeID <= 4)
            color = OCEAN_PALETTE[biomeID];
        else
            color = OCEAN_PALETTE[5];  // No-data (magenta)

        // Write RGB values
        rgbImage[i * CHANNELS + 0] = color.r;
        rgbImage[i * CHANNELS + 1] = color.g;
        rgbImage[i * CHANNELS + 2] = color.b;
    }
}

//============================================================
// Export to PNG with Legend
//============================================================
bool exportToPNG(const std::vector<uint8_t>& rgbImage, const std::string& filename)
{
    // stb_image_write expects row-major order (which we already have)
    int result = stbi_write_png(filename.c_str(), WIDTH, HEIGHT, CHANNELS,
                                 rgbImage.data(), WIDTH * CHANNELS);
    
    if (!result)
    {
        std::cerr << "❌ Failed to write PNG: " << filename << std::endl;
        return false;
    }

    return true;
}

//============================================================
// Create Legend Image
//============================================================
bool createLegend(const std::string& filename)
{
    const int LEGEND_WIDTH = 400;
    const int LEGEND_HEIGHT = 250;
    const int LEGEND_CHANNELS = 3;
    
    std::vector<uint8_t> legendImage(LEGEND_WIDTH * LEGEND_HEIGHT * LEGEND_CHANNELS, 255);
    
    // Fill background with light gray
    for (size_t i = 0; i < LEGEND_WIDTH * LEGEND_HEIGHT; ++i)
    {
        legendImage[i * LEGEND_CHANNELS + 0] = 245;
        legendImage[i * LEGEND_CHANNELS + 1] = 245;
        legendImage[i * LEGEND_CHANNELS + 2] = 245;
    }
    
    // Draw color swatches (simple rectangular blocks)
    const int SWATCH_HEIGHT = 30;
    const int SWATCH_WIDTH = 60;
    const int START_Y = 40;
    const int START_X = 30;
    const int SPACING = 40;
    
    const char* labels[5] = {
        "Frozen Ocean",
        "Cold Ocean",
        "Temperate Ocean",
        "Warm Ocean",
        "Tropical Ocean"
    };
    
    for (int biome = 0; biome < 5; ++biome)
    {
        int y_offset = START_Y + biome * SPACING;
        RGB color = OCEAN_PALETTE[biome];
        
        // Draw swatch
        for (int y = 0; y < SWATCH_HEIGHT; ++y)
        {
            for (int x = 0; x < SWATCH_WIDTH; ++x)
            {
                int py = y_offset + y;
                int px = START_X + x;
                if (py < LEGEND_HEIGHT && px < LEGEND_WIDTH)
                {
                    size_t idx = (py * LEGEND_WIDTH + px) * LEGEND_CHANNELS;
                    legendImage[idx + 0] = color.r;
                    legendImage[idx + 1] = color.g;
                    legendImage[idx + 2] = color.b;
                }
            }
        }
        
        // Note: Text rendering requires a font library (FreeType, etc.)
        // For simplicity, we're just creating color swatches
        // Users can add labels in image editing software or we can print them to console
    }
    
    int result = stbi_write_png(filename.c_str(), LEGEND_WIDTH, LEGEND_HEIGHT, 
                                 LEGEND_CHANNELS, legendImage.data(), 
                                 LEGEND_WIDTH * LEGEND_CHANNELS);
    
    return result != 0;
}

//============================================================
// Compute Statistics
//============================================================
void printStatistics(const std::vector<uint8_t>& biomeMap)
{
    std::vector<size_t> counts(6, 0);
    
    for (auto biome : biomeMap)
    {
        if (biome <= 4)
            counts[biome]++;
        else
            counts[5]++;  // No-data
    }
    
    size_t totalCells = WIDTH * HEIGHT;
    
    std::cout << "\n📊 Biome Statistics:\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    
    const char* names[5] = {
        "Frozen Ocean   ",
        "Cold Ocean     ",
        "Temperate Ocean",
        "Warm Ocean     ",
        "Tropical Ocean "
    };
    
    for (int i = 0; i < 5; ++i)
    {
        double percent = (counts[i] * 100.0) / totalCells;
        std::cout << "  " << i << " = " << names[i] << ": "
                  << counts[i] << " cells (" << percent << "%)\n";
    }
    
    if (counts[5] > 0)
    {
        double percent = (counts[5] * 100.0) / totalCells;
        std::cout << "  ⚠️  No-data/Invalid: " << counts[5] 
                  << " cells (" << percent << "%)\n";
    }
}

//============================================================
// Main Function
//============================================================
int main(int argc, char** argv)
{
    std::string inputFile  = "BiomeMap_Ocean.bin";
    std::string outputFile = "BiomeMap_Ocean.png";
    std::string legendFile = "BiomeMap_Legend.png";
    bool createLegendImage = true;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-i") == 0 && i+1 < argc)
            inputFile = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc)
            outputFile = argv[++i];
        else if (strcmp(argv[i], "--no-legend") == 0)
            createLegendImage = false;
        else if (strcmp(argv[i], "--help") == 0)
        {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -i <file>      Input biome map (default: BiomeMap_Ocean.bin)\n"
                      << "  -o <file>      Output PNG file (default: BiomeMap_Ocean.png)\n"
                      << "  --no-legend    Don't create legend image\n"
                      << "  --help         Show this help message\n";
            return 0;
        }
    }

    std::cout << "🖼️  Biome Map PNG Exporter\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << "Input:  " << inputFile << "\n";
    std::cout << "Output: " << outputFile << "\n\n";

    // Load biome map
    std::cout << "📂 Loading biome map...\n";
    std::vector<uint8_t> biomeMap;
    if (!loadBiomeMap(inputFile, biomeMap))
        return EXIT_FAILURE;

    std::cout << "✅ Loaded " << biomeMap.size() << " cells\n";

    // Print statistics
    printStatistics(biomeMap);

    // Convert to RGB
    std::cout << "\n🎨 Converting to RGB...\n";
    std::vector<uint8_t> rgbImage;
    biomeMapToRGB(biomeMap, rgbImage);

    // Export to PNG
    std::cout << "💾 Exporting to PNG...\n";
    if (!exportToPNG(rgbImage, outputFile))
        return EXIT_FAILURE;

    std::cout << "✅ Exported: " << outputFile << "\n";
    std::cout << "   Resolution: " << WIDTH << "x" << HEIGHT << " pixels\n";
    std::cout << "   File size: ~" << (rgbImage.size() / (1024.0 * 1024.0)) << " MB (uncompressed)\n";

    // Create legend
    if (createLegendImage)
    {
        std::cout << "\n🏷️  Creating legend...\n";
        if (createLegend(legendFile))
        {
            std::cout << "✅ Legend saved: " << legendFile << "\n";
        }
        else
        {
            std::cerr << "⚠️  Failed to create legend (non-fatal)\n";
        }
    }

    std::cout << "\n📋 Color Legend:\n";
    std::cout << "   Frozen Ocean    (0): Pale Ice Blue   #E6F2FF\n";
    std::cout << "   Cold Ocean      (1): Light Blue      #94C5E8\n";
    std::cout << "   Temperate Ocean (2): Medium Blue     #4A90D9\n";
    std::cout << "   Warm Ocean      (3): Deep Blue       #1565C0\n";
    std::cout << "   Tropical Ocean  (4): Navy Blue       #0D3B66\n";
    std::cout << "   No-data/Invalid   : Magenta         #800080\n";

    std::cout << "\n✅ Export complete!\n";
    std::cout << "═══════════════════════════════════════════════════════\n";

    return 0;
}