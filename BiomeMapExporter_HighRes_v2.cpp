//============================================================
// BiomeMapExporter_HighRes.cpp
// Exports 20-zone biome maps with gradient visualization
// Shows PNPL circulation patterns in oceans
//============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include "ClimateFileFormat.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

constexpr size_t WIDTH  = STANDARD_WIDTH;
constexpr size_t HEIGHT = STANDARD_HEIGHT;
constexpr int CHANNELS  = 3;

struct RGB {
    uint8_t r, g, b;
};

//============================================================
// 20-Color Ocean Gradient (Cold to Warm)
//============================================================
RGB getOceanColor(uint8_t zone)
{
    // Smooth blue gradient: pale ice blue → navy
    const RGB oceanGradient[20] = {
        {240, 248, 255}, {230, 242, 255}, {220, 236, 255}, {210, 230, 255},
        {200, 224, 255}, {190, 218, 255}, {180, 212, 255}, {170, 206, 255},
        {160, 200, 255}, {150, 194, 255}, {140, 188, 255}, {130, 182, 255},
        {120, 176, 255}, {110, 170, 255}, {100, 164, 255}, {90, 158, 255},
        {80, 152, 255}, {70, 146, 255}, {60, 140, 255}, {50, 134, 255}
    };
    
    if (zone < 20) return oceanGradient[zone];
    return {255, 0, 255};  // Magenta for errors
}

//============================================================
// 20-Color Land Gradient (Cold to Warm)
//============================================================
RGB getLandColor(uint8_t zone)
{
    // White (ice) → gray-blue (tundra) → green (temperate) → dark green (tropical)
    const RGB landGradient[20] = {
        {255, 255, 255}, {245, 245, 245}, {232, 244, 248}, {208, 232, 240},
        {184, 220, 232}, {160, 208, 224}, {144, 200, 176}, {128, 192, 144},
        {112, 184, 112}, {96, 176, 96}, {80, 168, 80}, {128, 160, 64},
        {160, 152, 48}, {192, 144, 32}, {208, 136, 16}, {224, 128, 0},
        {96, 192, 48}, {80, 176, 32}, {64, 160, 16}, {48, 128, 0}
    };
    
    if (zone < 20) return landGradient[zone];
    return {255, 0, 255};
}

//============================================================
// Load Land Mask
//============================================================
bool loadLandMask(const std::string& filename, std::vector<uint8_t>& mask)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "⚠️ No terrain file\n";
        mask.resize(WIDTH * HEIGHT, 0);
        return false;
    }
    
    // Read standard 28-byte header
    ClimateFileHeader header;
    if (!readClimateHeader(file, header)) {
        std::cerr << "⚠️ Invalid terrain header\n";
        mask.resize(WIDTH * HEIGHT, 0);
        return false;
    }
    
    // Skip elevation data (float array)
    file.seekg(CLIMATE_HEADER_SIZE + WIDTH * HEIGHT * sizeof(float), std::ios::beg);
    
    // Read land mask
    mask.resize(WIDTH * HEIGHT);
    file.read(reinterpret_cast<char*>(mask.data()), WIDTH * HEIGHT);
    file.close();
    
    size_t landCount = std::count(mask.begin(), mask.end(), 1);
    std::cout << "✅ Land mask: " << (landCount * 100.0 / mask.size()) << "% land\n";
    return true;
}

//============================================================
// Load Biome Map
//============================================================
bool loadBiomeMap(const std::string& filename, std::vector<uint8_t>& map)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Failed to open: " << filename << "\n";
        return false;
    }
    
    // Read standard 28-byte header
    ClimateFileHeader header;
    if (!readClimateHeader(file, header)) {
        return false;
    }

    // Validate magic
    if (header.magic != BIOME_MAGIC) {
        std::cerr << "❌ Invalid magic: 0x" << std::hex << header.magic << "\n";
        return false;
    }
    
    if (header.width != WIDTH || header.height != HEIGHT) {
        std::cerr << "❌ Size mismatch: " << header.width << "x" << header.height << "\n";
        return false;
    }
    
    // Read biome data (starts at byte 28)
    map.resize(header.width * header.height);
    file.read(reinterpret_cast<char*>(map.data()), header.width * header.height);
    file.close();
    
    std::cout << "✅ Biome map loaded\n";
    return true;
}

//============================================================
// Convert to RGB
//============================================================
void biomeMapToRGB(const std::vector<uint8_t>& biomeMap,
                   const std::vector<uint8_t>& landMask,
                   std::vector<uint8_t>& rgbImage)
{
    rgbImage.resize(WIDTH * HEIGHT * CHANNELS);
    
    for (size_t i = 0; i < WIDTH * HEIGHT; ++i) {
        uint8_t biome = biomeMap[i];
        bool isLand = (landMask[i] == 1);
        
        RGB color;
        if (biome < 20) {
            color = isLand ? getLandColor(biome) : getOceanColor(biome);
        } else {
            color = {255, 0, 255};  // Magenta for no-data
        }
        
        rgbImage[i * CHANNELS + 0] = color.r;
        rgbImage[i * CHANNELS + 1] = color.g;
        rgbImage[i * CHANNELS + 2] = color.b;
    }
}

//============================================================
// Export PNG
//============================================================
bool exportPNG(const std::vector<uint8_t>& rgbImage, const std::string& filename)
{
    int result = stbi_write_png(filename.c_str(), WIDTH, HEIGHT, CHANNELS,
                                 rgbImage.data(), WIDTH * CHANNELS);
    
    if (result) {
        std::cout << "✅ Exported: " << filename << "\n";
        return true;
    }
    std::cerr << "❌ Export failed\n";
    return false;
}

//============================================================
// Statistics
//============================================================
void printStatistics(const std::vector<uint8_t>& biomeMap,
                     const std::vector<uint8_t>& landMask)
{
    std::vector<size_t> oceanCounts(21, 0), landCounts(21, 0);
    
    for (size_t i = 0; i < biomeMap.size(); ++i) {
        uint8_t b = biomeMap[i];
        bool isLand = (landMask[i] == 1);
        
        if (b < 20) {
            if (isLand) landCounts[b]++;
            else oceanCounts[b]++;
        } else {
            if (isLand) landCounts[20]++;
            else oceanCounts[20]++;
        }
    }
    
    size_t totalOcean = std::accumulate(oceanCounts.begin(), oceanCounts.begin() + 20, 0ull);
    size_t totalLand = std::accumulate(landCounts.begin(), landCounts.begin() + 20, 0ull);
    
    std::cout << "\n📊 Statistics:\n";
    std::cout << "   Ocean: " << totalOcean << " pixels\n";
    std::cout << "   Land:  " << totalLand << " pixels\n\n";
    
    if (totalOcean > 0) {
        std::cout << "🌊 Ocean zones (0=coldest, 19=warmest):\n";
        for (int i = 0; i < 20; ++i) {
            if (oceanCounts[i] > 0) {
                double pct = (oceanCounts[i] * 100.0) / totalOcean;
                std::cout << "   Zone " << i << ": " << pct << "%\n";
            }
        }
    }
    
    if (totalLand > 0) {
        std::cout << "\n🏔️ Land zones:\n";
        for (int i = 0; i < 20; ++i) {
            if (landCounts[i] > 0) {
                double pct = (landCounts[i] * 100.0) / totalLand;
                std::cout << "   Zone " << i << ": " << pct << "%\n";
            }
        }
    }
}

//============================================================
// Main
//============================================================
int main(int argc, char** argv)
{
    std::string biomeFile = "output/BiomeMap_HighRes.bin";
    std::string terrainFile = "output/TerrainData.bin";
    std::string outputFile = "output/BiomeMap_HighRes.png";
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0 && i+1 < argc)
            biomeFile = argv[++i];
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc)
            terrainFile = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc)
            outputFile = argv[++i];
        else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  -i <file>   Input biome map (20 zones)\n"
                      << "  -t <file>   Terrain data\n"
                      << "  -o <file>   Output PNG\n";
            return 0;
        }
    }
    
    std::cout << "🗺️ High-Resolution Biome Exporter (20 zones)\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "Biome:   " << biomeFile << "\n";
    std::cout << "Terrain: " << terrainFile << "\n";
    std::cout << "Output:  " << outputFile << "\n\n";
    
    std::vector<uint8_t> landMask, biomeMap;
    
    if (!loadLandMask(terrainFile, landMask))
        std::cout << "⚠️ Proceeding without terrain\n";
    
    if (!loadBiomeMap(biomeFile, biomeMap))
        return EXIT_FAILURE;
    
    std::cout << "\n🎨 Generating 20-color gradient visualization...\n";
    std::vector<uint8_t> rgbImage;
    biomeMapToRGB(biomeMap, landMask, rgbImage);
    
    if (!exportPNG(rgbImage, outputFile))
        return EXIT_FAILURE;
    
    printStatistics(biomeMap, landMask);
    
    std::cout << "\n✅ Export complete!\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "\nℹ️  The 20-zone gradient should reveal PNPL circulation\n";
    std::cout << "    patterns in the oceans (wavy bands, gyres, swirls)\n";
    
    return 0;
}