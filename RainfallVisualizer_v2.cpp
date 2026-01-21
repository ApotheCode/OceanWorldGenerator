//============================================================
// RainfallVisualizer.cpp
// Visualize monthly rainfall patterns with terrain overlay
// Author: Claude & Mark Devereux (2025-11-11)
//============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <cstdint>
#include "ClimateFileFormat.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// MinGW compatibility: std::to_string workaround
namespace {
    template<typename T>
    std::string to_string_compat(T value) {
        std::ostringstream oss;
        oss << value;
        return oss.str();
    }
}

constexpr int WIDTH = STANDARD_WIDTH;
constexpr int HEIGHT = STANDARD_HEIGHT;
constexpr int MONTHS = STANDARD_MONTHS;
constexpr int VARIABLES = STANDARD_VARIABLES;
constexpr int VAR_RAINFALL = 0;

//============================================================
// Color Ramps
//============================================================

struct RGB {
    unsigned char r, g, b;
};

// Rainfall color ramp (dry to wet)
RGB getRainfallColor(double rainfall) {
    // rainfall in [0, 1]
    // 0.0 = Desert (tan/brown)
    // 0.3 = Semi-arid (yellow)
    // 0.5 = Moderate (light green)
    // 0.7 = Wet (dark green)
    // 0.9 = Very wet (blue-green)
    // 1.0 = Rainforest (deep blue)
    
    RGB color;
    
    if (rainfall < 0.15) {
        // Desert: Brown to Tan (0.0 - 0.15)
        double t = rainfall / 0.15;
        color.r = static_cast<unsigned char>(139 + (210 - 139) * t);
        color.g = static_cast<unsigned char>(69 + (180 - 69) * t);
        color.b = static_cast<unsigned char>(19 + (140 - 19) * t);
    }
    else if (rainfall < 0.30) {
        // Semi-arid: Tan to Yellow (0.15 - 0.30)
        double t = (rainfall - 0.15) / 0.15;
        color.r = static_cast<unsigned char>(210 + (255 - 210) * t);
        color.g = static_cast<unsigned char>(180 + (215 - 180) * t);
        color.b = static_cast<unsigned char>(140 + (0 - 140) * t);
    }
    else if (rainfall < 0.50) {
        // Moderate: Yellow to Light Green (0.30 - 0.50)
        double t = (rainfall - 0.30) / 0.20;
        color.r = static_cast<unsigned char>(255 + (144 - 255) * t);
        color.g = static_cast<unsigned char>(215 + (238 - 215) * t);
        color.b = static_cast<unsigned char>(0 + (144 - 0) * t);
    }
    else if (rainfall < 0.70) {
        // Wet: Light Green to Dark Green (0.50 - 0.70)
        double t = (rainfall - 0.50) / 0.20;
        color.r = static_cast<unsigned char>(144 + (34 - 144) * t);
        color.g = static_cast<unsigned char>(238 + (139 - 238) * t);
        color.b = static_cast<unsigned char>(144 + (34 - 144) * t);
    }
    else if (rainfall < 0.85) {
        // Very Wet: Dark Green to Teal (0.70 - 0.85)
        double t = (rainfall - 0.70) / 0.15;
        color.r = static_cast<unsigned char>(34 + (0 - 34) * t);
        color.g = static_cast<unsigned char>(139 + (206 - 139) * t);
        color.b = static_cast<unsigned char>(34 + (209 - 34) * t);
    }
    else {
        // Rainforest: Teal to Deep Blue (0.85 - 1.0)
        double t = (rainfall - 0.85) / 0.15;
        color.r = static_cast<unsigned char>(0 + (25 - 0) * t);
        color.g = static_cast<unsigned char>(206 + (25 - 206) * t);
        color.b = static_cast<unsigned char>(209 + (112 - 209) * t);
    }
    
    return color;
}

// Grayscale for terrain elevation (optional overlay)
RGB getElevationGrayscale(float elevation, float maxElev) {
    unsigned char gray = static_cast<unsigned char>(255.0 * (elevation / maxElev));
    return {gray, gray, gray};
}

//============================================================
// Data Loading
//============================================================

bool loadClimateData(const std::string& filename, std::vector<double>& data)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Failed to open: " << filename << "\n";
        return false;
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Read standard 28-byte header
    ClimateFileHeader header;
    if (!readClimateHeader(file, header)) {
        return false;
    }
    
    if (header.width != WIDTH || header.height != HEIGHT) {
        std::cerr << "❌ Dimension mismatch: " << header.width << "×" << header.height << "\n";
        return false;
    }
    
    std::cout << "📦 Climate data: " << header.width << "×" << header.height << "\n";
    std::cout << "   Magic: 0x" << std::hex << header.magic << std::dec << "\n";
    std::cout << "   Version: " << header.version << "\n";
    std::cout << "   Channels: " << header.channels << "\n";
    std::cout << "   Depth: " << header.depth << "\n";
    
    // Calculate expected size
    size_t totalDataSize = (size_t)WIDTH * HEIGHT * VARIABLES * MONTHS;
    size_t expectedFileSize = CLIMATE_HEADER_SIZE + totalDataSize * sizeof(double);
    
    std::cout << "   Expected: " << expectedFileSize << " bytes\n";
    
    if (fileSize < expectedFileSize) {
        std::cerr << "❌ File too small!\n";
        std::cerr << "   Missing: " << (expectedFileSize - fileSize) << " bytes\n";
        std::cerr << "   This file may not have rainfall data (Variable 0) populated.\n";
        std::cerr << "   Did RainfallCalculator complete successfully?\n";
        return false;
    }
    
    // Read data in chunks (8.7 GB is too large for single read)
    data.resize(totalDataSize);
    
    const size_t CHUNK_SIZE = 100 * 1024 * 1024 / sizeof(double); // 100 MB chunks
    size_t totalRead = 0;
    
    std::cout << "   Reading in chunks...\n";
    
    for (size_t offset = 0; offset < totalDataSize; offset += CHUNK_SIZE) {
        size_t chunkSize = std::min(CHUNK_SIZE, totalDataSize - offset);
        
        file.read(reinterpret_cast<char*>(data.data() + offset), chunkSize * sizeof(double));
        
        if (!file) {
            std::cerr << "❌ Failed to read chunk at offset " << offset << "\n";
            std::cerr << "   Read " << file.gcount() << " bytes this chunk\n";
            std::cerr << "   Total read so far: " << (totalRead * sizeof(double)) << " bytes\n";
            return false;
        }
        
        totalRead += chunkSize;
        
        // Progress indicator
        int percent = (totalRead * 100) / totalDataSize;
        std::cout << "   Progress: " << percent << "%\r" << std::flush;
    }
    
    std::cout << "\n   ✅ Read " << totalDataSize << " values successfully\n";
    
    file.close();
    return true;
}

bool loadTerrainData(const std::string& filename, 
                     std::vector<float>& elevation,
                     std::vector<unsigned char>& isLand)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        // Terrain is optional
        return false;
    }
    
    // Read standard 28-byte header
    ClimateFileHeader header;
    if (!readClimateHeader(file, header)) {
        std::cerr << "⚠️ Invalid terrain header\n";
        return false;
    }
    
    if (header.width != WIDTH || header.height != HEIGHT) {
        std::cerr << "⚠️ Terrain dimension mismatch\n";
        return false;
    }
    
    size_t totalPixels = WIDTH * HEIGHT;
    elevation.resize(totalPixels);
    isLand.resize(totalPixels);
    
    // Read elevation (starts at byte 28)
    file.read(reinterpret_cast<char*>(elevation.data()), totalPixels * sizeof(float));
    
    // Read land mask
    file.read(reinterpret_cast<char*>(isLand.data()), totalPixels * sizeof(unsigned char));
    
    file.close();
    
    std::cout << "✅ Terrain loaded (overlay enabled)\n";
    return true;
}

//============================================================
// Visualization
//============================================================

void generateRainfallVisualization(const std::vector<double>& climateData,
                                   const std::vector<float>& elevation,
                                   const std::vector<unsigned char>& isLand,
                                   const std::string& outputFile,
                                   int month,
                                   bool showTerrain)
{
    std::cout << "🎨 Rendering month " << (month + 1) << "/12...\n";
    
    // Allocate image buffer
    std::vector<unsigned char> image(WIDTH * HEIGHT * 3);
    
    // Find max elevation for terrain overlay
    float maxElev = 1.0f;
    if (showTerrain && !elevation.empty()) {
        maxElev = *std::max_element(elevation.begin(), elevation.end());
    }
    
    // Render each pixel
    size_t totalPixels = WIDTH * HEIGHT;
    for (size_t i = 0; i < totalPixels; ++i) {
        int x = i % WIDTH;
        int y = i / WIDTH;
        
        // Get rainfall value for this cell and month
        size_t cellBase = i * VARIABLES * MONTHS;
        size_t rainfallIdx = cellBase + VAR_RAINFALL * MONTHS + month;
        double rainfall = climateData[rainfallIdx];
        
        RGB color;
        
        // Check if land or ocean
        bool land = (!isLand.empty() && isLand[i] == 1);
        
        if (showTerrain && land && !elevation.empty()) {
            // Land: Blend rainfall color with terrain grayscale
            RGB rainfallColor = getRainfallColor(rainfall);
            RGB terrainColor = getElevationGrayscale(elevation[i], maxElev);
            
            // 70% rainfall, 30% terrain
            color.r = static_cast<unsigned char>(rainfallColor.r * 0.7 + terrainColor.r * 0.3);
            color.g = static_cast<unsigned char>(rainfallColor.g * 0.7 + terrainColor.g * 0.3);
            color.b = static_cast<unsigned char>(rainfallColor.b * 0.7 + terrainColor.b * 0.3);
        }
        else {
            // Ocean or no terrain: Pure rainfall color
            color = getRainfallColor(rainfall);
            
            // Darken ocean slightly for contrast
            if (!land) {
                color.r = static_cast<unsigned char>(color.r * 0.85);
                color.g = static_cast<unsigned char>(color.g * 0.85);
                color.b = static_cast<unsigned char>(color.b * 0.85);
            }
        }
        
        // Write to image buffer
        size_t imgIdx = i * 3;
        image[imgIdx + 0] = color.r;
        image[imgIdx + 1] = color.g;
        image[imgIdx + 2] = color.b;
    }
    
    // Save PNG
    if (!stbi_write_png(outputFile.c_str(), WIDTH, HEIGHT, 3, image.data(), WIDTH * 3)) {
        std::cerr << "❌ Failed to write: " << outputFile << "\n";
        return;
    }
    
    std::cout << "   ✅ Saved: " << outputFile << "\n";
}

//============================================================
// Statistics
//============================================================

void printRainfallStatistics(const std::vector<double>& climateData, int month)
{
    double minRain = 1e9, maxRain = -1e9, avgRain = 0.0;
    size_t count = 0;
    
    size_t totalPixels = WIDTH * HEIGHT;
    for (size_t i = 0; i < totalPixels; ++i) {
        size_t cellBase = i * VARIABLES * MONTHS;
        size_t rainfallIdx = cellBase + VAR_RAINFALL * MONTHS + month;
        double rainfall = climateData[rainfallIdx];
        
        minRain = std::min(minRain, rainfall);
        maxRain = std::max(maxRain, rainfall);
        avgRain += rainfall;
        count++;
    }
    
    avgRain /= count;
    
    std::cout << "📊 Month " << (month + 1) << " Statistics:\n";
    std::cout << "   Range: [" << minRain << ", " << maxRain << "]\n";
    std::cout << "   Average: " << avgRain << "\n";
}

//============================================================
// Main
//============================================================

int main(int argc, char** argv)
{
    std::string climateFile = "output/Kyushu_Climate_Rainfall.bin";
    std::string terrainFile = "output/KyushuTerrainData.bin";
    std::string outputPrefix = "output/Rainfall_Month_";
    int monthToRender = -1; // -1 = all months
    bool showTerrain = true;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            climateFile = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            terrainFile = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            outputPrefix = argv[++i];
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            monthToRender = atoi(argv[++i]) - 1; // User input 1-12, internal 0-11
        } else if (strcmp(argv[i], "--no-terrain") == 0) {
            showTerrain = false;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  -i <file>      Input climate file (default: output/Kyushu_Climate_Rainfall.bin)\n";
            std::cout << "  -t <file>      Terrain file for overlay (default: output/KyushuTerrainData.bin)\n";
            std::cout << "  -o <prefix>    Output file prefix (default: output/Rainfall_Month_)\n";
            std::cout << "  -m <month>     Render specific month 1-12 (default: all months)\n";
            std::cout << "  --no-terrain   Disable terrain overlay\n";
            std::cout << "  -h, --help     Show this help\n";
            std::cout << "\nColor Scale:\n";
            std::cout << "  Brown/Tan:     Very dry (desert)\n";
            std::cout << "  Yellow:        Semi-arid\n";
            std::cout << "  Light Green:   Moderate rainfall\n";
            std::cout << "  Dark Green:    Wet\n";
            std::cout << "  Teal:          Very wet\n";
            std::cout << "  Deep Blue:     Rainforest\n";
            return 0;
        }
    }
    
    std::cout << "🌧️ Rainfall Visualizer\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "Climate: " << climateFile << "\n";
    std::cout << "Terrain: " << terrainFile << "\n";
    std::cout << "Output:  " << outputPrefix << "*.png\n\n";
    
    // Load climate data
    std::vector<double> climateData;
    if (!loadClimateData(climateFile, climateData)) {
        return EXIT_FAILURE;
    }
    
    // Load terrain data (optional)
    std::vector<float> elevation;
    std::vector<unsigned char> isLand;
    bool terrainLoaded = loadTerrainData(terrainFile, elevation, isLand);
    
    if (!terrainLoaded) {
        std::cout << "⚠️ Terrain not loaded (pure rainfall visualization)\n";
        showTerrain = false;
    }
    
    std::cout << "\n";
    
    // Render month(s)
    if (monthToRender >= 0 && monthToRender < MONTHS) {
        // Single month
        std::string outputFile = outputPrefix + to_string_compat(monthToRender + 1) + ".png";
        printRainfallStatistics(climateData, monthToRender);
        generateRainfallVisualization(climateData, elevation, isLand, 
                                     outputFile, monthToRender, showTerrain);
    }
    else {
        // All months
        for (int m = 0; m < MONTHS; ++m) {
            std::string outputFile = outputPrefix + to_string_compat(m + 1) + ".png";
            printRainfallStatistics(climateData, m);
            generateRainfallVisualization(climateData, elevation, isLand,
                                         outputFile, m, showTerrain);
        }
    }
    
    std::cout << "\n══════════════════════════════════════════════════════════\n";
    std::cout << "✅ Visualization complete!\n";
    
    if (monthToRender >= 0) {
        std::cout << "   Output: " << outputPrefix << (monthToRender + 1) << ".png\n";
    } else {
        std::cout << "   Output: " << outputPrefix << "1.png through " 
                  << outputPrefix << MONTHS << ".png\n";
    }
    
    std::cout << "\n💡 Tips:\n";
    std::cout << "   - Look for ITCZ band near equator (wet)\n";
    std::cout << "   - Check subtropical dry zones at ~25° (brown/tan)\n";
    std::cout << "   - Verify mid-latitude storm tracks at ~45° (green)\n";
    std::cout << "   - On Kyushu: windward (west) should be greener than leeward (east)\n";
    
    return 0;
}