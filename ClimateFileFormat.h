//============================================================
// ClimateFileFormat.h
// STANDARD HEADER FORMAT for all OceanWorldGenerator pipeline files
// Version: 1.0
// Author: Mark Devereux (2025-11-13)
//============================================================

#ifndef CLIMATE_FILE_FORMAT_H
#define CLIMATE_FILE_FORMAT_H

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

//============================================================
// STANDARD CLIMATE FILE FORMAT
//============================================================
/*
    All climate data files in the pipeline use this format:
    
    HEADER (28 bytes):
    - Bytes 0-3:   Magic number (identifies file type)
    - Bytes 4-7:   Width (typically 3600)
    - Bytes 8-11:  Height (typically 1800)
    - Bytes 12-15: Channels (variables × months, e.g., 168 = 14 × 12)
    - Bytes 16-19: Data type (0 = double, 1 = float)
    - Bytes 20-23: Version number (currently 1)
    - Bytes 24-27: Depth (number of months, typically 12)
    
    DATA (starting at byte 28):
    - Double precision values in [cell][variable][month] order
    - Total size: width × height × variables × months × sizeof(double)
*/

//============================================================
// Magic Numbers (4-byte identifiers)
//============================================================
static constexpr uint32_t OCEANWORLD_MAGIC = 0x574E434F; // 'OCNW' - Ocean world data
static constexpr uint32_t CLIMATE_MAGIC    = 0x4D494C43; // 'CLIM' - Climate data
static constexpr uint32_t BIOME_MAGIC      = 0x42494F4D; // 'BIOM' - Biome map
static constexpr uint32_t TERRAIN_MAGIC    = 0x4E525254; // 'TRRN' - Terrain data
static constexpr uint32_t RAINFALL_MAGIC   = 0x4E494152; // 'RAIN' - Rainfall data
static constexpr uint32_t PNPL_MAGIC       = 0x4C504E50; // 'PNPL' - PNPL perturbation
static constexpr uint32_t BATH_MAGIC       = 0x42415448; // 'BATH' - Bathymetry data
static constexpr uint32_t BASIN_MAGIC      = 0x42415349; // 'BASI' - Basin map (BASIN doesn't fit)

//============================================================
// Standard Header Structure
//============================================================
struct ClimateFileHeader {
    uint32_t magic;      // File type identifier
    uint32_t width;      // Grid width (longitude)
    uint32_t height;     // Grid height (latitude)
    uint32_t channels;   // Total channels (variables × months)
    uint32_t dtype;      // 0 = double, 1 = float, 2 = int
    uint32_t version;    // Format version (currently 1)
    uint32_t depth;      // Number of time steps (months)
    
    // Helper methods
    bool isValid() const {
        return (magic == OCEANWORLD_MAGIC || 
                magic == CLIMATE_MAGIC || 
                magic == BIOME_MAGIC || 
                magic == TERRAIN_MAGIC ||
                magic == RAINFALL_MAGIC ||
                magic == PNPL_MAGIC ||
                magic == BATH_MAGIC ||
                magic == BASIN_MAGIC) &&
               width > 0 && height > 0 && 
               channels > 0 && version > 0;
    }
    
    size_t getTotalPixels() const {
        return (size_t)width * height;
    }
    
    size_t getDataSize() const {
        size_t elementSize = (dtype == 0) ? sizeof(double) : 
                            (dtype == 1) ? sizeof(float) : sizeof(int);
        return getTotalPixels() * channels * elementSize;
    }
    
    void print() const {
        const char* typeStr = (dtype == 0) ? "double" : 
                             (dtype == 1) ? "float" : "int";
        std::cout << "📋 Header Info:\n";
        std::cout << "   Magic: 0x" << std::hex << magic << std::dec << "\n";
        std::cout << "   Dimensions: " << width << " × " << height << "\n";
        std::cout << "   Channels: " << channels << " (depth=" << depth << ")\n";
        std::cout << "   Type: " << typeStr << "\n";
        std::cout << "   Version: " << version << "\n";
        std::cout << "   Data size: " << (getDataSize() / (1024.0*1024.0)) << " MB\n";
    }
};

static constexpr size_t CLIMATE_HEADER_SIZE = 28; // bytes

//============================================================
// Helper Functions: Read Header
//============================================================
inline bool readClimateHeader(std::ifstream& file, ClimateFileHeader& header)
{
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(&header.magic), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&header.width), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&header.height), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&header.channels), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&header.dtype), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&header.version), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&header.depth), sizeof(uint32_t));
    
    if (!header.isValid()) {
        std::cerr << "❌ Invalid header: magic=0x" << std::hex 
                  << header.magic << std::dec << "\n";
        return false;
    }
    
    return true;
}

inline bool readClimateHeader(const std::string& filename, ClimateFileHeader& header)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Failed to open: " << filename << "\n";
        return false;
    }
    
    bool success = readClimateHeader(file, header);
    file.close();
    return success;
}

//============================================================
// Helper Functions: Write Header
//============================================================
inline void writeClimateHeader(std::ofstream& file, const ClimateFileHeader& header)
{
    file.write(reinterpret_cast<const char*>(&header.magic), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&header.width), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&header.height), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&header.channels), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&header.dtype), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&header.version), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&header.depth), sizeof(uint32_t));
}

//============================================================
// Helper Functions: Load Climate Data
//============================================================
inline bool loadClimateData(const std::string& filename, 
                           std::vector<double>& data,
                           ClimateFileHeader& header)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Failed to open: " << filename << "\n";
        return false;
    }
    
    // Read header
    if (!readClimateHeader(file, header)) {
        return false;
    }
    
    header.print();
    
    // Verify expected size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    size_t expectedSize = CLIMATE_HEADER_SIZE + header.getDataSize();
    
    if (fileSize != expectedSize) {
        std::cerr << "❌ Size mismatch!\n";
        std::cerr << "   Expected: " << expectedSize << " bytes\n";
        std::cerr << "   Got: " << fileSize << " bytes\n";
        std::cerr << "   Difference: " << (int64_t)fileSize - (int64_t)expectedSize << " bytes\n";
        return false;
    }
    
    // Read data (assuming double for now)
    file.seekg(CLIMATE_HEADER_SIZE, std::ios::beg);
    size_t numElements = header.getTotalPixels() * header.channels;
    data.resize(numElements);
    file.read(reinterpret_cast<char*>(data.data()), numElements * sizeof(double));
    
    file.close();
    
    std::cout << "✓ Loaded " << (data.size() * sizeof(double) / (1024.0*1024.0)) 
              << " MB\n";
    
    return true;
}

//============================================================
// Helper Functions: Save Climate Data
//============================================================
inline bool saveClimateData(const std::string& filename,
                           const std::vector<double>& data,
                           const ClimateFileHeader& header)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Failed to create: " << filename << "\n";
        return false;
    }
    
    // Write header
    writeClimateHeader(file, header);
    
    // Write data
    file.write(reinterpret_cast<const char*>(data.data()), 
               data.size() * sizeof(double));
    
    file.close();
    
    size_t totalSize = CLIMATE_HEADER_SIZE + data.size() * sizeof(double);
    std::cout << "✓ Saved " << (totalSize / (1024.0*1024.0)) << " MB to " 
              << filename << "\n";
    
    return true;
}

//============================================================
// Constants for Standard Grid
//============================================================
static constexpr uint32_t STANDARD_WIDTH = 3600;
static constexpr uint32_t STANDARD_HEIGHT = 1800;
static constexpr uint32_t STANDARD_MONTHS = 12;
static constexpr uint32_t STANDARD_VARIABLES = 14;
static constexpr uint32_t STANDARD_CHANNELS = STANDARD_VARIABLES * STANDARD_MONTHS; // 168
static constexpr uint32_t STANDARD_VERSION = 1;

//============================================================
// Helper: Create Standard Header
//============================================================
inline ClimateFileHeader createStandardHeader(uint32_t magic, 
                                               uint32_t variables = STANDARD_VARIABLES,
                                               uint32_t months = STANDARD_MONTHS)
{
    ClimateFileHeader header;
    header.magic = magic;
    header.width = STANDARD_WIDTH;
    header.height = STANDARD_HEIGHT;
    header.channels = variables * months;
    header.dtype = 0; // double
    header.version = STANDARD_VERSION;
    header.depth = months;
    return header;
}

#endif // CLIMATE_FILE_FORMAT_H