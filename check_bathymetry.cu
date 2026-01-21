//============================================================
// check_bathymetry.cu
// Quick diagnostic to check bathymetry file contents
//============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "ClimateFileFormat.h"

constexpr int WIDTH = 3600;
constexpr int HEIGHT = 1800;

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <bathymetry.bath>\n";
        return 1;
    }
    
    std::string bath_file = argv[1];
    
    std::cout << "Checking bathymetry file: " << bath_file << "\n";
    std::cout << "==========================================\n\n";
    
    std::ifstream bathFile(bath_file, std::ios::binary);
    if (!bathFile) {
        std::cerr << "❌ Cannot open file\n";
        return 1;
    }
    
    // Read header
    ClimateFileHeader bath_header;
    if (!readClimateHeader(bathFile, bath_header)) {
        std::cerr << "❌ Failed to read header\n";
        return 1;
    }
    
    std::cout << "Header:\n";
    std::cout << "  Magic: 0x" << std::hex << bath_header.magic << std::dec << "\n";
    std::cout << "  Dimensions: " << bath_header.width << " × " << bath_header.height << "\n";
    std::cout << "  Channels: " << bath_header.channels << "\n";
    std::cout << "  Dtype: " << bath_header.dtype << " (0=float, 1=double)\n";
    std::cout << "  Version: " << bath_header.version << "\n";
    std::cout << "  Depth: " << bath_header.depth << "\n\n";
    
    if (bath_header.width != WIDTH || bath_header.height != HEIGHT) {
        std::cerr << "⚠️  Size mismatch! Expected " << WIDTH << "×" << HEIGHT << "\n";
    }
    
    // Read data
    size_t N = bath_header.width * bath_header.height;
    std::vector<float> bath_float(N);
    bathFile.read(reinterpret_cast<char*>(bath_float.data()), N * sizeof(float));
    bathFile.close();
    
    // Analyze
    float min_val = 1e30f;
    float max_val = -1e30f;
    double sum = 0.0;
    size_t positive_count = 0;
    size_t zero_count = 0;
    size_t negative_count = 0;
    size_t nan_count = 0;
    
    for (size_t i = 0; i < N; i++) {
        float val = bath_float[i];
        
        if (std::isnan(val)) {
            nan_count++;
            continue;
        }
        
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
        
        if (val > 0.0f) positive_count++;
        else if (val == 0.0f) zero_count++;
        else negative_count++;
    }
    
    double mean = sum / (N - nan_count);
    double ocean_pct = 100.0 * positive_count / N;
    
    std::cout << "Data Statistics:\n";
    std::cout << "  Min value: " << min_val << "\n";
    std::cout << "  Max value: " << max_val << "\n";
    std::cout << "  Mean value: " << mean << "\n";
    std::cout << "  Positive (ocean): " << positive_count << " (" << ocean_pct << "%)\n";
    std::cout << "  Zero: " << zero_count << "\n";
    std::cout << "  Negative (land): " << negative_count << "\n";
    std::cout << "  NaN: " << nan_count << "\n\n";
    
    std::cout << "==========================================\n";
    std::cout << "DIAGNOSIS:\n";
    std::cout << "==========================================\n";
    
    if (positive_count == 0) {
        std::cout << "❌ CRITICAL: NO OCEAN CELLS!\n";
        std::cout << "   All values are ≤ 0\n";
        std::cout << "   This will produce ZERO velocities\n";
        std::cout << "\n";
        std::cout << "   FIX: Bathymetry must have depth > 0 for ocean\n";
        std::cout << "   Example values:\n";
        std::cout << "     Deep ocean: 4000.0\n";
        std::cout << "     Shallow: 100.0\n";
        std::cout << "     Coast: 0.0 (boundary)\n";
        std::cout << "     Land: -100.0 or < 0\n";
    } else if (ocean_pct < 10.0) {
        std::cout << "⚠️  WARNING: Very little ocean (" << ocean_pct << "%)\n";
        std::cout << "   Gyres may not form properly\n";
    } else if (ocean_pct > 99.0) {
        std::cout << "⚠️  NOTE: Nearly 100% ocean (" << ocean_pct << "%)\n";
        std::cout << "   No land boundaries for western intensification\n";
        std::cout << "   Gyres will be weak and zonally symmetric\n";
    } else {
        std::cout << "✓ Ocean coverage looks reasonable (" << ocean_pct << "%)\n";
    }
    
    std::cout << "==========================================\n";
    
    return 0;
}
