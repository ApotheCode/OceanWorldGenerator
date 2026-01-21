//============================================================
// GyreIntegrator.cpp
// Integrates gyre velocity fields into climate data file
// Adds u,v as variables 15,16 (channels 180-203)
//============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include "ClimateFileFormat.h"

constexpr int WIDTH = STANDARD_WIDTH;
constexpr int HEIGHT = STANDARD_HEIGHT;
constexpr int MONTHS = 12;
constexpr int OLD_VARS = 14;
constexpr int NEW_VARS = 16;  // Add u,v

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <climate.bin> <gyre_u.bin> <gyre_v.bin> <output.bin>\n";
        return 1;
    }
    
    std::string climate_file = argv[1];
    std::string gyre_u_file = argv[2];
    std::string gyre_v_file = argv[3];
    std::string output_file = argv[4];
    
    std::cout << "🔗 Gyre Integrator\n";
    std::cout << "==================\n\n";
    
    // Load climate data (14 variables)
    ClimateFileHeader climate_header;
    std::vector<double> climate_data;
    if (!loadClimateData(climate_file, climate_data, climate_header)) {
        return 1;
    }
    
    // Load gyre u
    ClimateFileHeader u_header;
    std::vector<double> u_data;
    if (!loadClimateData(gyre_u_file, u_data, u_header)) {
        return 1;
    }
    
    // Load gyre v
    ClimateFileHeader v_header;
    std::vector<double> v_data;
    if (!loadClimateData(gyre_v_file, v_data, v_header)) {
        return 1;
    }
    
    // Create extended climate data (16 variables)
    size_t N = WIDTH * HEIGHT;
    std::vector<double> extended_data(N * NEW_VARS * MONTHS);
    
    std::cout << "🔧 Integrating gyres into climate data...\n";
    
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int idx = y * WIDTH + x;
            
            for (int m = 0; m < MONTHS; m++) {
                // Copy original 14 variables
                for (int v = 0; v < OLD_VARS; v++) {
                    int old_idx = idx * OLD_VARS * MONTHS + v * MONTHS + m;
                    int new_idx = idx * NEW_VARS * MONTHS + v * MONTHS + m;
                    extended_data[new_idx] = climate_data[old_idx];
                }
                
                // Add u (variable 14)
                int u_idx = idx * MONTHS + m;
                int new_u_idx = idx * NEW_VARS * MONTHS + 14 * MONTHS + m;
                extended_data[new_u_idx] = u_data[u_idx];
                
                // Add v (variable 15)
                int v_idx = idx * MONTHS + m;
                int new_v_idx = idx * NEW_VARS * MONTHS + 15 * MONTHS + m;
                extended_data[new_v_idx] = v_data[v_idx];
            }
        }
    }
    
    // Save extended climate file
    ClimateFileHeader output_header = createStandardHeader(CLIMATE_MAGIC, NEW_VARS, MONTHS);
    
    if (!saveClimateData(output_file, extended_data, output_header)) {
        return 1;
    }
    
    std::cout << "\n✅ Integration complete\n";
    std::cout << "   Variables: 14 → 16 (added u,v)\n";
    std::cout << "   Channels: " << OLD_VARS * MONTHS << " → " << NEW_VARS * MONTHS << "\n";
    std::cout << "   Output: " << output_file << "\n";
    
    return 0;
}
