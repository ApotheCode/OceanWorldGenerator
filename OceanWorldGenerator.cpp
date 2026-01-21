//#include "OceanWorldGenerator.h"
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <cassert>
#include <cmath>

// Output climate struct
struct FClimateCell {
    double SST[12];
    double Rainfall[12];
    double AirTemp[12];
    double Albedo[12];
    double NDVI[12];
    double NetFlux[12];
    double SWFlux[12];
    double LWFlux[12];
    double Insolation[12];
    double SeaIceMask[12];
};

constexpr int Width = 3600;
constexpr int Height = 1800;
constexpr int CellCount = Width * Height;
constexpr int MonthCount = 12;
//constexpr double Pi = 3.1415926535f;

// ---- Helper Function ----
bool LoadClimateChannel(const std::string& path, double* outData, int expectedCount = CellCount * MonthCount) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << path << std::endl;
        return false;
    }
    file.read(reinterpret_cast<char*>(outData), expectedCount * sizeof(double));
    return true;
}

void GenerateOceanWorldClimate(const std::string& outputPath, const std::string& inputFolder) {
    std::cout << "[1] Starting chunked climate generator..." << std::endl;

    constexpr int ChunkSize = 100000; // number of cells per chunk (~96 MB)
    constexpr size_t CellsTotal = CellCount;
    constexpr size_t ChunkBytes = ChunkSize * sizeof(FClimateCell);

    // Open all input files
    auto openFile = [&](const std::string& name) {
        std::ifstream f(inputFolder + "/" + name, std::ios::binary);
        if (!f) {
            std::cerr << "❌ Failed to open " << name << std::endl;
            exit(1);
        }
        return f;
    };

    std::ifstream fSST       = openFile("SST.bin");
    std::ifstream fRainfall  = openFile("Rainfall.bin");
    std::ifstream fAlbedo    = openFile("Albedo.bin");
    std::ifstream fNDVI      = openFile("NDVI.bin");
    std::ifstream fNetFlux   = openFile("NetFlux.bin");
    std::ifstream fSWFlux    = openFile("SWFlux.bin");
    std::ifstream fLWFlux    = openFile("LWFlux.bin");
    std::ifstream fInsolation= openFile("Insolation.bin");
    std::ifstream fSISM      = openFile("SISM.bin");

    // Output file
    std::filesystem::create_directories("climate_output");
    std::ofstream out(outputPath, std::ios::binary);
    if (!out) {
        std::cerr << "❌ Failed to open output file: " << outputPath << std::endl;
        return;
    }

    // Temporary buffers for one chunk of doubles (per channel)
    std::vector<double> bufSST(ChunkSize * MonthCount);
    std::vector<double> bufRain(ChunkSize * MonthCount);
    std::vector<double> bufAlbedo(ChunkSize * MonthCount);
    std::vector<double> bufNDVI(ChunkSize * MonthCount);
    std::vector<double> bufNet(ChunkSize * MonthCount);
    std::vector<double> bufSW(ChunkSize * MonthCount);
    std::vector<double> bufLW(ChunkSize * MonthCount);
    std::vector<double> bufIns(ChunkSize * MonthCount);
    std::vector<double> bufSISM(ChunkSize * MonthCount);

    // Buffer for assembled climate cells
    std::vector<FClimateCell> chunk(ChunkSize);

    size_t processed = 0;
    while (processed < CellsTotal) {
        int thisChunk = std::min<int>(ChunkSize, CellsTotal - processed);

        // Read this chunk from each input file
        auto readBlock = [&](std::ifstream& f, std::vector<double>& buf, const std::string& name) {
        if (f.tellg() == 0) {
            int depth, height, width;
            f.read(reinterpret_cast<char*>(&depth),  sizeof(int));
            f.read(reinterpret_cast<char*>(&height), sizeof(int));
            f.read(reinterpret_cast<char*>(&width),  sizeof(int));
        }
        f.read(reinterpret_cast<char*>(buf.data()), thisChunk * MonthCount * sizeof(double));
    };


        readBlock(fSST, bufSST, "SST.bin");
        readBlock(fRainfall, bufRain, "Rainfall.bin");
        readBlock(fAlbedo, bufAlbedo, "Albedo.bin");
        readBlock(fNDVI, bufNDVI, "NDVI.bin");
        readBlock(fNetFlux, bufNet, "NetFlux.bin");
        readBlock(fSWFlux, bufSW, "SWFlux.bin");
        readBlock(fLWFlux, bufLW, "LWFlux.bin");
        readBlock(fInsolation, bufIns, "Insolation.bin");
        readBlock(fSISM, bufSISM, "SISM.bin");

        // Assemble FClimateCell for this chunk
        for (int i = 0; i < thisChunk; ++i) {
            for (int m = 0; m < MonthCount; ++m) {
                int idx = i * MonthCount + m;

                chunk[i].SST[m]        = bufSST[idx];
                chunk[i].Rainfall[m]   = bufRain[idx];
                chunk[i].Albedo[m]     = bufAlbedo[idx];
                chunk[i].NDVI[m]       = bufNDVI[idx];
                chunk[i].NetFlux[m]    = bufNet[idx];
                chunk[i].SWFlux[m]     = bufSW[idx];
                chunk[i].LWFlux[m]     = bufLW[idx];
                chunk[i].Insolation[m] = bufIns[idx];
                chunk[i].SeaIceMask[m] = bufSISM[idx];

                // Derived: AirTemp
                chunk[i].AirTemp[m] = bufSST[idx] - 2.0;
            }
        }

        // Write assembled chunk to file
        out.write(reinterpret_cast<const char*>(chunk.data()), thisChunk * sizeof(FClimateCell));
        if (!out) {
            std::cerr << "❌ File write error!" << std::endl;
            exit(1);
        }

        processed += thisChunk;
        std::cout << "   → Processed " << processed << " / " << CellsTotal << " cells\r" << std::flush;
    }

    double sizeGB = static_cast<double>(CellsTotal) * sizeof(FClimateCell) / (1024.0*1024.0*1024.0);
    std::cout << "✅ Finished. Wrote " 
          << sizeGB << " GB to " << outputPath << std::endl;

}

int main() {
    try {
        const std::string inputFolder = "climate_inputs";
        const std::string outputPath = "climate_output/OceanWorld_Climate.bin";

        std::cout << "🌍 OceanWorld Generator starting..." << std::endl;

        GenerateOceanWorldClimate(outputPath, inputFolder);
    }
    catch (const std::bad_alloc& e) {
        std::cerr << "❌ Memory allocation failed: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "❌ Exception: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "❌ Unknown fatal error" << std::endl;
        return 1;
    }

    return 0;
}

