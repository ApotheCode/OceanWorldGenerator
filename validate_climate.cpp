#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <limits>

constexpr size_t WIDTH = 3600;
constexpr size_t HEIGHT = 1800;
constexpr size_t MONTHS = 12;
constexpr size_t VARIABLES = 10;
constexpr size_t TOTAL_CELLS = static_cast<size_t>(WIDTH) * HEIGHT;

struct ValidationStats {
    size_t totalValues = 0;
    size_t nanCount = 0;
    size_t infCount = 0;
    size_t negInfCount = 0;
    size_t validCount = 0;
    double minVal = std::numeric_limits<double>::max();
    double maxVal = std::numeric_limits<double>::lowest();
    double sum = 0.0;
    double sqsum = 0.0;
};

int main(int argc, char** argv)
{
    std::string filename = (argc > 1) ? argv[1] : "OceanWorld_Climate.bin";

    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "❌ Failed to open file: " << filename << std::endl;
        return 1;
    }

    // Check file size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t expectedSize = TOTAL_CELLS * MONTHS * VARIABLES * sizeof(double);
    std::cout << "📦 File size: " << (fileSize / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    std::cout << "📐 Expected: " << (expectedSize / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;

    if (fileSize != expectedSize)
    {
        std::cerr << "⚠️  File size mismatch! Expected " << expectedSize 
                  << " bytes, got " << fileSize << " bytes.\n";
        std::cerr << "   Difference: " << (static_cast<long long>(fileSize) - static_cast<long long>(expectedSize)) 
                  << " bytes\n";
        return 2;
    }

    // Read file in chunks to avoid memory issues
    size_t totalDoubles = TOTAL_CELLS * MONTHS * VARIABLES;
    constexpr size_t CHUNK_SIZE = 1024 * 1024; // 1M doubles = ~8MB per chunk
    std::vector<double> buffer(CHUNK_SIZE);
    
    std::cout << "📖 Reading and validating in chunks..." << std::flush;

    // Overall validation
    std::cout << " Done.\n\n";
    std::cout << "🔍 VALIDATION REPORT\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    ValidationStats overall;
    overall.totalValues = totalDoubles;

    // Validate all values in chunks
    size_t doublesProcessed = 0;
    while (doublesProcessed < totalDoubles)
    {
        size_t doublesThisChunk = std::min(CHUNK_SIZE, totalDoubles - doublesProcessed);
        
        file.read(reinterpret_cast<char*>(buffer.data()), doublesThisChunk * sizeof(double));
        
        if (!file && !file.eof())
        {
            std::cerr << "\n❌ Failed to read file at position " << doublesProcessed << "!" << std::endl;
            return 3;
        }
        
        // Validate this chunk
        for (size_t i = 0; i < doublesThisChunk; ++i)
        {
            double val = buffer[i];
            
            if (std::isnan(val))
            {
                overall.nanCount++;
            }
            else if (std::isinf(val))
            {
                if (val > 0)
                    overall.infCount++;
                else
                    overall.negInfCount++;
            }
            else
            {
                overall.validCount++;
                overall.minVal = std::min(overall.minVal, val);
                overall.maxVal = std::max(overall.maxVal, val);
                overall.sum += val;
                overall.sqsum += val * val;
            }
        }
        
        doublesProcessed += doublesThisChunk;
        
        // Progress indicator
        if (doublesProcessed % (10 * CHUNK_SIZE) == 0 || doublesProcessed == totalDoubles)
        {
            int percent = static_cast<int>(100.0 * doublesProcessed / totalDoubles);
            std::cout << "\r🔍 Validating: " << percent << "% (" 
                      << doublesProcessed << "/" << totalDoubles << ")" << std::flush;
        }
    }
    
    file.close();
    std::cout << "\r🔍 Validating: 100% (" << totalDoubles << "/" << totalDoubles << ")          \n\n";

    // Print overall statistics
    std::cout << "📊 OVERALL STATISTICS:\n";
    std::cout << "   Total values:      " << overall.totalValues << "\n";
    std::cout << "   Valid values:      " << overall.validCount 
              << " (" << (100.0 * overall.validCount / overall.totalValues) << "%)\n";
    std::cout << "   NaN values:        " << overall.nanCount 
              << " (" << (100.0 * overall.nanCount / overall.totalValues) << "%)\n";
    std::cout << "   +Infinity values:  " << overall.infCount 
              << " (" << (100.0 * overall.infCount / overall.totalValues) << "%)\n";
    std::cout << "   -Infinity values:  " << overall.negInfCount 
              << " (" << (100.0 * overall.negInfCount / overall.totalValues) << "%)\n";

    if (overall.validCount > 0)
    {
        double mean = overall.sum / overall.validCount;
        double stddev = std::sqrt((overall.sqsum / overall.validCount) - (mean * mean));
        
        std::cout << "\n   Min value:         " << overall.minVal << "\n";
        std::cout << "   Max value:         " << overall.maxVal << "\n";
        std::cout << "   Mean:              " << mean << "\n";
        std::cout << "   Std deviation:     " << stddev << "\n";
    }

    // Validation result
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    if (overall.nanCount == 0 && overall.infCount == 0 && overall.negInfCount == 0)
    {
        std::cout << "✅ VALIDATION PASSED - No NaN or Infinity values found!\n";
    }
    else
    {
        std::cout << "❌ VALIDATION FAILED - Found invalid values!\n";
    }
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    // Per-variable breakdown (requires second pass through file)
    std::cout << "📋 PER-VARIABLE BREAKDOWN:\n";
    std::cout << "(Re-reading file for detailed analysis...)\n\n";
    
    file.open(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "❌ Failed to reopen file for variable analysis\n";
        return 4;
    }
    
    std::vector<std::string> varNames = {
        "Variable 1", "Variable 2", "Variable 3", "Variable 4", "Variable 5",
        "Variable 6", "Variable 7", "Variable 8", "Variable 9", "Variable 10"
    };

    std::vector<ValidationStats> varStats(VARIABLES);
    for (auto& vs : varStats)
        vs.totalValues = TOTAL_CELLS * MONTHS;

    // Process file in chunks, tracking variables
    doublesProcessed = 0;
    while (doublesProcessed < totalDoubles)
    {
        size_t doublesThisChunk = std::min(CHUNK_SIZE, totalDoubles - doublesProcessed);
        
        file.read(reinterpret_cast<char*>(buffer.data()), doublesThisChunk * sizeof(double));
        
        if (!file && !file.eof())
        {
            std::cerr << "❌ Failed to read file during variable analysis\n";
            return 5;
        }
        
        // Process this chunk
        for (size_t i = 0; i < doublesThisChunk; ++i)
        {
            size_t globalIdx = doublesProcessed + i;
            size_t varIdx = globalIdx % VARIABLES;
            double val = buffer[i];

            if (std::isnan(val))
                varStats[varIdx].nanCount++;
            else if (std::isinf(val))
            {
                if (val > 0)
                    varStats[varIdx].infCount++;
                else
                    varStats[varIdx].negInfCount++;
            }
            else
            {
                varStats[varIdx].validCount++;
                varStats[varIdx].minVal = std::min(varStats[varIdx].minVal, val);
                varStats[varIdx].maxVal = std::max(varStats[varIdx].maxVal, val);
                varStats[varIdx].sum += val;
            }
        }
        
        doublesProcessed += doublesThisChunk;
    }
    
    file.close();

    for (size_t v = 0; v < VARIABLES; ++v)
    {
        std::cout << "   " << varNames[v] << ":\n";
        std::cout << "      Valid: " << varStats[v].validCount 
                  << " | NaN: " << varStats[v].nanCount
                  << " | +Inf: " << varStats[v].infCount
                  << " | -Inf: " << varStats[v].negInfCount;
        
        if (varStats[v].validCount > 0)
        {
            double mean = varStats[v].sum / varStats[v].validCount;
            std::cout << "\n      Range: [" << varStats[v].minVal << ", " << varStats[v].maxVal << "]"
                      << " | Mean: " << mean;
        }
        std::cout << "\n\n";
    }

    // Sample some specific locations if requested
    std::cout << "🔍 Sample specific location? (y/n): ";
    char response;
    if (std::cin >> response && (response == 'y' || response == 'Y'))
    {
        int lon, lat, month, var;
        std::cout << "Enter lon (0-" << (WIDTH-1) << "), lat (0-" << (HEIGHT-1) 
                  << "), month (1-12), variable (1-10): ";
        if (std::cin >> lon >> lat >> month >> var)
        {
            if (lon >= 0 && lon < WIDTH && lat >= 0 && lat < HEIGHT && 
                month >= 1 && month <= 12 && var >= 1 && var <= 10)
            {
                // Reopen file and seek to specific position
                file.open(filename, std::ios::binary);
                if (!file)
                {
                    std::cerr << "Failed to reopen file for sampling\n";
                }
                else
                {
                    size_t cellIdx = lat * WIDTH + lon;
                    size_t idx = (cellIdx * MONTHS * VARIABLES) + ((month - 1) * VARIABLES) + (var - 1);
                    
                    file.seekg(idx * sizeof(double), std::ios::beg);
                    double val;
                    file.read(reinterpret_cast<char*>(&val), sizeof(double));
                    file.close();
                    
                    std::cout << "\nValue at (lon=" << lon << ", lat=" << lat 
                              << ", month=" << month << ", var=" << var << ") = " << val;
                    
                    if (std::isnan(val))
                        std::cout << " [NaN]";
                    else if (std::isinf(val))
                        std::cout << " [Infinity]";
                    
                    std::cout << "\n";
                }
            }
        }
    }

    std::cout << "\n✅ Validation complete.\n";
    return (overall.nanCount == 0 && overall.infCount == 0 && overall.negInfCount == 0) ? 0 : 1;
}