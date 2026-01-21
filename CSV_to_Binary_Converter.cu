#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

// Resolution constants
const int WIDTH = 3600;
const int HEIGHT = 1800;
const int TOTAL_PIXELS = WIDTH * HEIGHT;

// Month dates for GPM rainfall files (2024 data)
const char* MONTH_DATES[] = {
    "2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01",
    "2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01",
    "2024-09-01", "2024-10-01", "2024-11-01", "2024-12-01"
};

const char* MONTH_NAMES[] = {
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
};

/**
 * Parse a CSV value, handling empty strings and invalid data
 */
float parseCSVValue(const std::string& value, bool isElevation = false) {
    if (value.empty() || value == "NA" || value == "NaN") {
        return 0.0f;
    }
    try {
        float val = std::stof(value);
        // Handle 99999 = no data/ocean marker (both in elevation and rainfall)
        if (val >= 99999.0f || val == 99999.0f) {
            return 0.0f;
        }
        return val;
    } catch (...) {
        return 0.0f;
    }
}

/**
 * Load a single CSV file into a float array
 */
bool loadCSVFile(const std::string& filename, float* data, bool isElevation = false) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open file: " << filename << std::endl;
        return false;
    }

    std::cout << "Loading: " << filename << std::endl;
    
    std::string line;
    int row = 0;
    int dataRow = 0;
    
    // For elevation data, skip first row (longitude header)
    if (isElevation) {
        std::getline(file, line); // Skip header row
        std::cout << "  (Skipped header row with longitude values)" << std::endl;
    }
    
    while (std::getline(file, line) && dataRow < HEIGHT) {
        std::stringstream ss(line);
        std::string value;
        int col = 0;
        int dataCol = 0;
        
        // For elevation data, skip first column (latitude)
        if (isElevation) {
            std::getline(ss, value, ','); // Skip latitude column
        }
        
        while (std::getline(ss, value, ',') && dataCol < WIDTH) {
            data[dataRow * WIDTH + dataCol] = parseCSVValue(value, isElevation);
            dataCol++;
        }
        
        if (dataCol != WIDTH) {
            std::cerr << "WARNING: Row " << dataRow << " has " << dataCol 
                      << " columns (expected " << WIDTH << ")" << std::endl;
        }
        
        dataRow++;
        
        // Progress indicator every 200 rows
        if (dataRow % 200 == 0) {
            float progress = (dataRow * 100.0f) / HEIGHT;
            std::cout << "  Progress: " << progress << "% (" << dataRow << "/" << HEIGHT << " rows)\r" << std::flush;
        }
    }
    
    std::cout << "  Progress: 100% (" << dataRow << "/" << HEIGHT << " rows) - COMPLETE" << std::endl;
    
    if (dataRow != HEIGHT) {
        std::cerr << "WARNING: Expected " << HEIGHT << " rows, got " << dataRow << std::endl;
    }
    
    file.close();
    return true;
}

/**
 * Convert rainfall CSVs to binary
 */
bool convertRainfallToBinary(const std::string& inputFolder, const std::string& outputFile) {
    std::cout << "\n=== CONVERTING RAINFALL DATA ===" << std::endl;
    std::cout << "Input folder: " << inputFolder << std::endl;
    std::cout << "Output file: " << outputFile << std::endl;
    std::cout << "Resolution: " << WIDTH << "x" << HEIGHT << std::endl;
    std::cout << "Months: 12" << std::endl;
    
    // Allocate memory for all 12 months
    const int TOTAL_SIZE = TOTAL_PIXELS * 12;
    float* rainfallData = new float[TOTAL_SIZE];
    
    if (!rainfallData) {
        std::cerr << "ERROR: Failed to allocate memory for rainfall data" << std::endl;
        return false;
    }
    
    // Load each month
    for (int month = 0; month < 12; month++) {
        std::string filename = inputFolder + "/GPM_3IMERGM_" + std::string(MONTH_DATES[month]) + "_rgb_3600x1800.csv";
        float* monthData = rainfallData + (month * TOTAL_PIXELS);
        
        std::cout << "\n[" << (month + 1) << "/12] Month: " << MONTH_NAMES[month] << std::endl;
        
        if (!loadCSVFile(filename, monthData)) {
            std::cerr << "ERROR: Failed to load " << filename << std::endl;
            delete[] rainfallData;
            return false;
        }
        
        // Calculate statistics
        float minVal = monthData[0];
        float maxVal = monthData[0];
        double sum = 0.0;
        int validPixels = 0;
        
        for (int i = 0; i < TOTAL_PIXELS; i++) {
            float val = monthData[i];
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
            if (val > 0.0f) {
                sum += val;
                validPixels++;
            }
        }
        
        float avgVal = validPixels > 0 ? (sum / validPixels) : 0.0f;
        std::cout << "  Stats: Min=" << minVal << " mm, Max=" << maxVal 
                  << " mm, Avg=" << avgVal << " mm (over " << validPixels << " valid pixels)" << std::endl;
    }
    
    // Write binary file
    std::cout << "\nWriting binary file..." << std::endl;
    std::ofstream outFile(outputFile, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "ERROR: Cannot create output file: " << outputFile << std::endl;
        delete[] rainfallData;
        return false;
    }
    
    outFile.write(reinterpret_cast<char*>(rainfallData), TOTAL_SIZE * sizeof(float));
    outFile.close();
    
    delete[] rainfallData;
    
    std::cout << "✓ Successfully created: " << outputFile << std::endl;
    std::cout << "  File size: " << (TOTAL_SIZE * sizeof(float)) / (1024 * 1024) << " MB" << std::endl;
    
    return true;
}

/**
 * Convert elevation CSV to binary
 */
bool convertElevationToBinary(const std::string& inputFolder, const std::string& outputFile) {
    std::cout << "\n=== CONVERTING ELEVATION DATA ===" << std::endl;
    
    std::string filename = inputFolder + "/SRTM_RAMP2_TOPO_2000-02-11_rgb_3600x1800.SS.CSV";
    std::cout << "Input file: " << filename << std::endl;
    std::cout << "Output file: " << outputFile << std::endl;
    std::cout << "Resolution: " << WIDTH << "x" << HEIGHT << std::endl;
    
    // Allocate memory
    float* elevationData = new float[TOTAL_PIXELS];
    
    if (!elevationData) {
        std::cerr << "ERROR: Failed to allocate memory for elevation data" << std::endl;
        return false;
    }
    
    // Load elevation data
    if (!loadCSVFile(filename, elevationData, true)) {  // true = isElevation
        std::cerr << "ERROR: Failed to load elevation file" << std::endl;
        delete[] elevationData;
        return false;
    }
    
    // Calculate statistics
    float minVal = elevationData[0];
    float maxVal = elevationData[0];
    double sum = 0.0;
    int landPixels = 0;
    int oceanPixels = 0;
    
    for (int i = 0; i < TOTAL_PIXELS; i++) {
        float val = elevationData[i];
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
        sum += val;
        
        // Ocean = 0m (was 99999) or below sea level, Land = above 0m
        if (val > 0.0f) {
            landPixels++;
        } else {
            oceanPixels++;
        }
    }
    
    float avgVal = sum / TOTAL_PIXELS;
    float landPercentage = (landPixels * 100.0f) / TOTAL_PIXELS;
    
    std::cout << "\nElevation Statistics:" << std::endl;
    std::cout << "  Min: " << minVal << " m" << std::endl;
    std::cout << "  Max: " << maxVal << " m" << std::endl;
    std::cout << "  Avg: " << avgVal << " m" << std::endl;
    std::cout << "  Land pixels (>0m): " << landPixels << " (" << landPercentage << "%)" << std::endl;
    std::cout << "  Ocean pixels (≤0m): " << oceanPixels << " (" << (100.0f - landPercentage) << "%)" << std::endl;
    std::cout << "  Note: 99999 values converted to 0m (sea level)" << std::endl;
    
    // Write binary file
    std::cout << "\nWriting binary file..." << std::endl;
    std::ofstream outFile(outputFile, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "ERROR: Cannot create output file: " << outputFile << std::endl;
        delete[] elevationData;
        return false;
    }
    
    outFile.write(reinterpret_cast<char*>(elevationData), TOTAL_PIXELS * sizeof(float));
    outFile.close();
    
    delete[] elevationData;
    
    std::cout << "✓ Successfully created: " << outputFile << std::endl;
    std::cout << "  File size: " << (TOTAL_PIXELS * sizeof(float)) / (1024 * 1024) << " MB" << std::endl;
    
    return true;
}

int main(int argc, char** argv) {
    std::cout << "=====================================" << std::endl;
    std::cout << "  CSV to Binary Converter v1.0" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Default paths
    std::string inputFolder = ".";
    std::string rainfallOutput = "Earth_Rainfall.bin";
    std::string elevationOutput = "Earth_Elevation.bin";
    
    // Parse command line arguments
    if (argc >= 2) {
        inputFolder = argv[1];
    }
    if (argc >= 3) {
        rainfallOutput = argv[2];
    }
    if (argc >= 4) {
        elevationOutput = argv[3];
    }
    
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Input folder: " << inputFolder << std::endl;
    std::cout << "  Rainfall output: " << rainfallOutput << std::endl;
    std::cout << "  Elevation output: " << elevationOutput << std::endl;
    std::cout << std::endl;
    
    // Convert rainfall data
    if (!convertRainfallToBinary(inputFolder, rainfallOutput)) {
        std::cerr << "\n✗ FAILED: Rainfall conversion" << std::endl;
        return 1;
    }
    
    // Convert elevation data
    if (!convertElevationToBinary(inputFolder, elevationOutput)) {
        std::cerr << "\n✗ FAILED: Elevation conversion" << std::endl;
        return 1;
    }
    
    std::cout << "\n=====================================" << std::endl;
    std::cout << "  ✓ ALL CONVERSIONS COMPLETE!" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "\nOutput files:" << std::endl;
    std::cout << "  • " << rainfallOutput << " (~311 MB)" << std::endl;
    std::cout << "  • " << elevationOutput << " (~32 MB)" << std::endl;
    std::cout << "\nTotal conversion time: ~15 minutes" << std::endl;
    std::cout << "These binary files can now be used by RainfallGradientAnalyzer" << std::endl;
    std::cout << "for fast loading (~10 seconds vs 6 hours!)" << std::endl;
    
    return 0;
}
