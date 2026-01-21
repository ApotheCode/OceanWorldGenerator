//============================================================
// SSTAnalyzer.cpp
// Advanced SST analysis to diagnose gyre patterns
// Creates multiple diagnostic visualizations
//============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "ClimateFileFormat.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

constexpr int WIDTH = STANDARD_WIDTH;
constexpr int HEIGHT = STANDARD_HEIGHT;
constexpr int MONTHS = STANDARD_MONTHS;
constexpr int VARIABLES = STANDARD_VARIABLES;
constexpr int VAR_SST = 1;

struct RGB { uint8_t r, g, b; };

//============================================================
// Get SST value
//============================================================
double getSSTValue(const std::vector<double>& data, int x, int y, int month)
{
    size_t cellIdx = y * WIDTH + x;
    size_t cellBase = cellIdx * VARIABLES * MONTHS;
    size_t dataIdx = cellBase + VAR_SST * MONTHS + month;
    return data[dataIdx];
}

//============================================================
// 1. ZONAL ANOMALY: SST - Zonal Mean (shows gyres)
//============================================================
void generateZonalAnomalyMap(const std::vector<double>& climateData,
                             int month,
                             const std::string& outputFile)
{
    std::cout << "\n🔍 ZONAL ANOMALY (SST minus latitude average)\n";
    std::cout << "   This removes the basic pole-to-equator gradient\n";
    std::cout << "   and shows ONLY the circulation patterns!\n\n";
    
    // Calculate zonal mean for each latitude
    std::vector<double> zonalMean(HEIGHT, 0.0);
    std::vector<int> zonalCount(HEIGHT, 0);
    
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double sst = getSSTValue(climateData, x, y, month);
            if (std::isfinite(sst)) {
                zonalMean[y] += sst;
                zonalCount[y]++;
            }
        }
        if (zonalCount[y] > 0) {
            zonalMean[y] /= zonalCount[y];
        }
    }
    
    // Calculate anomalies
    std::vector<double> anomalies(WIDTH * HEIGHT);
    double maxAnom = -1e9, minAnom = 1e9;
    
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double sst = getSSTValue(climateData, x, y, month);
            double anom = sst - zonalMean[y];
            anomalies[y * WIDTH + x] = anom;
            
            if (std::isfinite(anom)) {
                maxAnom = std::max(maxAnom, anom);
                minAnom = std::min(minAnom, anom);
            }
        }
    }
    
    std::cout << "   Anomaly range: " << minAnom << "°C to " << maxAnom << "°C\n";
    std::cout << "   Total range: " << (maxAnom - minAnom) << "°C\n";
    
    // Generate high-contrast image
    std::vector<uint8_t> image(WIDTH * HEIGHT * 3);
    
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double anom = anomalies[y * WIDTH + x];
            
            // High contrast diverging colormap: blue (cold) - white (neutral) - red (warm)
            double t = (anom - minAnom) / (maxAnom - minAnom);
            t = std::max(0.0, std::min(1.0, t));
            
            RGB color;
            if (t < 0.5) {
                // Cold anomaly: deep blue to white
                double local = t * 2.0;
                color.r = static_cast<uint8_t>(0 + (255 - 0) * local);
                color.g = static_cast<uint8_t>(100 + (255 - 100) * local);
                color.b = static_cast<uint8_t>(255);
            } else {
                // Warm anomaly: white to deep red
                double local = (t - 0.5) * 2.0;
                color.r = static_cast<uint8_t>(255);
                color.g = static_cast<uint8_t>(255 + (100 - 255) * local);
                color.b = static_cast<uint8_t>(255 + (0 - 255) * local);
            }
            
            size_t idx = (y * WIDTH + x) * 3;
            image[idx + 0] = color.r;
            image[idx + 1] = color.g;
            image[idx + 2] = color.b;
        }
    }
    
    stbi_write_png(outputFile.c_str(), WIDTH, HEIGHT, 3, image.data(), WIDTH * 3);
    std::cout << "   ✅ Saved: " << outputFile << "\n";
}

//============================================================
// 2. LATITUDE PROFILE: Show subtropical peaks
//============================================================
void generateLatitudeProfile(const std::vector<double>& climateData,
                             int month,
                             const std::string& outputFile)
{
    std::cout << "\n📊 LATITUDE PROFILE\n";
    std::cout << "   Shows variation vs latitude (should peak at ±30°)\n\n";
    
    // Calculate zonal mean and standard deviation for each latitude
    std::vector<double> zonalMean(HEIGHT, 0.0);
    std::vector<double> zonalStd(HEIGHT, 0.0);
    std::vector<int> zonalCount(HEIGHT, 0);
    
    // First pass: mean
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double sst = getSSTValue(climateData, x, y, month);
            if (std::isfinite(sst)) {
                zonalMean[y] += sst;
                zonalCount[y]++;
            }
        }
        if (zonalCount[y] > 0) {
            zonalMean[y] /= zonalCount[y];
        }
    }
    
    // Second pass: standard deviation
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double sst = getSSTValue(climateData, x, y, month);
            if (std::isfinite(sst)) {
                double diff = sst - zonalMean[y];
                zonalStd[y] += diff * diff;
            }
        }
        if (zonalCount[y] > 1) {
            zonalStd[y] = std::sqrt(zonalStd[y] / (zonalCount[y] - 1));
        }
    }
    
    // Create profile image (1800 height, 600 width for graph)
    int imgWidth = 600;
    int imgHeight = HEIGHT;
    std::vector<uint8_t> image(imgWidth * imgHeight * 3, 255);  // White background
    
    // Find max std for scaling
    double maxStd = *std::max_element(zonalStd.begin(), zonalStd.end());
    
    std::cout << "   Max zonal standard deviation: " << maxStd << "°C\n";
    std::cout << "   (Higher = more longitudinal variation = gyres)\n";
    
    // Draw std dev profile
    for (int y = 0; y < HEIGHT; ++y) {
        int barWidth = static_cast<int>((zonalStd[y] / maxStd) * (imgWidth - 100));
        
        // Color based on expected gyre locations
        double lat = 90.0 - (y * 180.0 / HEIGHT);
        bool isSubtropical = (fabs(lat - 30.0) < 10.0 || fabs(lat + 30.0) < 10.0);
        bool isSubpolar = (fabs(lat - 60.0) < 10.0 || fabs(lat + 60.0) < 10.0);
        
        RGB barColor;
        if (isSubtropical) {
            barColor = {255, 0, 0};  // Red: should be HIGH here
        } else if (isSubpolar) {
            barColor = {0, 0, 255};  // Blue: moderate here
        } else {
            barColor = {100, 100, 100};  // Gray: low elsewhere
        }
        
        for (int x = 50; x < 50 + barWidth; ++x) {
            if (x < imgWidth) {
                size_t idx = (y * imgWidth + x) * 3;
                image[idx + 0] = barColor.r;
                image[idx + 1] = barColor.g;
                image[idx + 2] = barColor.b;
            }
        }
    }
    
    // Draw latitude markers
    for (int lat = -90; lat <= 90; lat += 30) {
        int y = static_cast<int>((90.0 - lat) * HEIGHT / 180.0);
        if (y >= 0 && y < HEIGHT) {
            for (int x = 0; x < imgWidth; ++x) {
                size_t idx = (y * imgWidth + x) * 3;
                image[idx + 0] = 0;
                image[idx + 1] = 0;
                image[idx + 2] = 0;
            }
        }
    }
    
    stbi_write_png(outputFile.c_str(), imgWidth, imgHeight, 3, image.data(), imgWidth * 3);
    std::cout << "   ✅ Saved: " << outputFile << "\n";
    std::cout << "   RED bars = expected subtropical gyre peaks (±30°)\n";
    std::cout << "   BLUE bars = expected subpolar gyres (±60°)\n";
}

//============================================================
// 3. HEMISPHERIC COMPARISON: North vs South
//============================================================
void generateHemisphericComparison(const std::vector<double>& climateData,
                                   int month,
                                   const std::string& outputFile)
{
    std::cout << "\n🌍 HEMISPHERIC COMPARISON\n";
    std::cout << "   Shows Northern vs Southern hemisphere patterns\n";
    std::cout << "   Should look DIFFERENT if gyres are working!\n\n";
    
    int halfHeight = HEIGHT / 2;
    std::vector<uint8_t> image(WIDTH * HEIGHT * 3);
    
    // Calculate zonal means for anomaly
    std::vector<double> zonalMean(HEIGHT, 0.0);
    for (int y = 0; y < HEIGHT; ++y) {
        double sum = 0.0;
        int count = 0;
        for (int x = 0; x < WIDTH; ++x) {
            double sst = getSSTValue(climateData, x, y, month);
            if (std::isfinite(sst)) {
                sum += sst;
                count++;
            }
        }
        zonalMean[y] = (count > 0) ? sum / count : 0.0;
    }
    
    // Find anomaly range
    double maxAnom = -1e9, minAnom = 1e9;
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double sst = getSSTValue(climateData, x, y, month);
            double anom = sst - zonalMean[y];
            if (std::isfinite(anom)) {
                maxAnom = std::max(maxAnom, anom);
                minAnom = std::min(minAnom, anom);
            }
        }
    }
    
    // Generate image
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double sst = getSSTValue(climateData, x, y, month);
            double anom = sst - zonalMean[y];
            
            // High contrast anomaly colors
            double t = (anom - minAnom) / (maxAnom - minAnom);
            t = std::max(0.0, std::min(1.0, t));
            
            RGB color;
            if (t < 0.5) {
                double local = t * 2.0;
                color.r = static_cast<uint8_t>(0 + (255 - 0) * local);
                color.g = static_cast<uint8_t>(100 + (255 - 100) * local);
                color.b = 255;
            } else {
                double local = (t - 0.5) * 2.0;
                color.r = 255;
                color.g = static_cast<uint8_t>(255 + (100 - 255) * local);
                color.b = static_cast<uint8_t>(255 + (0 - 255) * local);
            }
            
            size_t idx = (y * WIDTH + x) * 3;
            image[idx + 0] = color.r;
            image[idx + 1] = color.g;
            image[idx + 2] = color.b;
        }
    }
    
    // Draw equator line
    int equatorY = HEIGHT / 2;
    for (int x = 0; x < WIDTH; ++x) {
        for (int dy = -2; dy <= 2; ++dy) {
            int y = equatorY + dy;
            if (y >= 0 && y < HEIGHT) {
                size_t idx = (y * WIDTH + x) * 3;
                image[idx + 0] = 0;
                image[idx + 1] = 255;
                image[idx + 2] = 0;
            }
        }
    }
    
    stbi_write_png(outputFile.c_str(), WIDTH, HEIGHT, 3, image.data(), WIDTH * 3);
    std::cout << "   ✅ Saved: " << outputFile << "\n";
    std::cout << "   Green line = Equator\n";
    std::cout << "   Patterns should be ASYMMETRIC across equator\n";
}

//============================================================
// Main
//============================================================
int main(int argc, char** argv)
{
    std::string inputFile = "output/OceanWorld_PNPL.bin";
    std::string prefix = "output/Analysis";
    int month = 0;
    
    std::cout << "🔬 SST ANALYZER - Diagnose Gyre Patterns\n";
    std::cout << "═══════════════════════════════════════════\n\n";
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            inputFile = argv[++i];
        }
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            prefix = argv[++i];
        }
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            month = atoi(argv[++i]) - 1;
        }
    }
    
    std::cout << "Input: " << inputFile << "\n";
    std::cout << "Month: " << (month + 1) << "\n\n";
    
    // Load data
    std::vector<double> climateData;
    ClimateFileHeader header;
    
    if (!loadClimateData(inputFile, climateData, header)) {
        std::cerr << "❌ Failed to load\n";
        return 1;
    }
    
    std::cout << "✅ Loaded " << (climateData.size() * sizeof(double) / (1024.0*1024.0)) << " MB\n";
    
    // Generate diagnostic visualizations
    generateZonalAnomalyMap(climateData, month, prefix + "_ZonalAnomaly.png");
    generateLatitudeProfile(climateData, month, prefix + "_LatitudeProfile.png");
    generateHemisphericComparison(climateData, month, prefix + "_Hemispheric.png");
    
    std::cout << "\n═══════════════════════════════════════════\n";
    std::cout << "✅ ANALYSIS COMPLETE!\n\n";
    std::cout << "📊 Review these files:\n";
    std::cout << "   1. " << prefix << "_ZonalAnomaly.png (shows gyres clearly)\n";
    std::cout << "   2. " << prefix << "_LatitudeProfile.png (should peak at ±30°)\n";
    std::cout << "   3. " << prefix << "_Hemispheric.png (N vs S asymmetry)\n\n";
    std::cout << "🔍 What to look for:\n";
    std::cout << "   ✅ Red/blue patches in ZonalAnomaly (not random noise)\n";
    std::cout << "   ✅ RED bars tallest at ±30° in LatitudeProfile\n";
    std::cout << "   ✅ Different patterns above/below green line in Hemispheric\n\n";
    
    return 0;
}
