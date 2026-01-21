#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

constexpr size_t WIDTH = 3600;
constexpr size_t HEIGHT = 1800;
constexpr size_t MONTHS = 12;
constexpr size_t TOTAL_CELLS = WIDTH * HEIGHT;

struct FClimateCell {
    double SST[MONTHS];
    double Rainfall[MONTHS];
    double AirTemp[MONTHS];
    double Albedo[MONTHS];
    double NDVI[MONTHS];
    double NetFlux[MONTHS];
    double SWFlux[MONTHS];
    double LWFlux[MONTHS];
    double Insolation[MONTHS];
    double SeaIceMask[MONTHS];
};

// Names for readability
const std::vector<std::string> FIELD_NAMES = {
    "SST",
    "Rainfall",
    "AirTemp",
    "Albedo",
    "NDVI",
    "NetFlux",
    "SWFlux",
    "LWFlux",
    "Insolation",
    "SeaIceMask"
};

int main(int argc, char** argv)
{
    std::string filename = (argc > 1) ? argv[1] : "climate_output/OceanWorld_Climate.bin";

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

    size_t expectedSize = TOTAL_CELLS * sizeof(FClimateCell);
    std::cout << "📦 File size: " << (fileSize / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    std::cout << "📐 Expected: " << (expectedSize / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;

    if (fileSize != expectedSize)
    {
        std::cerr << "⚠️ File size mismatch! Aborting diagnostic.\n";
        return 2;
    }

    // Read entire file
    std::vector<FClimateCell> grid(TOTAL_CELLS);
    file.read(reinterpret_cast<char*>(grid.data()), fileSize);
    file.close();

    std::cout << "✅ Successfully loaded " << grid.size() << " climate cells.\n\n";

    // ---- Compute per-variable monthly statistics ----
    for (size_t f = 0; f < FIELD_NAMES.size(); ++f)
    {
        std::cout << "\n🌍 Variable: " << FIELD_NAMES[f] << std::endl;

        for (int m = 0; m < MONTHS; ++m)
        {
            double minVal = 1e30, maxVal = -1e30, sum = 0.0, sqsum = 0.0;
            size_t count = 0;

            for (const auto& cell : grid)
            {
                double val = 0.0;
                switch (f)
                {
                    case 0: val = cell.SST[m]; break;
                    case 1: val = cell.Rainfall[m]; break;
                    case 2: val = cell.AirTemp[m]; break;
                    case 3: val = cell.Albedo[m]; break;
                    case 4: val = cell.NDVI[m]; break;
                    case 5: val = cell.NetFlux[m]; break;
                    case 6: val = cell.SWFlux[m]; break;
                    case 7: val = cell.LWFlux[m]; break;
                    case 8: val = cell.Insolation[m]; break;
                    case 9: val = cell.SeaIceMask[m]; break;
                }

                if (std::isnan(val)) continue;

                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
                sum += val;
                sqsum += val * val;
                count++;
            }

            double mean = sum / count;
            double stddev = std::sqrt((sqsum / count) - (mean * mean));

            std::cout << "📅 Month " << std::setw(2) << (m + 1)
                      << " | Min: " << std::setw(10) << minVal
                      << " | Max: " << std::setw(10) << maxVal
                      << " | Mean: " << std::setw(10) << mean
                      << " | StdDev: " << std::setw(10) << stddev
                      << std::endl;
        }
    }

    // ---- Optional sample lookup ----
    int lon = 0, lat = 0, var = 0, month = 0;
    std::cout << "\n🔍 Enter sample (lon 0–3599, lat 0–1799, variable 0–9, month 1–12): ";
    if (std::cin >> lon >> lat >> var >> month)
    {
        if (lon >= 0 && lon < WIDTH && lat >= 0 && lat < HEIGHT &&
            var >= 0 && var < 10 && month >= 1 && month <= 12)
        {
            size_t idx = static_cast<size_t>(lat * WIDTH + lon);
            double value = 0.0;
            const auto& c = grid[idx];
            switch (var)
            {
                case 0: value = c.SST[month - 1]; break;
                case 1: value = c.Rainfall[month - 1]; break;
                case 2: value = c.AirTemp[month - 1]; break;
                case 3: value = c.Albedo[month - 1]; break;
                case 4: value = c.NDVI[month - 1]; break;
                case 5: value = c.NetFlux[month - 1]; break;
                case 6: value = c.SWFlux[month - 1]; break;
                case 7: value = c.LWFlux[month - 1]; break;
                case 8: value = c.Insolation[month - 1]; break;
                case 9: value = c.SeaIceMask[month - 1]; break;
            }
            std::cout << "Value at (" << lon << ", " << lat << ") "
                      << FIELD_NAMES[var] << " month " << month
                      << " = " << value << std::endl;
        }
        else
        {
            std::cout << "Invalid input range.\n";
        }
    }

    std::cout << "\n✅ Diagnostic complete.\n";
    return 0;
}
