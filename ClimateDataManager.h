#pragma once

#include <string>
#include <vector>

// Stores all monthly climate values for a single grid cell
struct FClimateCell {
    double SST[12];
    double Rainfall[12];
    double AirTemp[12];
    double NDVI[12];
    double Albedo[12];
    double NetFlux[12];
    double SWFlux[12];
    double LWFlux[12];
    double Insolation[12];
    double SeaIceMask[12];
};

class FClimateDataManager {
public:
    bool LoadFromFile(const std::string& FilePath, int InWidth = 3600, int InHeight = 1800);
    
    // Retrieves a monthly value by lat/lon
    bool GetClimateValue(double Latitude, double Longitude, const std::string& Variable, int Month, double& OutValue) const;

    // Direct access by cell index (latIndex, lonIndex)
    const FClimateCell& GetCell(int X, int Y) const;

private:
    int Width = 3600;
    int Height = 1800;
    std::vector<FClimateCell> ClimateGrid;

    int LatLonToIndex(double Lat, double Lon) const;
};
