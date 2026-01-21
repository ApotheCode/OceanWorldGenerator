#include "ClimateDataManager.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>


bool FClimateDataManager::LoadFromFile(const std::string& FilePath, int InWidth, int InHeight) {
    Width = InWidth;
    Height = InHeight;

    std::ifstream file(FilePath, std::ios::binary | std::ios::in);
    if (!file.is_open()) {
        std::cout << "Failed to open climate data file:"<< FilePath << std::endl;
        return false;
    }

    ClimateGrid.resize(Width * Height);
    file.read(reinterpret_cast<char*>(ClimateGrid.data()), ClimateGrid.size() * sizeof(FClimateCell));

    file.close();
    return true;
}

int FClimateDataManager::LatLonToIndex(double Lat, double Lon) const {
    // Clamp to [-90, 90] and [-180, 180]
    Lat = std::clamp(Lat, -90.0f, 90.0f);
    Lon = fmod(Lon + 180.0f, 360.0f); // [0, 360)
    if (Lon < 0) Lon += 360.0f;

    int y = static_cast<int>((90.0f - Lat) * Height / 180.0f);
    int x = static_cast<int>(Lon * Width / 360.0f);

    y = std::clamp(y, 0, Height - 1);
    x = std::clamp(x, 0, Width - 1);
    return y * Width + x;
}

const FClimateCell& FClimateDataManager::GetCell(int X, int Y) const {
    X = std::clamp(X, 0, Width - 1);
    Y = std::clamp(Y, 0, Height - 1);
    return ClimateGrid[Y * Width + X];
}

bool FClimateDataManager::GetClimateValue(double Latitude, double Longitude, const std::string& Variable, int Month, double& OutValue) const {
    if (Month < 0 || Month > 11) return false;

    int index = LatLonToIndex(Latitude, Longitude);
    const FClimateCell& Cell = ClimateGrid[index];

    if (Variable == "SST") OutValue = Cell.SST[Month];
    else if (Variable == "Rainfall") OutValue = Cell.Rainfall[Month];
    else if (Variable == "AirTemp") OutValue = Cell.AirTemp[Month];
    else if (Variable == "NDVI") OutValue = Cell.NDVI[Month];
    else if (Variable == "Albedo") OutValue = Cell.Albedo[Month];
    else if (Variable == "NetFlux") OutValue = Cell.NetFlux[Month];
    else if (Variable == "SWFlux") OutValue = Cell.SWFlux[Month];
    else if (Variable == "LWFlux") OutValue = Cell.LWFlux[Month];
    else if (Variable == "Insolation") OutValue = Cell.Insolation[Month];
    else if (Variable == "SeaIceMask") OutValue = Cell.SeaIceMask[Month];
    else return false;

    return true;
}
