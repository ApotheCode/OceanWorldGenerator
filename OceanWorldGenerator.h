#pragma once

#include <vector>
#include <string>

struct FClimateCell {
    float SST[12];
    float Rainfall[12];
    float AirTemp[12];
    float NDVI[12];
    float Albedo[12];
    float NetFlux[12];
    float SWFlux[12];
    float LWFlux[12];
    float Insolation[12];
    float SeaIceMask[12];
};
