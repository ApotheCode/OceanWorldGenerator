# PNPL v3 → v4 Upgrade: Basin-Aware Ocean Gyres

## Key Changes

### v3 (Hemispherical Gyres)
- **Global approach**: Gyres span entire hemispheres
- **Latitude-only**: Gyres placed at fixed latitude bands (20-40°N, 40-60°N)
- **No basin awareness**: Gyres extend across continents
- **Inputs**: Climate data + land mask

### v4 (Basin-Aware Gyres)
- **Basin-specific**: Each ocean basin gets its own gyre(s)
- **Dynamic placement**: Gyres placed at basin centroids
- **Confined circulation**: Gyres respect ocean basin boundaries
- **Inputs**: Climate data + bathymetry + basin map

## New Features

### 1. Basin Detection & Analysis
```cpp
analyzeBasins(basinMap, bathymetry)
```
- Computes centroid, latitude range, size for each basin
- Determines gyre type (subtropical, subpolar) per basin
- Filters out small basins (< 1000 pixels)

### 2. Per-Basin Gyre Placement
```cpp
struct BasinInfo {
    int basinID;
    double centroidLat, centroidLon;
    double minLat, maxLat, minLon, maxLon;
    bool hasSubtropicalGyre;  // Spans 20-40°?
    bool hasSubpolarGyre;     // Spans 40-60°?
    double gyreCenterLat, gyreCenterLon;
}
```

### 3. Basin Boundary Enforcement
- Pixels outside their basin get zero velocity
- Gyres confined to basin shape (Atlantic, Pacific, Indian Ocean)
- Handles multiple gyres per basin (e.g., North Pacific subtropical + subpolar)

## Usage

### Command Line
```bash
./PNPLGenerator_v4_BasinAware \
    -i Earth_Climate.bin \
    -b Bathymetry.bath \
    --basins Basins.basin \
    -o Earth_Climate_PNPL_v4.bin \
    --intensity 0.05 \
    --rotation 24.0
```

### Expected Files

**Bathymetry.bath** (28-byte header + float data)
- Header: magic(4) + width(4) + height(4) + channels(4=1) + dtype(4) + version(4) + depth(4)
- Data: float depths in meters (negative = ocean, positive = land)

**Basins.basin** (28-byte header + uint8 basin IDs)
- Header: Same 28-byte standard
- Data: uint8 basin IDs (0=land, 1-31=ocean basins)
- Basin IDs should be numbered (1=Atlantic, 2=Pacific, etc.)

## Physical Improvements

### Subtropical Gyres (v4 only)
- Centered at basin-specific locations (not global 30°)
- Size scales with basin extent (larger basin = larger gyre)
- Western intensification relative to basin's western boundary
- Examples: Gulf Stream (North Atlantic), Kuroshio (North Pacific)

### Subpolar Gyres (v4 only)
- Placed at ~50° if basin extends to subpolar latitudes
- Weaker than subtropical gyres (0.25 vs 0.50 strength)
- Opposite rotation to subtropical gyres
- Examples: Labrador Current, Norwegian Sea

### Multi-Gyre Basins
- Large basins (Pacific, Atlantic) can have BOTH:
  - Subtropical gyre at 30°
  - Subpolar gyre at 50°
- Each gyre confined to its basin

## Code Structure

### New Functions
```cpp
loadBathymetry()        // Load depth data
loadBasinMap()          // Load basin IDs
analyzeBasins()         // Compute gyre placement
basinGyrePattern()      // Device function: basin-specific gyres
inBasin()               // Device function: check pixel basin membership
```

### Modified Kernel
```cpp
applyBasinPNPLKernel()
```
- Takes basin map and basin info array
- Checks pixel's basin ID
- Generates gyre pattern for that specific basin
- Enforces basin boundaries

## Performance

- **CPU Analysis**: Basin analysis runs once at startup (~0.1s)
- **GPU Kernel**: Similar performance to v3 (added basin lookup is minimal)
- **Memory**: +~3.6 MB for basin map (3600x1800 uint8)

## Expected Output

For a planet with realistic ocean basins:
- **North Atlantic**: Subtropical gyre at 30°N (Gulf Stream-like)
- **North Pacific**: Subtropical gyre at 30°N + subpolar gyre at 50°N
- **South Atlantic**: Subtropical gyre at 30°S
- **South Pacific**: Subtropical gyre at 30°S
- **Indian Ocean**: Subtropical gyre at 30°S
- Small basins (Mediterranean, Red Sea): No gyres (too small)

## Testing

1. Verify basin file loads correctly
2. Check basin analysis output (console shows basin stats)
3. Look for gyre placement per basin
4. Validate SST patterns show basin-confined circulation

## Next Steps

If you want to visualize:
1. Export basin-specific SST for each month
2. Check gyre confinement (no circulation across continents)
3. Validate western intensification per basin
4. Compare v3 (global) vs v4 (basin-aware) outputs

Compile:
```bash
nvcc -O3 -std=c++17 PNPLGenerator_v4_BasinAware.cu -o PNPLGenerator_v4_BasinAware
```
