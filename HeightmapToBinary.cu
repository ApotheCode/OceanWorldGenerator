#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"

// GPU kernel to scale and position heightmap
__global__ void scaleAndPositionKernel(
    const unsigned char* input,
    int inputWidth,
    int inputHeight,
    float* elevation,
    unsigned char* landMask,
    int outputWidth,
    int outputHeight,
    int startX,
    int startY,
    int scaledWidth,
    int scaledHeight,
    float maxElevation,
    float seaLevelThreshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= outputWidth || y >= outputHeight) return;
    
    int idx = y * outputWidth + x;
    
    // Check if this pixel is within the placed terrain bounds
    if (x >= startX && x < startX + scaledWidth &&
        y >= startY && y < startY + scaledHeight) {
        
        // Map to input coordinates using bilinear interpolation
        float srcX = (float)(x - startX) * inputWidth / scaledWidth;
        float srcY = (float)(y - startY) * inputHeight / scaledHeight;
        
        int x0 = (int)srcX;
        int y0 = (int)srcY;
        int x1 = min(x0 + 1, inputWidth - 1);
        int y1 = min(y0 + 1, inputHeight - 1);
        
        float fx = srcX - x0;
        float fy = srcY - y0;
        
        // Sample input (assuming grayscale or taking first channel)
        float v00 = input[y0 * inputWidth + x0] / 255.0f;
        float v10 = input[y0 * inputWidth + x1] / 255.0f;
        float v01 = input[y1 * inputWidth + x0] / 255.0f;
        float v11 = input[y1 * inputWidth + x1] / 255.0f;
        
        // Bilinear interpolation
        float v0 = v00 * (1 - fx) + v10 * fx;
        float v1 = v01 * (1 - fx) + v11 * fx;
        float value = v0 * (1 - fy) + v1 * fy;
        
        // Set elevation
        elevation[idx] = value * maxElevation;
        
        // Set land mask
        landMask[idx] = (value > seaLevelThreshold) ? 1 : 0;
    } else {
        // Ocean pixels
        elevation[idx] = 0.0f;
        landMask[idx] = 0;
    }
}

int main(int argc, char** argv) {
    // Configuration
    const char* inputFile = "Normalized Kyushu 1024.png";
    const char* outputFile = "output/TerrainData.bin";
    
    const int outputWidth = 3600;
    const int outputHeight = 1800;
    
    // Latitude configuration
    float northLat = 75.0f;  // Northern boundary
    float southLat = 15.0f;  // Southern boundary (SOUTHERN HEMISPHERE!)
    float centerLon = 0.0f; // Center longitude
    
    // Elevation parameters
    float maxElevation = 3000.0f; // meters
    float seaLevelThreshold = 0.02f; // Values below this = ocean
    
    // Command line overrides
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            inputFile = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            outputFile = argv[++i];
        } else if (strcmp(argv[i], "--north") == 0 && i + 1 < argc) {
            northLat = atof(argv[++i]);
        } else if (strcmp(argv[i], "--south") == 0 && i + 1 < argc) {
            southLat = atof(argv[++i]);
        } else if (strcmp(argv[i], "--center-lon") == 0 && i + 1 < argc) {
            centerLon = atof(argv[++i]);
        } else if (strcmp(argv[i], "--max-elevation") == 0 && i + 1 < argc) {
            maxElevation = atof(argv[++i]);
        } else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            seaLevelThreshold = atof(argv[++i]);
        }
    }
    
    std::cout << "=== Heightmap to Binary Converter ===" << std::endl;
    std::cout << "Input: " << inputFile << std::endl;
    std::cout << "Output: " << outputFile << std::endl;
    std::cout << "Latitude range: " << northLat << "N to " << southLat << "S" << std::endl;
    std::cout << "Center longitude: " << centerLon << std::endl;
    std::cout << "Max elevation: " << maxElevation << " meters" << std::endl;
    
    // Load input image
    int inputWidth, inputHeight, channels;
    unsigned char* inputImage = stbi_load(inputFile, &inputWidth, &inputHeight, &channels, 1);
    
    if (!inputImage) {
        std::cerr << "ERROR: Failed to load " << inputFile << std::endl;
        return 1;
    }
    
    std::cout << "Loaded heightmap: " << inputWidth << "x" << inputHeight << std::endl;
    
    // Calculate placement on output grid
    // Convert latitude to pixel coordinates (90N = 0, 90S = 1800)
    int startY = (int)((90.0f - northLat) * 10.0f);
    int endY = (int)((90.0f + southLat) * 10.0f); // Note: southLat is negative value stored as positive
    int scaledHeight = endY - startY;
    
    // Maintain aspect ratio
    float aspectRatio = (float)inputWidth / inputHeight;
    int scaledWidth = (int)(scaledHeight * aspectRatio);
    
    // Center on specified longitude
    int startX = (int)((centerLon + 180.0f) * 10.0f - scaledWidth / 2.0f);
    
    std::cout << "Placement: X=" << startX << " Y=" << startY << std::endl;
    std::cout << "Scaled size: " << scaledWidth << "x" << scaledHeight << std::endl;
    
    // Allocate output buffers
    size_t outputSize = outputWidth * outputHeight;
    float* h_elevation = new float[outputSize];
    unsigned char* h_landMask = new unsigned char[outputSize];
    
    // Allocate GPU memory
    unsigned char* d_input;
    float* d_elevation;
    unsigned char* d_landMask;
    
    cudaMalloc(&d_input, inputWidth * inputHeight);
    cudaMalloc(&d_elevation, outputSize * sizeof(float));
    cudaMalloc(&d_landMask, outputSize);
    
    // Copy input to GPU
    cudaMemcpy(d_input, inputImage, inputWidth * inputHeight, cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((outputWidth + block.x - 1) / block.x,
              (outputHeight + block.y - 1) / block.y);
    
    scaleAndPositionKernel<<<grid, block>>>(
        d_input, inputWidth, inputHeight,
        d_elevation, d_landMask,
        outputWidth, outputHeight,
        startX, startY,
        scaledWidth, scaledHeight,
        maxElevation, seaLevelThreshold
    );
    
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Copy results back
    cudaMemcpy(h_elevation, d_elevation, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_landMask, d_landMask, outputSize, cudaMemcpyDeviceToHost);
    
    // Write binary file
    std::ofstream outFile(outputFile, std::ios::binary);
    if (!outFile) {
        std::cerr << "ERROR: Cannot write to " << outputFile << std::endl;
        return 1;
    }
    
    // Write header
    int32_t header[2] = {outputWidth, outputHeight};
    outFile.write(reinterpret_cast<char*>(header), sizeof(header));
    
    // Write elevation data
    outFile.write(reinterpret_cast<char*>(h_elevation), outputSize * sizeof(float));
    
    // Write land mask
    outFile.write(reinterpret_cast<char*>(h_landMask), outputSize);
    
    outFile.close();
    
    // Calculate statistics
    int landPixels = 0;
    float minElev = 9999.0f, maxElev = -9999.0f;
    for (size_t i = 0; i < outputSize; i++) {
        if (h_landMask[i] == 1) {
            landPixels++;
            minElev = std::min(minElev, h_elevation[i]);
            maxElev = std::max(maxElev, h_elevation[i]);
        }
    }
    
    std::cout << "\n=== Statistics ===" << std::endl;
    std::cout << "Land pixels: " << landPixels << " (" 
              << (100.0f * landPixels / outputSize) << "%)" << std::endl;
    std::cout << "Elevation range: " << minElev << " to " << maxElev << " meters" << std::endl;
    std::cout << "\nTerrainData.bin written successfully!" << std::endl;
    
    // Cleanup
    stbi_image_free(inputImage);
    delete[] h_elevation;
    delete[] h_landMask;
    cudaFree(d_input);
    cudaFree(d_elevation);
    cudaFree(d_landMask);
    
    return 0;
}