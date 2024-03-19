#pragma once 

#include <iostream>
#include <vector>
#include <climits>

void print_image(float* start){
    for (int i = 0; i < 28; i++){
        for (int j = 0; j < 28; j++)
            if (start[28 * i + j] > 0.25)
                std::cout << "█";
            else 
                std::cout << ' ';
        
        std::cout << std::endl;    
    }
}

// Deconstructor frees data and label, don't leave uninitialised
struct DigitData{
    float* data;
    char*  labels;
    int    data_size;
    int    label_size;
    int    data_count;
    int    label_count;

    ~DigitData(){
        delete[] data;
        delete[] labels;
    }
};

// https://stackoverflow.com/a/4956493
template <typename T>
T swap_endian(T u)
{
    static_assert (CHAR_BIT == 8, "CHAR_BIT != 8");

    union
    {
        T u;
        unsigned char u8[sizeof(T)];
    } source, dest;

    source.u = u;

    for (size_t k = 0; k < sizeof(T); k++)
        dest.u8[k] = source.u8[sizeof(T) - k - 1];

    return dest.u;
}


DigitData load_digits(){
    // Images stored as:
    // [offset] [type]          [value]          [description]
    // 0000     32 bit integer  0x00000803(2051) magic number
    // 0004     32 bit integer  10000            number of images
    // 0008     32 bit integer  28               number of rows
    // 0012     32 bit integer  28               number of columns
    // 0016     unsigned byte   ??               pixel
    // 0017     unsigned byte   ??               pixel
    // ........
    DigitData          out;
    unsigned char byte;

    FILE* imagef = fopen("../Data/Digits/images", "rb");
    int   iheader[4];
    
    // Read header and allocate space for images
    fread(iheader, sizeof(int), 4, imagef);
    for (int i = 0; i < 4; i++)
        iheader[i] = swap_endian(iheader[i]);
    
    out.data_count = iheader[1];
    out.data_size  = iheader[2] * iheader[3];
    out.data       = new float[out.data_count * out.data_size]();

    // Read greyscale pixels 
    for (int i = 0; fread(&byte, 1, 1, imagef); i++)
        out.data[i] = (float)swap_endian(byte) / 255.0f;

    
    // Labels stored as:
    // [offset] [type]          [value]          [description]
    // 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    // 0004     32 bit integer  10000            number of items
    // 0008     unsigned byte   ??               label
    // 0009     unsigned byte   ??               label
    // ........

    FILE* labelf = fopen("../Data/Digits/labels", "rb");
    int   lheader[2];

    fread(lheader, sizeof(int), 2, labelf);
    for (int i = 0; i < 2; i++)
        lheader[i] = swap_endian(lheader[i]);
    
    out.label_count = lheader[1];
    out.label_size  = 1;
    out.labels      = new char[out.label_count * out.label_size]();
    
    for (int i = 0; fread(&byte, 1, 1, labelf); i++)
        out.labels[i] = swap_endian(byte);
        
    return out;
}
