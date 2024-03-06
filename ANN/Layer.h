#pragma once

#include <string>

class Layer{
protected:
    float *values  = nullptr;
    float *pre_act = nullptr;
    float *weights = nullptr;
    float *bias    = nullptr;   
    float  bias_c  = 1.0f;
    bool   has_bias;

    int nunits;
    int prev_nunits;
    int nweights;

    Layer *prev;

public:
    Layer(int nunits, bool bias);
    ~Layer();

    int    get_nunits() { return nunits; }
    float* get_values() { return values; }

    // Calc new values by feed forward
    virtual void update() {};
    // Connect Layer to prev during Network compilation
    virtual void connect(Layer* prev) {};

    virtual std::string to_string(){};
};
