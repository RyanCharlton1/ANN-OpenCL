// DEFINE PARAMS: FILTERH, FILTERW, CHANNELS, STRIDEX, STRIDEY, 
// PREVH, PREVW, OUTH, OUTW, FEATURES, BSIZE

// Perform convolution optimised by setting params as preprocessor defines
// and unrolling loops
// https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf


// Filters: features[filterh[filterw[channels]]]
// Values:  batches[prevh[prevw[channels]]]
// result:  batches[filtersy[filtersx[features]]]
__kernel
void convolution(
__global float* filters,
__global float* values,
__global float* result){

    int filtery = get_global_id(0);
    int filterx = get_global_id(1);
    int feature = get_global_id(2);

    int filtersy = get_global_size(0) / BSIZE;
    int filtersx = get_global_size(1);
    int features = get_global_size(2);

    int batch = filtery / filtersy;

    int result_index = (filtery * filtersx + filterx) * FEATURES + feature;

    filtery %= filtersy;

    // Index of filter being applied from filter list
    int filters_index = feature * FILTERH * FILTERW * CHANNELS;

    // Start of image within batch
    int image_start = batch * PREVH * PREVW;
    // Move to start of filter within image
    image_start += filtery * STRIDEY * PREVW + filterx * STRIDEX;

    float acc = 0.0f;
    
    #pragma unroll
    for (int y = 0; y < FILTERH; y++){
        // Move to row within filter
        int filtery = image_start + y * PREVW;

        #pragma unroll
        for (int x = 0; x < FILTERW; x++){
            // Move to column within filter
            int filterx = filtery + x;
            int filterc = filterx * CHANNELS;

            #pragma unroll
            for (int c = 0; c < CHANNELS; c++){
                acc += values[filterc + c] * filters[filters_index];
                filters_index++;
            }
        }
    }

    result[result_index] = acc;

#ifdef DEBUG
    printf("conv[%d]: %f\n", result_index, acc);
#endif
}

#define PADDEDCOLS (FILTERW + (OUTW - 1) * STRIDEX + FILTERW - 1)
#define PADDEDROWS (FILTERH + (OUTH - 1) * STRIDEY + FILTERH - 1)
#define PADDEDSIZE PADDEDCOLS * PADDEDROWS * FEATURES

#define GRADSROWSIZE OUTW * FEATURES

// value_grads batches[height[width[features]]]
__kernel
void pad_and_dilate(
__global float* value_grads,
__global float* output){

    int y = get_global_id(0);
    int x = get_global_id(1);
    int c = get_global_id(2);

    int grads_index;
    grads_index  = y * GRADSROWSIZE;
    grads_index += x * FEATURES;
    grads_index += c;
    
    int batch = y / OUTH;
    y %= OUTH;

    int output_index;
    output_index  = batch * PADDEDSIZE;
    output_index += (FILTERH - 1 + y * STRIDEY) * PADDEDCOLS * FEATURES;
    output_index += (FILTERW - 1 + x * STRIDEX) * FEATURES; 
    output_index += c;

    output[output_index] = value_grads[grads_index];

#ifdef DEBUG
    printf("pad[%d]: %d\n", output_index, grads_index);
#endif
}

#define DILATEDCOLS (1 + (OUTW - 1) * STRIDEX)
#define DILATEDROWS (1 + (OUTH - 1) * STRIDEY)
#define DILATEDSIZE DILATEDCOLS * DILATEDROWS * FEATURES

__kernel
void dilate(
__global float* input,
__global float* output){

    int y = get_global_id(0);
    int x = get_global_id(1);
    int c = get_global_id(2);

    int grads_index;
    grads_index  = y * GRADSROWSIZE;
    grads_index += x * FEATURES;
    grads_index += c;
    
    int batch = y / OUTH;
    y %= OUTH;

    int output_index;
    output_index  = batch * DILATEDSIZE;
    output_index += (y * STRIDEY) * DILATEDCOLS * FEATURES;
    output_index += (x * STRIDEX) * FEATURES; 
    output_index += c;

    output[output_index] = input[grads_index];

#ifdef DEBUG
    printf("pad[%d]: %d\n", output_index, grads_index);
#endif
}

#define FILTERSIZE FILTERH * FILTERW * CHANNELS

// Perform convolution on a padded output gradient image using reversed 
// filters to find the gradients at input  
// Filters: features[filterh[filterw[channels]]]
// Grads:   batches[paddedh[paddedw[features]]]
// Result:  batches[prevh[prevw[channels]]]
__kernel
void deconvolution(
__global float* filters,
__global float* value_grads,
__global float* result){

    int prevy   = get_global_id(0);
    int prevx   = get_global_id(1);
    int channel = get_global_id(2);

    int prevh    = get_global_size(0) / BSIZE;
    int prevw    = get_global_size(1);
    int channels = get_global_size(2);

    int batch = prevy / prevh;

    int result_index = (prevy * prevw + prevx) * CHANNELS + channel;

    prevy %= prevh;

    // First position of filter to use as filter is read backwards
    // and only one channel is going to be used to calculate a single
    // channel of the result
    int filter_index = FILTERSIZE - CHANNELS + channel;

    // Start of batch's grads image 0,0
    int image_start = batch * PADDEDROWS * PADDEDCOLS;
    // Start of filter
    int filter_start = image_start + prevy * PADDEDCOLS + prevx;
    
    float acc = 0.0f;
    #pragma unroll
    for (int y = 0; y < FILTERH; y++){
        int filtery = filter_start + y * PADDEDCOLS;

        #pragma unroll
        for (int x = 0; x < FILTERW; x++){
            int filterx = filtery + x;
            int filterf = filterx * FEATURES;

            #pragma unroll
            for (int f = 0; f < FEATURES; f++){
                int feature_index = filter_index + f * FILTERSIZE;

                acc += value_grads[filterf + f] * filters[feature_index];
            }

            filter_index -= CHANNELS;
        }
    }

    result[result_index] = acc;

#ifdef DEBUG
    printf("deconv[%d]: %f\n", result_index, acc);
#endif
}

#define PREVSIZE PREVH * PREVW * CHANNELS

// Weight gradients are the result of applying convolution to the input 
// vlaues using the output grads dilated by stride - 1 as the filter. 
// Values:  batches[prevh[prevw[channels]]]
// Grads:   batches[dilatedh[dilatedw[features]]]
// Result:  features[filterh[filterw[channels]]]
__kernel 
void convolution_weight_grads(
__global float* values,
__global float* padded_grads,
__global float* weight_grad){
    
    int weighty = get_global_id(0);
    int weightx = get_global_id(1);
    int channel = get_global_id(2);

    int filterh  = get_global_size(0);
    int filterw  = get_global_size(1);
    int features = get_global_size(2) / CHANNELS;

    int batch = weighty / FILTERH;
    weighty %= FILTERH;
    
    // Max 3 dimensions for work group, include channels and features 
    // as one dimension of features x channels
    int feature = channel / CHANNELS;
    channel %= CHANNELS;

    int values_start = (batch * PREVH + weighty) * PREVW + weightx;
    int grad_start   = batch * DILATEDROWS * DILATEDCOLS;

    float acc = 0.0f;

    #pragma unroll
    for (int y = 0; y < DILATEDROWS; y++){
        int valuesy = values_start + y * PREVW;
        int gradsy  = grad_start + y * DILATEDCOLS;

        #pragma unroll
        for (int x = 0; x < DILATEDCOLS; x++){
            int valuesx = valuesy + x;
            int valuesc = valuesx * CHANNELS + channel;    
            int gradsx  = gradsy + x;
            int gradsc  = gradsx * FEATURES + feature;
            
            acc += values[valuesc] * padded_grads[gradsc];
        }
    }

    int weight_index;
    weight_index  = batch * FILTERH * FILTERW * FEATURES;
    weight_index += (feature * FILTERH + weighty) * FILTERW + weightx;
    weight_index *= CHANNELS;
    weight_index += channel;

    weight_grad[weight_index] = acc;

#ifdef DEBUG 
    printf("conv_grad[%d]: %f\n", weight_index, acc);
#endif
}

__kernel 
void average_weight_grads(
__global float* weight_grads,
__global float* result){

    int weight_index = get_global_id(0);

    float acc = 0.0f;

    #pragma unroll
    for (int b = 0; b < BSIZE; b++)
        acc += weight_grads[weight_index + b * get_global_size(0)];
    
    result[weight_index] = acc / (float)BSIZE;
}