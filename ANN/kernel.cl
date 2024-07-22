//#define DEBUG

// [0, .. cols] [v1]
// [..        ] [v2]
// [rows      ] [v3] 
// Find row vector dot product each pass. Vec can be larger than matrix
// for batches, needs to be set in local workgroup size.
__kernel 
void mat_vec_mult(
         int    cols,
         int    rows,
__global float* mat,
__global float* vec_in,
__global float* vec_out){
    
    int g = get_global_id(0);

    // where to start reading input
    int start = cols * (g / rows);
    int row   = g % rows;

    float f = 0.0f;
    for (int i = 0; i < cols; i++){
        f += mat[row * cols + i] * vec_in[start + i];
    }

    vec_out[g] = f;

#ifdef DEBUG
    printf("mat_vec_mult[%d]:=%f\n", g, f);
#endif
}

// Multiply vec_a and vec_b, store in vec_c
__kernel 
void vec_vec_mult(
__global float* vec_a,
__global float* vec_b,
__global float* vec_c){

    int i = get_global_id(0);

    vec_c[i] = vec_a[i] * vec_b[i];

#ifdef DEBUG
    printf("vec_vec_mult[%d]:%f*%f\n", i, vec_a[i], vec_b[i]);
#endif
}

// Add two vectors and store in the first, second may be larger than first,
// it will repeat if second length is set as local work size.
__kernel
void vec_vec_add_inplace(
         int    bias_len,
__global float* values,
__global float* bias){

    int g = get_global_id(0);

    //printf("g=%d: %f += %f\n", g, values[g], bias[g%s]);
    values[g] += bias[g % bias_len];
}

__kernel
void vec_scalar_div(
__global float* vec,
         float  n){

    int i = get_global_id(0);

    vec[i] /= n;
}

// Similar to a cartesian product but multiply the elements
// dL/dy * dy/dw(values) takes two vectors and returns dL/dw, a matrix
// To avoid conccurrency problems for batches, each job calculates an
// element of the weight grad matrix 
__kernel 
void weight_grad(
         int    bsize,
         int    rows,
         int    cols,
__global float* loss,
__global float* prev_values,
__global float* weight_grad){

    int g   = get_global_id(0);
    int col = g % cols;

    float f = 0.0f;
    for (int b = 0; b < bsize; b++){
        //printf("[%d]+= %f * %f\n", g, loss[b * rows + g / cols], prev_values[b * cols + col]);
        f += loss[b * rows + g / cols] 
           * prev_values[b * cols + col];
    }
    
    
    f /= (float)bsize;

    weight_grad[g] = f;

#ifdef DEBUG
    printf("weight_grad[%d]=%f\n", g, f);
#endif
}

// Calculate each element of bias in one job, same as weight_grad
__kernel 
void bias_grad(
         int    bsize,
__global float* values_grad,
__global float* bias_grad){

    int i         = get_global_id(0);
    int nfeatures = get_global_size(0) / bsize;

    float f = 0.0f;
    for (int b = 0; b < bsize; b++)
        f += values_grad[nfeatures * b + i];

    f /= (float)bsize;

    bias_grad[i] = f;

#ifdef DEBUG
    printf("bias_grad[%d]=%f\n", i, f);
#endif
}

// Weights matrix is prev_nunits by nunits:
// [0, 1, ... prev_nunits]
// [1, ...               ]
// [...                  ]
// [nunits, ...          ]
// Because we're calculating backwards for loss grad, going from y to z:
// Transpose, read collumns instead of rows.
__kernel 
void mat_vec_mult_trans(
         int    rows,
         int    cols,
__global float* mat,
__global float* vec_in,
__global float* vec_out){
    
    int g = get_global_id(0);

    // where to start reading input
    int start = rows * (g / cols);
    int col   = g % cols;

    float f = 0.0f;
    for (int i = 0; i < rows; i++){
        f += mat[i * cols + col] * vec_in[start + i];
    }
    
    vec_out[g] = f;

#ifdef DEBUG
    printf("loss_grad[%d]:=%f\n", g, f);
#endif
}

// ReLU activation 
__kernel
void ReLU(
__global float* pre_act_values,
__global float* values){

    int g = get_global_id(0);

    values[g] = pre_act_values[g] > 0.0f ? pre_act_values[g] : 0.0f;
}

// Leaky ReLU activation
__kernel
void leaky_ReLU(
__global float* pre_act_values,
__global float* values){

    int g = get_global_id(0);

    values[g] = pre_act_values[g] * (pre_act_values[g] < 0.0f ? 0.1f : 1.0f);
}

// ReLU activation der
__kernel
void ReLU_der(
__global float* pre_act_values,
__global float* values_grad){

    int g = get_global_id(0);

    values_grad[g] = pre_act_values[g] > 0.0f ? 1.0 : 0.0f;
}

// Leaky ReLU activation der
__kernel
void leaky_ReLU_der(
__global float* pre_act_values,
__global float* values_grad){

    int g = get_global_id(0);

    values_grad[g] = pre_act_values[g] < 0.0f ? 0.1f : 1.0f;
}

// MSE loss, can do multiple batches
__kernel 
void MSE(
__global float* y,
__global float* y_,
__global float* x){

    int i = get_global_id(0);
    int n = get_global_size(0);
    
    float err = y[i] - y_[i];

    x[i] = err * err / (float)n / 2.0f;

#ifdef DEBUG
    printf("MSE[%d]:err=(%f - %f): %f / %d / 2.0f\n", i, y[i], y_[i], err*err, n);
#endif
}

// MSE loss der, needs no adjustment for multiple batches
__kernel 
void MSE_der(
         int    size,
__global float* y,
__global float* y_,
__global float* der){

    int i = get_global_id(0);

    float err = y[i] - y_[i];
    
    der[i] = -err / (float)size;

#ifdef DEBUG
    printf("MSE_der[%d]:%f = -%f / %f\n", i, der[i], err, (float)size);
#endif
}

// Softmax bottom sum, giving each batch one job is more efficient than
// making a mutex for the sum
__kernel
void softmax_sum(
         int    zsize,
__global float* sums,
__global float* z){

    int g = get_global_id(0);

    float sum = 0.0f;
    for (int i = 0; i < zsize; i++)
        sum += exp(z[g * zsize + i]);

    sums[g] = sum;

#ifdef DEBUG
    printf("softmax_sum[%d]=%f\n", g, sum);
#endif
}

// Softmax, can do multiple batches, necerssary to set local work
// groups, even for a single batch 
__kernel
void softmax(
         int    nunits,
__global float* sums,
__global float* z,
__global float* out){

    int i = get_global_id(0);

    out[i] = exp(z[i]) / sums[i/nunits];

#ifdef DEBUG
    printf("softmax[%d] = %f / %f\n", i, exp(z[i]), sums[i/nunits]);
#endif
}

// Set local work groups for each batch 
__kernel 
void cross_entropy(
__global float* y,
__global float* y_,
__global float* x){

    int i = get_global_id(0);

    x[i] = -y[i] * log(y_[i]);
}

// Softmax der, combining the cross entropy and softmax der makes 
// calculation very efficient, but the diff calculate is dL/dA not the
// dL/dy like normal, so function must be called with value_grad as der
__kernel
void cross_entropy_der(
__global float* y,
__global float* y_,
__global float* der){

    int i = get_global_id(0);

    der[i] = y_[i] - y[i];

#ifdef DEBUG
    printf("cross_entropy_der[%d]: %f = %f - %f\n", i, der[i], y_[i], y[i]);
#endif
}

__kernel 
void GrdDsc(
         float  learn_rate,
__global float* weights,
__global float* grads){

    int i = get_global_id(0);

    weights[i] -= learn_rate * grads[i];

#ifdef DEBUG
    printf("GrdDsc[%d]: -= %f * %f\n", i, learn_rate, grads[i]);
#endif
}

// Adam optimiser, needs a separate clmem for average and squared average
// t is the number instance from the current batch

#define BETA1 0.9f
#define BETA2 0.999f
#define EPSILON 1e-6f

__kernel
void adam(
         float  learn_rate,
__global float* weights,
__global float* grads,
__global float* avgs,
__global float* square_avgs,
          int    t){

    int i = get_global_id(0);

    t++;

    avgs[i]        = BETA1 * avgs[i]        + (1.f - BETA1) * grads[i]; 
    square_avgs[i] = BETA2 * square_avgs[i] + (1.f - BETA2) * grads[i] * grads[i];

    float corrected_avg        = avgs[i]        / (1.f - pown(BETA1, t));
    float corrected_square_avg = square_avgs[i] / (1.f - pown(BETA2, t));

    weights[i] -= learn_rate * corrected_avg / (sqrt(corrected_square_avg) + EPSILON);

#ifdef DEBUG
    printf("ADAM[%d]: -= %f * %f / (sqrt(%f) + %f) : %f, %f\n", 
        i, learn_rate, corrected_avg, corrected_square_avg, EPSILON,
        avgs[i], square_avgs[i]);
#endif
}

__kernel
void l2_reg(
         float  lambda,
__global float* grads,
__global float* weights){

    int i = get_global_id(0);

    grads[i] += lambda * weights[i];
}

// Calculate average for each feature
__kernel
void avg(
         int    bsize,
__global float* values,
__global float* result){

    int g         = get_global_id(0);
    int nfeatures = get_global_size(0);

    float f = 0.0f;
    for (int i = 0; i < bsize; i++)
        f += values[i * nfeatures + g];

    f /= (float)bsize;

    result[g] = f;

#ifdef DEBUG
    printf("avg[%d]: %f\n", g, f);
#endif
}

// Calculate variance: sum (x - avg)^2 / n of each feature
__kernel
void var(
         int    bsize,
__global float* avg,
__global float* values,
__global float* result){

    int g         = get_global_id(0);
    int nfeatures = get_global_size(0);

    float f = 0.0f;
    for (int i = 0; i < bsize; i++)
        f += pow(values[i * nfeatures + g] - avg[g], 2) / (float)bsize;

    result[g] = f;
}

// Apply affine transform: y = gamma x + beta
__kernel 
void affine(
__global float* values,
__global float* gamma,
__global float* beta){

    int i = get_global_id(0);
    int l = get_local_id(0);

    values[i] = gamma[l] * values[i] + beta[l];

#ifdef DEBUG
    printf("affine[%d] = %f * %f + %f\n", i, gamma[l], values[i], beta[l]);
#endif
}

__kernel
void norm1d(
__global float* values,
__global float* avg,
__global float* var){

    int i = get_global_id(0);
    int l = get_local_id(0);

#ifdef DEBUG
    printf("norm1d[%d] = (%f - %f) / %f\n", i, values[i], avg[l], sqrt(var[l] + EPSILON));
#endif

    values[i] = (values[i] - avg[l]) / sqrt(var[l] + EPSILON);
}

// Combine the affine and norm der calcs with act grad to get grad 
// of all transforms after weights x prev values.
__kernel
void norm1d_der(
__global float* pre_norm,
__global float* avg,
__global float* var,
__global float* gamma,
__global float* act_grad){

    int i = get_global_id(0);
    int l = get_local_id(0);

    int len = get_local_size(0);

    float norm_der = (1.0f - 1.0f/(float)len) * (var[l] + EPSILON);
    norm_der      -= (1.0f/(float)len) * pow(pre_norm[i] - avg[l], 2.0f);
    norm_der      /= pow(var[l] + EPSILON, 1.5f);

#ifdef DEBUG
    printf("norm1d[%d]: %f * %f * %f\nx: %f avg: %f var:%f\n", 
        i, act_grad[i], gamma[l], norm_der, pre_norm[i], avg[l], var[l]);
#endif

    act_grad[i] = act_grad[i] * gamma[l] * norm_der;
}

__kernel 
void gamma_grad(
         int    bsize,
__global float* pre_affine_values,
__global float* value_grad,
__global float* gamma_grad){

    int g         = get_global_id(0);
    int nfeatures = get_global_size(0) / bsize;

    float f = 0.0f;
    for (int i = 0; i < bsize; i++){
        //printf("[%d]+= %f * %f\n", g, pre_affine_values[nunits * i + g], value_grad[nunits * i + g]);
        f += pre_affine_values[nfeatures * i + g] 
           * value_grad[nfeatures * i + g];
    }
    f /= (float)bsize;

    gamma_grad[g] = f;

#ifdef DEBUG
    printf("gamma_grad[%d]: %f\n", g, f);
#endif
}

// Filters: features[filterh[filterw[channels]]]
// Values:  batches[prevh[prevw[channels]]]
// result:  batches[filtersy[filtersx[features]]]
__kernel
void convolution(
         int    prevw,
         int    prevh,
         int    filtersy,
         int    filterw,
         int    filterh,
         int    channels,
         int    stridex,
         int    stridey,
__global float* filters,
__global float* values,
__global float* result){

    int filtery = get_global_id(0);
    int filterx = get_global_id(1);
    int feature = get_global_id(2);

    int filtersx   = get_global_size(1);
    int features = get_global_size(2);

    int batch = filtery / filtersy;
    filtery %= filtersy;

    // Index of filter being applied from filter list
    int filters_index = feature * filterh * filterw * channels;

    int image_start;
    // Start at batch's(image's) 0,0
    image_start  = batch * prevh * prevw ;
    // Move to start of filter within batch
    int filter_start;
    filter_start  = image_start;
    filter_start += filtery * stridey * prevw + filterx * stridex;

    int unit_index = (filtery * filtersx + filterx) * features + feature;

    // Index of output node being calculated
    int result_index;
    result_index  = batch * filtersy * filtersx;
    result_index += filtery * filtersx;
    result_index += filterx;
    result_index *= features;
    result_index += feature;

    float acc = 0.0f;
    for (int y = 0; y < filterh; y++){
        int filtery = filter_start + y * prevw;

        for (int x = 0; x < filterw; x++){
            int filterx = filtery + x;
            int filterc = filterx * channels;

            for (int c = 0; c < channels; c++){
                acc += values[filterc + c] * filters[filters_index];
                //printf("conv[%d] += %f * %f(%d * %d)\n", result_index, values[filterc + c], filters[filters_index], filterc + c, filters_index);
                filters_index++;
            }
        }
    }

    result[result_index] = acc;

#ifdef DEBUG
    printf("conv[%d]: %f\n", result_index, acc);
#endif
}

// value_grads batches[height[width[features]]]
__kernel
void pad_and_dilate(
         int    filterw,
         int    filterh,
         int    stridex,
         int    stridey,
         int    output_size,
         int    bsize,
__global float* value_grads,
__global float* output){

    int y = get_global_id(0);
    int x = get_global_id(1);
    int c = get_global_id(2);

    int gradsh = get_global_size(0) / bsize;
    int gradsw = get_global_size(1);
    int gradsc = get_global_size(2);

    // grads(w/h) is number of filters(x/y) 
    int output_cols = filterw + (gradsw - 1) * stridex + filterw - 1;

    int grads_row_size = gradsw * gradsc;

    int grads_index;
    grads_index  = y * grads_row_size;
    grads_index += x * gradsc;
    grads_index += c;
    
    int batch = y / gradsh;
    y %= gradsh;

    int output_index;
    output_index  = batch * output_size;
    output_index += (filterh - 1 + y * stridey) * output_cols * gradsc;
    output_index += (filterw - 1 + x * stridex) * gradsc; 
    output_index += c;

    output[output_index] = value_grads[grads_index];

#ifdef DEBUG
    printf("pad[%d]: %d\n", output_index, grads_index);
#endif
}

// Perform convolution on a padded output gradient image using reversed 
// filters to find the gradients at input  
// Filters: [features[filterh[filterw[channels]]]]
// Grads:   batches[paddedh[paddedw[features]]]
// Result: batches[prevh[prevw[channels]]]
__kernel
void deconvolution(
         int    paddedw,
         int    paddedh,
         int    filterw,
         int    filterh,
         int    features,
__global float* filters,
__global float* value_grads,
__global float* result){

    int prevy   = get_global_id(0);
    int prevx   = get_global_id(1);
    int channel = get_global_id(2);

    int prevh    = get_global_size(0);
    int prevw    = get_global_size(1);
    int channels = get_global_size(2);

    int batch = prevy / prevh;
    prevy %= prevh;

    int fitler_size  = filterh * filterw * channels;

    // First position of filter to use as filter is read backwards
    // for each channel
    int filter_index = fitler_size - channels + channel;

    // Start of batch's grads image 0,0
    int image_start = batch * paddedh * paddedw * features;
    // Start of filter
    int filter_start = image_start + prevy * paddedw + prevx;
    
    float acc = 0.0f;
    for (int y = 0; y < filterh; y++){
        int filtery = filter_start + y * paddedw;

        for (int x = 0; x < filterw; x++){
            int filterx = filtery + x;
            int filterf = filterx * features;

            for (int f = 0; f < features; f++){
                int feature_index = filter_index + f * fitler_size;

                acc += value_grads[filterf + f] * filters[feature_index];
            }

            filter_index -= channels;
        }
    }

    int result_index;
    result_index  = batch * prevh * prevx;
    result_index += prevy * prevw;
    result_index += prevx;
    result_index *= channels;
    result_index += channel;

    result[result_index] = acc;

#ifdef DEBUG
    printf("deconv[%d]: %f\n", result_index, acc);
#endif
}

// Weight gradients are the result of applying convolution to the input 
// vlaues using the output grads dilated by stride - 1 as the filter. 
// Values:  batches[prevh[prevw[channels]]]
// Grads:   batches[paddedh[paddedw[features]]]
// Result:  features[filterh[filterw[channels]]]
__kernel 
void convolution_weight_grads(
         int    prev_size,
         int    prevw,
         int    padded_size,
         int    paddedw,
         int    paddedh,
         int    channels,
         int    bsize,
__global float* values,
__global float* padded_grads,
__global float* weight_grad){
    
    int weighty = get_global_id(0);
    int weightx = get_global_id(1);
    int channel = get_global_id(2);

    int filterh  = get_global_size(0);
    int filterw  = get_global_size(1);
    int features = get_global_size(2) / channels;

    // Max 3 dimensions for work group, include channels and features 
    // as one dimension of features x channels
    int feature = channel / channels;
    channel %= channels;

    int values_start = weighty * prevw + weightx;

    float acc = 0.0f;
    for (int y = 0; y < paddedh; y++){
        int valuesy = values_start + y * prevw;
        int gradsy  = y * paddedw;

        for (int x = 0; x < paddedw; x++){
            int valuesx = valuesy + x;
            int valuesc = valuesx * channels + channel;    
            int gradsx  = gradsy + x;
            int gradsc  = gradsx * features + feature;
            
            for (int b = 0; b < bsize; b++){
                int valuesb = valuesc + b * prev_size;
                int gradsb  = gradsc + b * padded_size;

                acc += values[valuesb] * padded_grads[gradsb];
            }
        }
    }

    acc /= (float)bsize;

    int weight_index;
    weight_index  = feature * filterh * filterw;
    weight_index += weighty * filterw;
    weight_index += weightx;
    weight_index *= channels;
    weight_index += channel;

    weight_grad[weight_index] = acc;

#ifdef DEBUG 
    printf("conv_grad[%d]: %f\n", weight_index, acc);
#endif
}