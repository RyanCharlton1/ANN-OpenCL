// Find row vector dot product each pass. Vec can be larger than matrix
// for batches, needs to be set in local workgroup size.
__kernel 
void mat_vec_mult(int cols,
         __global float* mat,
         __global float* vec_in,
         __global float* vec_out){
    
    // Row of the output
    int g    = get_global_id(0);
    // Row of the mat
    int l    = get_local_id(0);
    int rows = get_local_size(0);

    // where to start reading input
    int start = cols * (g / rows);

    float f = 0.0f;
    for (int i = 0; i < cols; i++){
        f += mat[l * cols + i] * vec_in[start + i];
    }

    vec_out[g] = f;
}

// Add two vectors and store in the first, second may be larger than first,
// it will repeat if second length is set as local work size.
__kernel
void vec_vec_add_inplace(__global float* values,
                         __global float* bias){

    int g = get_global_id(0);
    // Length of bias vec
    int s = get_local_size(0);

    values[g] += bias[g % s];
}

// ReLU activation 
__kernel
void ReLU(__global float* pre_act_values,
          __global float* values){

    int g = get_global_id(0);

    values[g] = pre_act_values[g] > 0.0f ? pre_act_values[g] : 0.0f;
}

// Leaky ReLU activation
__kernel
void leaky_ReLU(__global float* pre_act_values,
                __global float* values){

    int g = get_global_id(0);

    values[g] = pre_act_values[g] * (pre_act_values[g] > 0.0f ? 1.0f : 0.1f);
}