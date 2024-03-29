//#define DEBUG

// [0, .. cols] [v1]
// [..        ] [v2]
// [rows      ] [v3] 
// Find row vector dot product each pass. Vec can be larger than matrix
// for batches, needs to be set in local workgroup size.
__kernel 
void mat_vec_mult(int    cols,
         __global float* mat,
         __global float* vec_in,
         __global float* vec_out){
    
    int g    = get_global_id(0);
    int l    = get_local_id(0);
    int rows = get_local_size(0);

    // where to start reading input
    int start = cols * (g / rows);

    float f = 0.0f;
    for (int i = 0; i < cols; i++){
        f += mat[l * cols + i] * vec_in[start + i];
    }

    vec_out[g] = f;

#ifdef DEBUG
    printf("value[%d]:=%f\n", g, f);
#endif
}

// Multiply vec_a and vec_b, store in vec_c
__kernel 
void vec_vec_mult(__global float* vec_a,
                  __global float* vec_b,
                  __global float* vec_c){

    int i = get_global_id(0);

    vec_c[i] = vec_a[i] * vec_b[i];

#ifdef DEBUG
    printf("vec_vec_mult %d:%f*%f\n", i, vec_a[i], vec_b[i]);
#endif
}

// Add two vectors and store in the first, second may be larger than first,
// it will repeat if second length is set as local work size.
__kernel
void vec_vec_add_inplace(__global float* values,
                         __global float* bias){

    int g = get_global_id(0);
    // Length of bias vec
    int s = get_local_size(0);

    //printf("g=%d: %f += %f\n", g, values[g], bias[g%s]);
    values[g] += bias[g % s];
}

__kernel
void vec_scalar_div(__global float* vec,
                             float  n){

    int i = get_global_id(0);

    vec[i] /= n;
}

// Similar to a cartesian product but multiply the elements
// dL/dy * dy/dw(values) takes two vectors and returns dL/dw, a matrix
// To avoid conccurrency problems for batches, each job calculates an
// element of the weight grad matrix 
__kernel 
void weight_grad(int    bsize,
        __global float* loss,
        __global float* prev_values,
        __global float* weight_grad){

    int s        = get_global_size(0);
    int col_size = get_local_size(0);   // prev_nunits, size of prev_values
    int row_size = s / col_size;        // nunits, size of loss

    int g   = get_global_id(0);
    int col = get_local_id(0);

    float f = 0.0f;
    for (int b = 0; b < bsize; b++)
        f += loss[b * row_size + g / col_size] 
           * prev_values[b * col_size + col];
    
    f /= (float)bsize;

    weight_grad[g] = f;

#ifdef DEBUG
    printf("weight_grad[%d]=%f\n", g, f);
#endif
}

// Calculate each element of bias in one job, same as weight_grad
__kernel 
void bias_grad(int    bsize,
               int    bias_size,
      __global float* values_grad,
      __global float* bias_grad){

    int i = get_global_id(0);

    float f = 0.0f;
    for (int b = 0; b < bsize; b++)
        f += values_grad[bias_size * b + i];

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
void mat_vec_mult_trans(int    rows,
               __global float* mat,
               __global float* vec_in,
               __global float* vec_out){
    
    int g    = get_global_id(0);
    int l    = get_local_id(0);
    int cols = get_local_size(0);

    // where to start reading input
    int start = rows * (g / cols);

    float f = 0.0f;
    for (int i = 0; i < rows; i++){
        f += mat[i * cols + l] * vec_in[start + i];
    }
    
    vec_out[g] = f;

#ifdef DEBUG
    printf("loss_grad[%d]:=%f\n", g, f);
#endif
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

// ReLU activation der
__kernel
void ReLU_der(__global float* pre_act_values,
              __global float* values_grad){

    int g = get_global_id(0);

    values_grad[g] = pre_act_values[g] > 0.0f ? 1.0 : 0.0f;
}

// Leaky ReLU activation der
__kernel
void leaky_ReLU_der(__global float* pre_act_values,
                    __global float* values_grad){

    int g = get_global_id(0);

    values_grad[g] = pre_act_values[g] > 0.0f ? 1.0f : 0.1f;
}

// MSE loss, can do multiple batches
__kernel 
void MSE(__global float* y,
         __global float* y_,
         __global float* x){

    int i = get_global_id(0);
    int n = get_global_size(0);
    
    float err = y[i] - y_[i];
    //printf("%d:err=(%f - %f): %f / %d / 2.0f\n", i, y[i], y_[i], err*err, n);
    x[i] = err * err / (float)n / 2.0f;
}

// MSE loss der, can do multiple batches, necerssary to set local work
// groups, even for a single batch
__kernel 
void MSE_der(__global float* y,
             __global float* y_,
             __global float* der){

    int i = get_global_id(0);
    int n = get_local_size(0);

    float err = y[i] - y_[i];
    
    der[i] = -err / (float)n;

#ifdef DEBUG
    printf("MSE_der:%f = -%f / %f\n", der[i], err, (float)n);
#endif
}

// Softmax bottom sum, giving each batch one job is more efficient than
// making a mutex for the sum
__kernel
void softmax_sum(int    zsize,
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
void softmax(__global float* sums,
             __global float* z,
             __global float* out){

    int i = get_global_id(0);
    int s = get_local_size(0);

    out[i] = exp(z[i]) / sums[i/s];

#ifdef DEBUG
    printf("softmax[%d] = %f / %f\n", i, exp(z[i]), sums[i/s]);
#endif
}

// Set local work groups for each batch 
__kernel 
void cross_entropy(__global float* y,
                   __global float* y_,
                   __global float* x){

    int   i = get_global_id(0);
    float s = (float)get_local_size(0);

    x[i] = -y[i] * log(y_[i]) / s;
}

// Softmax der, combining the cross entropy and softmax der makes 
// calculation very efficient, but the diff calculate is dL/dA not the
// dL/dy like normal, so function must be called with value_grad as der
__kernel
void cross_entropy_der(__global float* y,
                       __global float* y_,
                       __global float* der){

    int i = get_global_id(0);

    der[i] = y_[i] - y[i];
#ifdef DEBUG
    printf("cross_entropy_der[%d]: %f = %f - %f\n", i, der[i], y_[i], y[i]);
#endif
}

__kernel 
void GrdDsc(float  learn_rate,
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

#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-6

__kernel
void adam(float  learn_rate,
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