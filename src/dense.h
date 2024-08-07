#include "m_pd.h"
#include <stdlib.h>

#define RAND_HIGH_RANGE (0.10)
#define RAND_MIN_RANGE (-0.10)
#define INIT_BIASES (0.0)
#define NUM_ACT_FUNCS 8
#define SOFTMAX_INDEX NUM_ACT_FUNCS - 1
#define NUM_LOSS_FUNCS 4
#define NUM_OPTIMIZER_FUNCS 4
#define CAT_X_ENT_NDX 2
#define BIN_X_ENT_NDX 3

typedef struct _layer_dense
{
	/* main vectors */
	t_float **x_weights;
	t_float *x_biases;
	/* vectors that keep best weights and biases during training */
	t_float **x_weights_copy;
	t_float *x_biases_copy;
	/* vectors for morphing */
	t_float **x_set_weights;
	t_float *x_set_biases;
	t_float **x_target_weights;
	t_float *x_target_biases;
	/* output vectors for training, testing, and predicting */
	t_float **x_output;
	t_float **x_output_train;
	t_float **x_output_transposed;
	t_float **x_act_output;
	t_float **x_act_output_train;
	t_float **x_act_output_transposed;
	/* back propagation vectors */
	t_float **x_dinput;
	t_float **x_dweights;
	t_float *x_dbiases;
	t_float **x_act_dinput;
	/* optimization vectors */
	t_float **x_weight_momentums;
	t_float **x_weight_cache;
	t_float **x_weight_momentums_corrected;
	t_float **x_weight_cache_corrected;
	t_float **x_weights_transposed;
	t_float *x_bias_momentums;
	t_float *x_bias_cache;
	t_float *x_bias_momentums_corrected;
	t_float *x_bias_cache_corrected;
	/* softmax activation function vectors */
	t_float **x_eye;
	t_float **x_dot_product;
	t_float **x_jac_mtx; /* jacobian matrix */
	/* dropout vectors */
	t_float **x_binary_mask;
	t_float **x_dropout_output;
	/* dropout variables */
	t_float x_dropout_rate;
	int x_dropout_allocated;
	/* main structure variables */
	int x_input_size;
	int x_output_size;
	/* boolean for allocating softmax vectors */
	int x_allocate_softmax;
	/* booleans to store weights and biases to arrays during training */
	int *x_store_weights_during_training;
	int x_store_biases_during_training;
	/* symbols for the names of these arrays */
	t_symbol **x_weights_arrays;
	t_symbol *x_biases_array;
	/* and scale coefficients, as these values are likely to be very small */
	t_float x_weights_scale;
	t_float x_biases_scale;
} t_layer_dense;

void dropout_forward(t_layer_dense *layer, t_float **prev_layer_output, int rows);
void layer_dense_forward(t_float **prev_layer_output, t_layer_dense *this_layer, int input_size,
		int is_training, int is_testing);
void layer_dense_backward(t_layer_dense *layer_dense, t_float **previous_layer_output,
	float weight_regularizer_l1, float weight_regularizer_l2,
	float bias_regularizer_l1, float bias_regularizer_l2,int dot_loop_size);
