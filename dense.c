#include "dense.h"

void dropout_forward(t_layer_dense *layer, t_float **prev_layer_output, int rows)
{
	int i, j, k, rand_ndx;
	/* keep track of the random indexes so they don't repeat */
	int stored_random_ndx[layer->x_input_size];
	for (i = 0; i < rows; i++) {
		for (j = 0; j < layer->x_input_size; j++) {
			/* populate the array with -1s which will not be generated randomly
			   and set all the binary mask values to 1 / dropout rate */
			stored_random_ndx[j] = -1;
			/* the division by the dropout rate compensates for the
			   value difference that occurs due to less layers being active
			   in a forward pass */
			layer->x_binary_mask[i][j] = 1.0 / (1.0 - layer->x_dropout_rate);
		}
		for (j = 0; j < layer->x_input_size; j++) {
			int ndx = 0;
			int dropped_out = 0;
			while (((t_float)dropped_out / (t_float)layer->x_input_size) \
					< layer->x_dropout_rate) {
				int already_stored = 0;
				/* generate a random number in the range of the columns */
				rand_ndx = rand() % (layer->x_input_size + 1);
				for (k = 0; k < layer->x_input_size; k++) {
					/* check if it has already been stored */
					if (rand_ndx == stored_random_ndx[k]) {
						already_stored = 1;
						break;
					}
				}
				if (!already_stored) {
					layer->x_binary_mask[i][rand_ndx] = 0.0;
					stored_random_ndx[ndx++] = rand_ndx;
					dropped_out++;
				}
			}
		}
		/* finally mask the previout layer activation output */
		for (j = 0; j < layer->x_input_size; j++) {
			layer->x_dropout_output[i][j] = prev_layer_output[i][j] * layer->x_binary_mask[i][j];
		}
	}
}

void layer_dense_forward(t_float **prev_layer_output, t_layer_dense *this_layer, int input_size,
		int is_training, int is_testing)
{
	int i, j, k;
	t_float **input = prev_layer_output;
	t_float **output;
	if (is_training || is_testing) {
		output = this_layer->x_output_train;
		if (this_layer->x_dropout_allocated) {
			dropout_forward(this_layer, prev_layer_output, input_size);
			input = this_layer->x_dropout_output;
		}
	}
	else {
		output = this_layer->x_output;
	}
	/* dot product of previous layer output and this layer's weights */
	for (i = 0; i < input_size; i++) {
		for (j = 0; j < this_layer->x_output_size; j++){
			output[i][j] = 0.0;
			for (k = 0; k < this_layer->x_input_size; k++) {
				output[i][j] += (input[i][k] * this_layer->x_weights[k][j]);
			}
			/* add bias to dot product */
			output[i][j] += this_layer->x_biases[j];
		}
	}
}

void layer_dense_backward(t_layer_dense *layer_dense, t_float **previous_layer_output,
	float weight_regularizer_l1, float weight_regularizer_l2,
	float bias_regularizer_l1, float bias_regularizer_l2,int dot_loop_size)
{
	int i, j, k;
	/* dot product between previous layer output (this layer's
	   activation function's output) and this layer's input derivatives */
	for (i = 0; i < layer_dense->x_input_size; i++) {
		for (j = 0; j < layer_dense->x_output_size; j++) {
			layer_dense->x_dweights[i][j] = 0.0;
			for (k = 0; k < dot_loop_size; k++) {
				layer_dense->x_dweights[i][j] += (previous_layer_output[i][k] * \
						layer_dense->x_act_dinput[k][j]);
			}
		}
	}
	/* set the sum of layer_dense->x_act_dinput to layer_dense->x_dbiases */
	for (i = 0; i < layer_dense->x_output_size; i++) {
		layer_dense->x_dbiases[i] = 0.0;
		for (j = 0; j < dot_loop_size; j++) {
			layer_dense->x_dbiases[i] += layer_dense->x_act_dinput[j][i];
		}
	}
	/* check for regularizers */
	if (weight_regularizer_l1 > 0.0) {
		for (i = 0; i < layer_dense->x_input_size; i++) {
			for (j = 0; j < layer_dense->x_output_size; j++) {
				t_float dl1 = 1.0;
				if (layer_dense->x_weights[i][j] < 0.0) dl1 = -1.0;
				layer_dense->x_dweights[i][j] += (layer_dense->x_weights[i][j] * dl1);
			}
		}
	}
	if (weight_regularizer_l2 > 0.0) {
		for (i = 0; i < layer_dense->x_input_size; i++) {
			for (j = 0; j < layer_dense->x_output_size; j++) {
				layer_dense->x_dweights[i][j] += (2.0 * weight_regularizer_l2 * \
						layer_dense->x_weights[i][j]);
			}
		}
	}
	if (bias_regularizer_l1 > 0.0) {
		for (i = 0; i < layer_dense->x_output_size; i++) {
			t_float dl1 = 1.0;
			if (layer_dense->x_biases[i] < 0.0) dl1 = -1.0;
			layer_dense->x_dbiases[i] += (bias_regularizer_l1 * dl1);
		}
	}
	if (bias_regularizer_l2 > 0.0) {
		for (i = 0; i < layer_dense->x_output_size; i++) {
			layer_dense->x_dbiases[i] += (2.0 * bias_regularizer_l2 * \
					layer_dense->x_biases[i]);
		}
	}
	/* dot product between this layer's activation's input derivative
	   and this layer's transposed weights, results to this layer's
	   input derivatives */
	for (i = 0; i < dot_loop_size; i++) {
		for (j = 0; j < layer_dense->x_input_size; j++) {
			layer_dense->x_dinput[i][j] = 0.0;
			for (k = 0; k < layer_dense->x_output_size; k++) {
				layer_dense->x_dinput[i][j] += (layer_dense->x_act_dinput[i][k] * \
						layer_dense->x_weights_transposed[k][j]);
				if (layer_dense->x_dropout_allocated) {
					layer_dense->x_dinput[i][j] *= layer_dense->x_binary_mask[i][j];
				}
			}
		}
	}
}

