/*********************************************************************************
 * (c) 2022 Alexandros Drymonitis                                                 *
 * Code translated to C from Python from the book                                 *
 * "Neural Networks from Scratch in Python" by Harrison Kinsley & Daniel Kukieła  *
 * and based on code (c) 1997-1999 by Miller Puckette                             *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted provided that the following conditions are met:    *
 *                                                                                *
 * 1. Redistributions of source code must retain the above copyright notice, this *
 *    list of conditions and the following disclaimer.                            *
 *                                                                                *
 * 2. Redistributions in binary form must reproduce the above copyright notice,   *
 *    this list of conditions and the following disclaimer in the documentation   *
 *    and/or other materials provided with the distribution.                      *
 *                                                                                *
 * 3. Neither the name of the copyright holder nor the names of its               *
 *    contributors may be used to endorse or promote products derived from        *
 *    this software without specific prior written permission.                    *
 *                                                                                *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"    *
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE      *
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE *
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE   *
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL     *
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR     *
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER     *
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,  *
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE  *
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.           *
 **********************************************************************************/

#include "m_pd.h"
#include "float.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "limits.h"
#include "neuralnet.h"

static t_class *neuralnet_class;

/************************ layer functions ***************************/

static void set_seed(t_neuralnet *x, t_float f)
{
	/* srand() is initially called with the time(0) argument
	   but some systems may not have a reliable time (e.g. a Rasbperry Pi)
	   so we should be able to set the seed of rand() based on some other
	   random factor, externally */
	if (f < 0.0) {
		pd_error(x, "seed must be positive");
		return;
	}
	srand((unsigned int)f);
}

/* return a uniformly distributed random value */
static t_float rand_gen()
{
	return (t_float)(rand() + 1.0)/((t_float)(RAND_MAX) + 1.0);
}

/* return a normally distributed random value */
static t_float normal_random()
{
	t_float v1=rand_gen();
	t_float v2=rand_gen();
	return cos(2*3.14*v2)*sqrt(-2.0*log(v1));
}

static void layer_dense_init(t_neuralnet *x, t_layer_dense *layer, int input_size, int output_size)
{
	/* initialize one layer */
	int i;

	layer->x_input_size = input_size;
	layer->x_output_size = output_size;

	layer->x_weights = (t_float **)malloc(sizeof(t_float*) * input_size);
	for (i = 0; i < input_size; i++) {
		layer->x_weights[i] = (t_float *)malloc(sizeof(t_float) * output_size);
	}
	if(layer->x_weights == NULL){
		pd_error(x, "weights mem error");
		return;
	}

	layer->x_weights_arrays = (t_symbol **)malloc(sizeof(t_symbol*) * input_size);
	if(layer->x_weights_arrays == NULL){
		for (i = 0; i < input_size; i++) {
			free(layer->x_weights[i]);
		}
		free(layer->x_weights);
		pd_error(x, "weights_arrays mem error");
		return;
	}

	layer->x_store_weights_during_training = (int *)malloc(sizeof(int) * input_size);
	if(layer->x_store_weights_during_training == NULL){
		for (i = 0; i < input_size; i++) {
			free(layer->x_weights[i]);
		}
		free(layer->x_weights);
		free(layer->x_weights_arrays);
		pd_error(x, "store_weights_during_training mem error");
		return;
	}

	layer->x_biases = (t_float *)malloc(sizeof(t_float) * output_size);
	if(layer->x_biases == NULL){
		for (i = 0; i < input_size; i++) {
			free(layer->x_weights[i]);
		}
		free(layer->x_weights);
		free(layer->x_weights_arrays);
		free(layer->x_store_weights_during_training);
		pd_error(x, "biases mem error");
		return;
	}
	/* we're defining the x_output array as a 2D array with dimensions
	   1 x output_size to make it easier to work with the rest
	   of the 2D arrays of weights */
	layer->x_output = (t_float **)malloc(sizeof(t_float*) * 1);
	layer->x_output[0] = (t_float *)malloc(sizeof(t_float) * output_size);
	if(layer->x_output == NULL){
		for (i = 0; i < input_size; i++) {
			free(layer->x_weights[i]);
		}
		free(layer->x_weights);
		free(layer->x_weights_arrays);
		free(layer->x_store_weights_during_training);
		free(layer->x_biases);
		pd_error(x, "layer output mem error");
		return;
	}

	layer->x_act_output = (t_float **)malloc(sizeof(t_float*) * 1);
	layer->x_act_output[0] = (t_float *)malloc(sizeof(t_float) * output_size);
	if(layer->x_act_output == NULL){
		for (i = 0; i < input_size; i++) {
			free(layer->x_weights[i]);
		}
		free(layer->x_weights);
		free(layer->x_weights_arrays);
		free(layer->x_store_weights_during_training);
		free(layer->x_biases);
		free(layer->x_output[0]);
		free(layer->x_output);
		pd_error(x, "layer act output mem error");
		return;
	}

	init_layer_biases(layer);
	init_layer_weights(x, layer);
	layer->x_allocate_softmax = 0;
	/* disable dropout by default */
	layer->x_dropout_allocated = 0;
	layer->x_dropout_rate = 0.0;
	layer->x_store_biases_during_training = 0;
	for (i = 0; i < input_size; i++) {
		layer->x_store_weights_during_training[i] = 0;
	}
	layer->x_weights_scale = 1.0;
	layer->x_biases_scale = 1.0;
}

static void layer_init_common_vars(t_neuralnet *x, int layer_nr)
{
	x->x_layers[layer_nr].x_weight_regularizer_l1 = 0.0;
	x->x_layers[layer_nr].x_weight_regularizer_l2 = 0.0;
	x->x_layers[layer_nr].x_bias_regularizer_l1 = 0.0;
	x->x_layers[layer_nr].x_bias_regularizer_l2 = 0.0;
	/* set the sigmoid function by default
	   which can be changed via messages */
	x->x_layers[layer_nr].x_activation_index = 1;
}

static void init_biases(t_neuralnet *x)
{
	/* set the inital biases of all layers (called externally) */
	int i;
	if (x->x_layers_initialized) {
		for (i = 0; i < x->x_num_layers; i++) {
			if (x->x_layers[i].type == 0) init_layer_biases(&x->x_layers[i].dense);
		}
	}
}

static void set_weight_regularizer1(t_neuralnet *x, t_float layer, t_float reg)
{
	x->x_layers[(int)layer].x_weight_regularizer_l1 = reg;
}

static void set_weight_regularizer2(t_neuralnet *x, t_float layer, t_float reg)
{
	x->x_layers[(int)layer].x_weight_regularizer_l2 = reg;
}

static void set_bias_regularizer1(t_neuralnet *x, t_float layer, t_float reg)
{
	x->x_layers[(int)layer].x_bias_regularizer_l1 = reg;
}

static void set_bias_regularizer2(t_neuralnet *x, t_float layer, t_float reg)
{
	x->x_layers[(int)layer].x_bias_regularizer_l2 = reg;
}

static void set_dropout(t_neuralnet *x, t_float which_layer, t_float rate)
{
	/* set the percentage of dropout for one layer */
	int this_layer = (int)which_layer;
	if (this_layer >= x->x_num_layers) {
		pd_error(x, "max layer %d", x->x_num_layers-1);
		return;
	}
	if (x->x_layers[this_layer].type != 0) {
		pd_error(x, "dropout can only be set for dense layers");
		return;
	}
	x->x_layers[this_layer].dense.x_dropout_rate = rate;
}

static void set_weight_coeff(t_neuralnet *x, t_float f)
{
	/* set the initial weights of all layers (called externally) */
	int i;
	x->x_weight_coeff = f;
	if (x->x_layers_initialized) {
		for (i = 0; i < x->x_num_layers; i++) {
			if (x->x_layers[i].type == 0) init_layer_weights(x, &x->x_layers[i].dense);
		}
	}
}

static void init_layer_weights(t_neuralnet *x, t_layer_dense *layer)
{
	/* set the initial weights of one layer (called internally) */
	int i, j;
	for(i = 0; i < layer->x_input_size; i++){
		for (j = 0; j < layer->x_output_size; j++) {
			layer->x_weights[i][j] = normal_random() * x->x_weight_coeff;
		}
	}
}

static void init_layer_biases(t_layer_dense *layer)
{
	/* set the initial biases of one layer (called internally) */
	int i;
	for (i = 0; i < layer->x_output_size; i++) {
		/* start with all biases set to 0 */
		layer->x_biases[i] = INIT_BIASES;
	}
}

static void populate_eye(t_layer_dense *layer)
{
	/* populate the eye matrix (diagonal 1s) for a layer with
	   a softmax activation function */
	int i, j;
	for (i = 0; i < layer->x_output_size; i++) {
		for (j = 0; j < layer->x_output_size; j++) {
			if (i == j) layer->x_eye[i][j] = 1.0;
			else layer->x_eye[i][j] = 0.0;
		}
	}
}

/************************ activation functions ***********************/

static void activation_forward(t_neuralnet *x, int input_size, int index)
{
	int act_index = x->x_layers[index].x_activation_index;
	int i, out = 0;
	t_float out_size;
	int array_size;
	t_garray *a;
	t_word *vec;
	if (index == x->x_num_layers-1 && x->x_is_predicting) {
		out = 1;
	}
	/* call the activation function of the index in
	   the function pointer x->x_act_funcs
	   These functions return the size of the list to be output
	   this is useful in case of classification and we want a list
	   of confidences instead of a predicted class */
	out_size = x->x_act_funcs[act_index](x, input_size, index, out);
	if (out && x->x_is_confident) {
		if (x->x_pred_to_added) {
			if (!(a = (t_garray *)pd_findbyclass(x->x_predict_to, garray_class)))
				pd_error(x, "%s: no such array", x->x_predict_to->s_name);
			else if (!garray_getfloatwords(a, &array_size, &vec))
				pd_error(x, "%s: bad template for dense", x->x_predict_to->s_name);
			else {
				for (i = 0; i < array_size; i++) {
					vec[i].w_float = x->x_outvec[i].a_w.w_float;
				}
				/* update the graphic array */
				garray_redraw(a);
			}
		}
		else {
			outlet_list(x->x_out, &s_list, out_size, x->x_outvec);
		}
	}
	x->x_is_confident = 1;
}

/* common function for all activation functions forward pass */
static void set_vec_val(t_neuralnet *x, t_atom *vec, t_float val, int ndx)
{
	/* x->x_output_type tells us if we're outputting regression values
	   class predictions, or binary logistic regression values */
	switch (x->x_output_type) {
		case 0:
			/* regression */
			SETFLOAT(vec, val);
			vec->a_w.w_float *= x->x_max_out_vals[ndx];
			break;
		case 1:
			/* classification */
			/* redundant case, but unifies the interface of
			   activation functions forward passes */
			SETFLOAT(vec, val);
			break;
		case 2:
			/* binary logisic regression */
			SETFLOAT(vec, (val > 0.5 ? 1.0 : 0.0));
			break;
		default:
			break;
	}
}

/* forward passes */

static t_float sigmoid_forward(t_neuralnet *x, int input_size, int index, int out)
{
	int i, j;
	t_atom *outvec = x->x_outvec;
	t_float **layer_output = x->x_layers[index].dense.x_output;
	t_float **act_output = x->x_layers[index].dense.x_act_output;
	int output_size = x->x_layers[index].dense.x_output_size;
	switch (x->x_layers[index].type) {
		case 0:
			layer_output = x->x_layers[index].dense.x_output;
			act_output = x->x_layers[index].dense.x_act_output;
			output_size = x->x_layers[index].dense.x_output_size;
			break;
		default:
			break;
	}
	if ((x->x_is_training || x->x_is_validating) && x->x_layers[index].type == 0) {
		layer_output = x->x_layers[index].dense.x_output_train;
		act_output = x->x_layers[index].dense.x_act_output_train;
	}
	for (i = 0; i < input_size; i++) {
		for (j = 0; j < output_size; j++) {
			act_output[i][j] = 1.0 / (1.0 + exp(-layer_output[i][j]));
			if (out) {
				set_vec_val(x, outvec, act_output[i][j], j);
				outvec++;
			}
			if (x->x_is_training && x->x_layers[index].type == 0) {
				x->x_layers[index].dense.x_act_output_transposed[j][i] = act_output[i][j];
			}
		}
	}
	return x->x_output_size;
}

static t_float bipolar_sigmoid_forward(t_neuralnet *x, int input_size, int index, int out)
{
	int i, j;
	t_atom *outvec = x->x_outvec;
	t_float **layer_output = x->x_layers[index].dense.x_output;
	t_float **act_output = x->x_layers[index].dense.x_act_output;
	int output_size = x->x_layers[index].dense.x_output_size;
	switch (x->x_layers[index].type) {
		case 0:
			layer_output = x->x_layers[index].dense.x_output;
			act_output = x->x_layers[index].dense.x_act_output;
			output_size = x->x_layers[index].dense.x_output_size;
			break;
		default:
			break;
	}
	if ((x->x_is_training || x->x_is_validating) && x->x_layers[index].type == 0) {
		if (x->x_layers[index].type == 0) {
			layer_output = x->x_layers[index].dense.x_output_train;
			act_output = x->x_layers[index].dense.x_act_output_train;
		}
	}
	for (i = 0; i < input_size; i++) {
		for (j = 0; j < output_size; j++) {
			act_output[i][j] =  (1 - exp(-layer_output[i][j])) / (1 + exp(-layer_output[i][j]));
			if (out) {
				set_vec_val(x, outvec, act_output[i][j], j);
				outvec++;
			}
			if (x->x_is_training && x->x_layers[index].type == 0) {
				x->x_layers[index].dense.x_act_output_transposed[j][i] = act_output[i][j];
			}
		}
	}
	return x->x_output_size;
}

static t_float relu_forward(t_neuralnet *x, int input_size, int index, int out)
{
	int i, j;
	t_atom *outvec = x->x_outvec;
	t_float **layer_output = x->x_layers[index].dense.x_output;
	t_float **act_output = x->x_layers[index].dense.x_act_output;
	int output_size = x->x_layers[index].dense.x_output_size;
	switch (x->x_layers[index].type) {
		case 0:
			layer_output = x->x_layers[index].dense.x_output;
			act_output = x->x_layers[index].dense.x_act_output;
			output_size = x->x_layers[index].dense.x_output_size;
			break;
		default:
			break;
	}
	if ((x->x_is_training || x->x_is_validating) && x->x_layers[index].type == 0) {
		layer_output = x->x_layers[index].dense.x_output_train;
		act_output = x->x_layers[index].dense.x_act_output_train;
	}
	for (i = 0; i < input_size; i++) {
		for (j = 0; j < output_size; j++) {
			if (layer_output[i][j] < 0.0) act_output[i][j] = 0.0;
			else act_output[i][j] = layer_output[i][j];
			if (out) {
				set_vec_val(x, outvec, act_output[i][j], j);
				outvec++;
			}
			if (x->x_is_training && x->x_layers[index].type == 0) {
				x->x_layers[index].dense.x_act_output_transposed[j][i] = act_output[i][j];
			}
		}
	}
	return x->x_output_size;
}

static t_float leaky_relu_forward(t_neuralnet *x, int input_size, int index, int out)
{
	int i, j;
	t_atom *outvec = x->x_outvec;
	t_float **layer_output = x->x_layers[index].dense.x_output;
	t_float **act_output = x->x_layers[index].dense.x_act_output;
	int output_size = x->x_layers[index].dense.x_output_size;
	switch (x->x_layers[index].type) {
		case 0:
			layer_output = x->x_layers[index].dense.x_output;
			act_output = x->x_layers[index].dense.x_act_output;
			output_size = x->x_layers[index].dense.x_output_size;
			break;
		default:
			break;
	}
	if ((x->x_is_training || x->x_is_validating) && x->x_layers[index].type == 0) {
		
		layer_output = x->x_layers[index].dense.x_output_train;
		act_output = x->x_layers[index].dense.x_act_output_train;
	}
	for (i = 0; i < input_size; i++) {
		for (j = 0; j < output_size; j++) {
			if (layer_output[i][j] < 0.0) act_output[i][j] = layer_output[i][j] * x->x_leaky_relu_coeff;
			else act_output[i][j] = layer_output[i][j];
			if (out) {
				set_vec_val(x, outvec, act_output[i][j], j);
				outvec++;
			}
			if (x->x_is_training && x->x_layers[index].type == 0) {
				x->x_layers[index].dense.x_act_output_transposed[j][i] = act_output[i][j];
			}
		}
	}
	return x->x_output_size;
}

static t_float rect_softplus_forward(t_neuralnet *x, int input_size, int index, int out)
{
	int i, j;
	t_atom *outvec = x->x_outvec;
	t_float **layer_output = x->x_layers[index].dense.x_output;
	t_float **act_output = x->x_layers[index].dense.x_act_output;
	int output_size = x->x_layers[index].dense.x_output_size;
	switch (x->x_layers[index].type) {
		case 0:
			layer_output = x->x_layers[index].dense.x_output;
			act_output = x->x_layers[index].dense.x_act_output;
			output_size = x->x_layers[index].dense.x_output_size;
			break;
		default:
			break;
	}
	if ((x->x_is_training || x->x_is_validating) && x->x_layers[index].type == 0) {
		layer_output = x->x_layers[index].dense.x_output_train;
		act_output = x->x_layers[index].dense.x_act_output_train;
	}
	for (i = 0; i < input_size; i++) {
		for (j = 0; j < output_size; j++) {
			act_output[i][j] = log(1 + exp(layer_output[i][j]));
			if (out) {
				set_vec_val(x, outvec, act_output[i][j], j);
				outvec++;
			}
			if (x->x_is_training && x->x_layers[index].type == 0) {
				x->x_layers[index].dense.x_act_output_transposed[j][i] = act_output[i][j];
			}
		}
	}
	return x->x_output_size;
}

static t_float tanh_forward(t_neuralnet *x, int input_size, int index, int out)
{
	int i, j;
	t_atom *outvec = x->x_outvec;
	t_float **layer_output = x->x_layers[index].dense.x_output;
	t_float **act_output = x->x_layers[index].dense.x_act_output;
	int output_size = x->x_layers[index].dense.x_output_size;
	switch (x->x_layers[index].type) {
		case 0:
			layer_output = x->x_layers[index].dense.x_output;
			act_output = x->x_layers[index].dense.x_act_output;
			output_size = x->x_layers[index].dense.x_output_size;
			break;
		default:
			break;
	}
	if ((x->x_is_training || x->x_is_validating) && x->x_layers[index].type == 0) {
		layer_output = x->x_layers[index].dense.x_output_train;
		act_output = x->x_layers[index].dense.x_act_output_train;
	}
	for (i = 0; i < input_size; i++) {
		for (j = 0; j < output_size; j++) {
			act_output[i][j] = (exp(layer_output[i][j]) - exp(-layer_output[i][j])) / \
							   (exp(layer_output[i][j]) + exp(-layer_output[i][j]));
			if (out) {
				set_vec_val(x, outvec, act_output[i][j], j);
				outvec++;
			}
			if (x->x_is_training && x->x_layers[index].type == 0) {
				x->x_layers[index].dense.x_act_output_transposed[j][i] = act_output[i][j];
			}
		}
	}
	return x->x_output_size;
}

static t_float linear_forward(t_neuralnet *x, int input_size, int index, int out)
{
	int i, j;
	t_atom *outvec = x->x_outvec;
	t_float **layer_output = x->x_layers[index].dense.x_output;
	t_float **act_output = x->x_layers[index].dense.x_act_output;
	int output_size = x->x_layers[index].dense.x_output_size;
	switch (x->x_layers[index].type) {
		case 0:
			layer_output = x->x_layers[index].dense.x_output;
			act_output = x->x_layers[index].dense.x_act_output;
			output_size = x->x_layers[index].dense.x_output_size;
			break;
		default:
			break;
	}
	if ((x->x_is_training || x->x_is_validating) && x->x_layers[index].type == 0) {
		layer_output = x->x_layers[index].dense.x_output_train;
		act_output = x->x_layers[index].dense.x_act_output_train;
	}
	for (i = 0; i < input_size; i++) {
		for (j = 0; j < output_size; j++) {
			act_output[i][j] = layer_output[i][j];
			if (out) {
				set_vec_val(x, outvec, act_output[i][j], j);
				outvec++;
			}
			if (x->x_is_training && x->x_layers[index].type == 0) {
				x->x_layers[index].dense.x_act_output_transposed[j][i] = act_output[i][j];
			}
		}
	}
	return x->x_output_size;
}

static t_float softmax_forward(t_neuralnet *x, int input_size, int index, int out)
{
	int i, j;
	t_atom *outvec = x->x_outvec;
	t_float **layer_output = x->x_layers[index].dense.x_output;
	t_float **act_output = x->x_layers[index].dense.x_act_output;
	t_float out_size = x->x_output_size;
	int output_size = x->x_layers[index].dense.x_output_size;
	switch (x->x_layers[index].type) {
		case 0:
			layer_output = x->x_layers[index].dense.x_output;
			act_output = x->x_layers[index].dense.x_act_output;
			output_size = x->x_layers[index].dense.x_output_size;
			break;
		default:
			break;
	}
	if (!x->x_confidences) {
		/* in case we're thresholding confidence
		   and outputting class predictions */
		x->x_is_confident = 0;
	}
	if ((x->x_is_training || x->x_is_validating) && x->x_layers[index].type == 0) {
		layer_output = x->x_layers[index].dense.x_output_train;
		act_output = x->x_layers[index].dense.x_act_output_train;
	}
	for (i = 0; i < input_size; i++) {
		/* initialize with maximum place holder */
		t_float max_val = layer_output[i][0];
		t_float sum = 0.0;
		int max_ndx = 0;
		/* we make seperate runs on the outputs of each sample
		   so we can first get the maximum value in the row
		   then apply the exponent defined in the second run
		   get the sum of the outputs, and finally divide
		   the activation outputs by this sum */
		for (j = 0; j < output_size; j++) {
			if (layer_output[i][j] > max_val) {
				max_val = layer_output[i][j];
				max_ndx = j;
			}
		}
		for (j = 0; j < output_size; j++) {
			act_output[i][j] = exp(layer_output[i][j] - max_val);
			sum += act_output[i][j];
		}
		for (j = 0; j < output_size; j++) {
			act_output[i][j] /= sum;
			if (act_output[i][j] > x->x_confidence_thresh) {
				x->x_is_confident = 1;
			}
			if (out && x->x_confidences) {
				/* if we want a list of confidences instead of a predicted class
				   j here is a dummy argument, but set_vec_val() is used for code integrity */
				set_vec_val(x, outvec, act_output[i][j], j);
				outvec++;
			}
			if (x->x_is_training && x->x_layers[index].type == 0) {
				x->x_layers[index].dense.x_act_output_transposed[j][i] = act_output[i][j];
			}
		}
		if (out && !x->x_confidences) {
			/* when we're outputting class predictions and not confidence lists
			   we check if the confidence is high enough, based on its threshold */
			if (x->x_is_confident) {
				/* no need to call set_vec_val() since we don't even have
				   an index to pass (could pass i, but it is meaningless) */
				SETFLOAT(outvec, (t_float)max_ndx);
				out_size = 1.0;
			}
			else {
				/* no need to call set_vec_val() here either, we're also
				   outputting out the second outlet, so the whole thing
				   is different in this case */
				SETFLOAT(outvec, (t_float)max_ndx);
				outvec++;
				SETFLOAT(outvec, act_output[i][max_ndx]);
				outlet_list(x->x_non_conf_out, &s_list, 2, x->x_outvec);
				out_size = 0.0;
			}
		}
	}
	return out_size;
}

/* backward passes */

static void activation_backward(t_neuralnet *x, t_float **dvalues, int index)
{
	int act_index = x->x_layers[index].x_activation_index;
	int i, j;
	t_atom *outvec, *storevec;
	t_float **act_output_train = x->x_layers[index].dense.x_act_output_train;
	int output_size = x->x_layers[index].dense.x_output_size;
	/* if we want to output predictions during training, we do it in the back propagation
	   to free the forward pass from any such task, as the forward pass is called for
	   real predictions as well
	   we output a list with the index of the input sample and the list of the predicitons */
	if (index == x->x_num_layers - 1 && x->x_output_during_training) {
		storevec = (t_atom *)getbytes((t_float)(x->x_output_size+1) * sizeof(t_atom));
		outvec = storevec;
		switch (x->x_layers[index].type) {
			case 0:
				act_output_train = x->x_layers[index].dense.x_act_output_train;
				output_size = x->x_layers[index].dense.x_output_size;
				break;
			default:
				break;
		}
		for (i = 0; i < x->x_num_in_samples; i++) {
			SETFLOAT(outvec, (t_float)i);
			outvec++;
			for (j = 0; j < output_size; j++) {
				set_vec_val(x, outvec, act_output_train[i][j], j);
				outvec++;
			}
			outlet_list(x->x_out, &s_list, x->x_output_size+1, storevec);
			outvec = storevec; /* reset the vector to its starting position */
		}
		freebytes(storevec, (x->x_output_size+1) * sizeof(t_atom));
	}
	x->x_act_back[act_index](x, dvalues, index);
}

static void sigmoid_backward(t_neuralnet *x, t_float **dvalues, int index)
{
	int i, j;
	t_float **dinput = x->x_layers[index].dense.x_act_dinput;
	t_float **output = x->x_layers[index].dense.x_act_output_train;
	int output_size = x->x_layers[index].dense.x_output_size;
	switch (x->x_layers[index].type) {
		case 0:
			dinput = x->x_layers[index].dense.x_act_dinput;
			output = x->x_layers[index].dense.x_act_output_train;
			output_size = x->x_layers[index].dense.x_output_size;
			break;
		default:
			break;
	}
	for (i = 0; i < x->x_num_in_samples; i++) {
		for (j = 0; j < output_size; j++) {
			dinput[i][j] = dvalues[i][j] * (1.0 - output[i][j]) * output[i][j];
		}
	}
}

static void bipolar_sigmoid_backward(t_neuralnet *x, t_float **dvalues, int index)
{
	int i, j;
	t_float **dinput = x->x_layers[index].dense.x_act_dinput;
	/*t_float **output = x->x_layers[index].x_act_output_train;*/
	t_float **prev_output = x->x_layers[index].dense.x_output_train;
	int output_size = x->x_layers[index].dense.x_output_size;
	switch (x->x_layers[index].type) {
		case 0:
			dinput = x->x_layers[index].dense.x_act_dinput;
			prev_output = x->x_layers[index].dense.x_output_train;
			output_size = x->x_layers[index].dense.x_output_size;
			break;
		default:
			break;
	}
	for (i = 0; i < x->x_num_in_samples; i++) {
		for (j = 0; j < output_size; j++) {
			t_float denominator = (1.0 + exp(-prev_output[i][j]));
			dinput[i][j] = ((2.0 * exp(-prev_output[i][j])) / (denominator * denominator)) * dvalues[i][j];
		}
	}
}

static void relu_backward(t_neuralnet *x, t_float **dvalues, int index)
{
	int i, j;
	t_float **prev_output = x->x_layers[index].dense.x_output_train;
	t_float **act_dinput = x->x_layers[index].dense.x_act_dinput;
	int output_size = x->x_layers[index].dense.x_output_size;
	switch (x->x_layers[index].type) {
		case 0:
			prev_output = x->x_layers[index].dense.x_output_train;
			act_dinput = x->x_layers[index].dense.x_act_dinput;
			output_size = x->x_layers[index].dense.x_output_size;
			break;
		default:
			break;
	}
	for (i = 0; i < x->x_num_in_samples; i++) {
		for (j = 0; j < output_size; j++) {
			if (prev_output[i][j] <= 0.0) {
				act_dinput[i][j] = 0.0;
			}
			else {
				act_dinput[i][j] = dvalues[i][j];
			}
		}
	}
}

static void leaky_relu_backward(t_neuralnet *x, t_float **dvalues, int index)
{
	int i, j;
	t_float **prev_output = x->x_layers[index].dense.x_output_train;
	t_float **act_dinput = x->x_layers[index].dense.x_act_dinput;
	int output_size = x->x_layers[index].dense.x_output_size;
	switch (x->x_layers[index].type) {
		case 0:
			prev_output = x->x_layers[index].dense.x_output_train;
			act_dinput = x->x_layers[index].dense.x_act_dinput;
			output_size = x->x_layers[index].dense.x_output_size;
			break;
		default:
			break;
	}
	for (i = 0; i < x->x_num_in_samples; i++) {
		for (j = 0; j < output_size; j++) {
			if (prev_output[i][j] <= 0.0) {
				act_dinput[i][j] = dvalues[i][j] * x->x_leaky_relu_coeff;
			}
			else {
				act_dinput[i][j] = dvalues[i][j];
			}
		}
	}
}

static void rect_softplus_backward(t_neuralnet *x, t_float **dvalues, int index)
{
	int i, j;
	t_float **dinput = x->x_layers[index].dense.x_act_dinput;
	/*t_float **output = x->x_layers[index].x_act_output_train;*/
	t_float **prev_output = x->x_layers[index].dense.x_output_train;
	int output_size = x->x_layers[index].dense.x_output_size;
	switch (x->x_layers[index].type) {
		case 0:
			dinput = x->x_layers[index].dense.x_act_dinput;
			prev_output = x->x_layers[index].dense.x_output_train;
			output_size = x->x_layers[index].dense.x_output_size;
			break;
		default:
			break;
	}
	for (i = 0; i < x->x_num_in_samples; i++) {
		for (j = 0; j < output_size; j++) {
			dinput[i][j] = (exp(prev_output[i][j]) / (1.0 + exp(prev_output[i][j]))) * dvalues[i][j];
		}
	}
}

static void tanh_backward(t_neuralnet *x, t_float **dvalues, int index)
{
	int i, j;
	t_float **dinput = x->x_layers[index].dense.x_act_dinput;
	/*t_float **output = x->x_layers[index].x_act_output_train;*/
	t_float **prev_output = x->x_layers[index].dense.x_output_train;
	int output_size = x->x_layers[index].dense.x_output_size;
	switch (x->x_layers[index].type) {
		case 0:
			dinput = x->x_layers[index].dense.x_act_dinput;
			prev_output = x->x_layers[index].dense.x_output_train;
			output_size = x->x_layers[index].dense.x_output_size;
			break;
		default:
			break;
	}
	for (i = 0; i < x->x_num_in_samples; i++) {
		for (j = 0; j < output_size; j++) {
			dinput[i][j] = (4 / pow((exp(prev_output[i][j]) + exp(-prev_output[i][j])), 2)) * dvalues[i][j];
		}
	}
}

static void linear_backward(t_neuralnet *x, t_float **dvalues, int index)
{
	int i, j;
	t_float **act_dinput = x->x_layers[index].dense.x_act_dinput;
	int output_size = x->x_layers[index].dense.x_output_size;
	switch (x->x_layers[index].type) {
		case 0:
			act_dinput = x->x_layers[index].dense.x_act_dinput;
			output_size = x->x_layers[index].dense.x_output_size;
			break;
		default:
			break;
	}
	for (i = 0; i < x->x_num_in_samples; i++) {
		for (j = 0; j < output_size; j++) {
			act_dinput[i][j] = dvalues[i][j];
		}
	}
}

static void softmax_backward(t_neuralnet *x, t_float **dvalues, int index)
{
	int i, j, k;
	t_float **dot_product = x->x_layers[index].dense.x_dot_product;
	t_float **act_output = x->x_layers[index].dense.x_act_output_train;
	t_float **jac_mtx = x->x_layers[index].dense.x_jac_mtx;
	t_float **eye = x->x_layers[index].dense.x_eye;
	t_float **act_dinput = x->x_layers[index].dense.x_act_dinput;
	int output_size = x->x_layers[index].dense.x_output_size;
	switch (x->x_layers[index].type) {
		case 0:
			dot_product = x->x_layers[index].dense.x_dot_product;
			act_output = x->x_layers[index].dense.x_act_output_train;
			jac_mtx = x->x_layers[index].dense.x_jac_mtx;
			eye = x->x_layers[index].dense.x_eye;
			act_dinput = x->x_layers[index].dense.x_act_dinput;
			output_size = x->x_layers[index].dense.x_output_size;
			break;
		default:
			break;
	}
	for (i = 0; i < x->x_num_in_samples; i++) {
		for (j = 0; j < output_size; j++) {
			act_dinput[i][j] = 0.0;
			for (k = 0; k < output_size; k++) {
				/* np.dot(single_output, single_output.T) */
				dot_product[j][k] = act_output[i][j] * act_output[i][k];
				/* inside the parenthesis is the np.diagflat(single_output) */
				jac_mtx[j][k] = (act_output[i][j] * eye[j][k]) - dot_product[j][k];
				/* finally the dot product of jac_mtx and dvalues */
				act_dinput[i][j] += jac_mtx[j][k] * dvalues[i][k];
			}
		}
	}
}

/* miscellaneous functions that concern activation functions */

static void set_activation_function(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv)
{
	int i;
	int layer;
	int found_func = 0;
	(void)(s); /* unused */	

	if (argc != 2) {
		pd_error(x, "set_activation_function takes 2 arguments");
		return;
	}
	if (argv[0].a_type != A_FLOAT || argv[1].a_type != A_SYMBOL) {
		pd_error(x, "first argument must be a float and second must be a symbol");
		return;
	}

	layer = (int)atom_getfloat(argv);
	t_symbol *func = atom_getsymbol(argv+1);

	if (layer >= x->x_num_layers) {
		pd_error(x,"max layer is %d", (int)(x->x_num_layers-1));
		return;
	}
	for (i = 0; i < NUM_ACT_FUNCS; i++) {
		if (strcmp(func->s_name, x->x_act_func_str[i]) == 0) {
			x->x_layers[layer].x_activation_index = i;
			found_func = 1;
			break;
		}
	}
	if (!found_func) {
		pd_error(x, "%s: no such activation function", func->s_name);
	}

	if (x->x_layers[layer].type == 0) {
		if (x->x_layers[layer].x_activation_index == SOFTMAX_INDEX) {
			/* we need to know if we must allocate the eye
			   and jacobian matrices for every layer which are
			   used by the softmax activation function */
			x->x_layers[layer].dense.x_allocate_softmax = 1;
		}
		else {
			if (x->x_layers[layer].dense.x_allocate_softmax) {
				x->x_layers[layer].dense.x_allocate_softmax = 0;
			}
		}
	}
}

static void set_confidences(t_neuralnet *x, t_float f)
{
	/* set whether we should output a list of confidence values
	   for classification only */
	if (x->x_output_type == 1) {
		x->x_confidences = (int)f;
	}
	else {
		pd_error(x, "confidences boolean works for classification only");
	}
}

static void confidence_thresh(t_neuralnet *x, t_float f)
{
	/* sets whether the prediction will be sent out the first outlet
	   or, in case the maximum confidence is below this threshold,
	   out the second outlet together with the confidence, as a list */
	if (f < 0.0) {
		pd_error(x, "confidence threshold must be positive");
		return;
	}
	x->x_confidence_thresh = f;
}

static void set_leaky_relu_coeff(t_neuralnet *x, t_float f)
{
	if (f < 0.0) {
		pd_error(x, "leaky ReLU coefficient can't be negative");
		return;
	}
	if (f >= 1.0) {
		pd_error(x, "can't set a leaky ReLU coefficient of 1 or greater");
		return;
	}
	x->x_leaky_relu_coeff = f;
}

/*************************** loss functions **************************/

/* forward passes */

static t_float loss_forward(t_neuralnet *x)
{
	/* call loss function from function pointer array */
	return x->x_loss_funcs[x->x_loss_index](x);
}

/* Mean Squared Error */
static t_float mse_forward(t_neuralnet *x)
{
	int i, j;
	double sample_losses = 0.0;
	double sample_loss;
	double diff;
	t_float mean_loss;
	int output_size = x->x_layers[x->x_num_layers-1].dense.x_output_size;
	t_float **output_train = x->x_layers[x->x_num_layers-1].dense.x_act_output_train;
	switch (x->x_layers[x->x_num_layers-1].type) {
		case 0:
			output_train = x->x_layers[x->x_num_layers-1].dense.x_act_output_train;
			output_size = x->x_layers[x->x_num_layers-1].dense.x_output_size;
			break;
	}
	for (i = 0; i < x->x_num_in_samples; i++) {
		for (j = 0; j < output_size; j++) {
			diff = (double)(x->x_target_vals[i][j] - output_train[i][j]);
			sample_loss = pow(diff, 2);
			sample_losses += sample_loss;
		}
	}
	mean_loss = (t_float)(sample_losses / (double)(x->x_num_in_samples * output_size));
	return mean_loss;
}

/* Mean Absolute Error */
static t_float mae_forward(t_neuralnet *x)
{
	int i, j;
	double sample_losses = 0.0;
	t_float mean_loss;
	int output_size = x->x_layers[x->x_num_layers-1].dense.x_output_size;
	t_float **output_train = x->x_layers[x->x_num_layers-1].dense.x_act_output_train;
	switch (x->x_layers[x->x_num_layers-1].type) {
		case 0:
			output_train = x->x_layers[x->x_num_layers-1].dense.x_act_output_train;
			output_size = x->x_layers[x->x_num_layers-1].dense.x_output_size;
			break;
	}
	for (i = 0; i < x->x_num_in_samples; i++) {
		for (j = 0; j < output_size; j++) {
			sample_losses += fabs((double)(x->x_target_vals[i][j] - output_train[i][j]));
		}
	}
	mean_loss = (t_float)(sample_losses / (double)(x->x_num_in_samples * output_size));
	return mean_loss;
}

/* Categorical Cross-Entropy */
static t_float categ_x_entropy_loss_forward(t_neuralnet *x)
{
	int i, j;
	t_float upper_clip_bound = 1.0 - 1e-07;
	t_float mean_loss = 0.0;
	int output_size = x->x_layers[x->x_num_layers-1].dense.x_output_size;
	t_float **output_train = x->x_layers[x->x_num_layers-1].dense.x_act_output_train;
	switch (x->x_layers[x->x_num_layers-1].type) {
		case 0:
			output_train = x->x_layers[x->x_num_layers-1].dense.x_act_output_train;
			output_size = x->x_layers[x->x_num_layers-1].dense.x_output_size;
			break;
	}
	for (i = 0; i < x->x_num_in_samples; i++) {
		t_float confidence = 0.0;
		for (j = 0; j < output_size; j++) {
			t_float output_clipped = output_train[i][j];
			/* clip data to prevent division by 0 */
			if (output_clipped > upper_clip_bound) output_clipped = upper_clip_bound;
			else if (output_clipped < 1e-07) output_clipped = 1e-07;
			confidence += (output_clipped * x->x_target_vals[i][j]);
		}
		mean_loss += -log(confidence);
	}
	mean_loss /= (x->x_num_in_samples * output_size);
	return mean_loss;
}

/* Binary Cross-Entropy */
static t_float bin_x_entropy_loss_forward(t_neuralnet *x)
{
	int i, j;
	t_float upper_clip_bound = 1.0 - 1e-07;
	t_float mean_loss = 0.0;
	int output_size = x->x_layers[x->x_num_layers-1].dense.x_output_size;
	t_float **output_train = x->x_layers[x->x_num_layers-1].dense.x_act_output_train;
	switch (x->x_layers[x->x_num_layers-1].type) {
		case 0:
			output_train = x->x_layers[x->x_num_layers-1].dense.x_act_output_train;
			output_size = x->x_layers[x->x_num_layers-1].dense.x_output_size;
			break;
	}
	for (i = 0; i < x->x_num_in_samples; i++) {
		for (j = 0; j < output_size; j++) {
			t_float output_clipped = output_train[i][j];
			/* clip data to prevent division by 0 */
			if (output_clipped > upper_clip_bound) output_clipped = upper_clip_bound;
			else if (output_clipped < 1e-07) output_clipped = 1e-07;
			mean_loss += -((x->x_target_vals[i][j] * log(output_clipped)) + \
					(1.0 - x->x_target_vals[i][j]) * \
					log(1.0 - output_clipped));
		}
	}
	mean_loss /= (x->x_num_in_samples * output_size);
	return mean_loss;
}

/* backward passes */

static void loss_backward(t_neuralnet *x)
{
	/* call loss function from function pointer array */
	x->x_loss_back[x->x_loss_index](x);
}

static void mse_backward(t_neuralnet *x)
{
	int i, j;
	int output_size = x->x_layers[x->x_num_layers-1].dense.x_output_size;
	t_float **output_train = x->x_layers[x->x_num_layers-1].dense.x_act_output_train;
	switch (x->x_layers[x->x_num_layers-1].type) {
		case 0:
			output_train = x->x_layers[x->x_num_layers-1].dense.x_act_output_train;
			output_size = x->x_layers[x->x_num_layers-1].dense.x_output_size;
			break;
		default:
			break;
	}
	for (i = 0; i < x->x_num_in_samples; i++) {
		for (j = 0; j < output_size; j++) {
			t_float dvalue = output_train[i][j];
			t_float dinput = -2.0 * (x->x_target_vals[i][j] - dvalue) / (t_float)output_size;
			x->x_loss_dinput[i][j] = dinput / x->x_num_in_samples;
		}
	}
}

static void mae_backward(t_neuralnet *x)
{
	int i, j;
	int output_size = x->x_layers[x->x_num_layers-1].dense.x_output_size;
	t_float **output_train = x->x_layers[x->x_num_layers-1].dense.x_act_output_train;
	switch (x->x_layers[x->x_num_layers-1].type) {
		case 0:
			output_train = x->x_layers[x->x_num_layers-1].dense.x_act_output_train;
			output_size = x->x_layers[x->x_num_layers-1].dense.x_output_size;
			break;
		default:
			break;
	}
	for (i = 0; i < x->x_num_in_samples; i++) {
		for (j = 0; j < output_size; j++) {
			t_float dvalue = output_train[i][j];
			t_float dinput = x->x_target_vals[i][j] - dvalue;
			/* apply np.sign() */
			if (dinput >= 0.0) {
				x->x_loss_dinput[i][j] = 1.0;
			}
			else {
				x->x_loss_dinput[i][j] = -1.0;
			}
			x->x_loss_dinput[i][j] /= (t_float)output_size;
			x->x_loss_dinput[i][j] /=  x->x_num_in_samples;
		}
	}
}

static void categ_x_entropy_loss_backward(t_neuralnet *x)
{
	int i, j;
	int output_size = x->x_layers[x->x_num_layers-1].dense.x_output_size;
	t_float **output_train = x->x_layers[x->x_num_layers-1].dense.x_act_output_train;
	switch (x->x_layers[x->x_num_layers-1].type) {
		case 0:
			output_train = x->x_layers[x->x_num_layers-1].dense.x_act_output_train;
			output_size = x->x_layers[x->x_num_layers-1].dense.x_output_size;
			break;
		default:
			break;
	}
	for (i = 0; i < x->x_num_in_samples; i++) {
		for (j = 0; j < output_size; j++) {
			t_float dvalue = output_train[i][j];
			x->x_loss_dinput[i][j] = -x->x_target_vals[i][j] / dvalue;
			x->x_loss_dinput[i][j] /= x->x_num_in_samples;
		}
	}
}

static void bin_x_entropy_loss_backward(t_neuralnet *x)
{
	int i, j;
	t_float upper_clip_bound = 1.0 - 1e-07;
	int output_size = x->x_layers[x->x_num_layers-1].dense.x_output_size;
	t_float **output_train = x->x_layers[x->x_num_layers-1].dense.x_act_output_train;
	switch (x->x_layers[x->x_num_layers-1].type) {
		case 0:
			output_train = x->x_layers[x->x_num_layers-1].dense.x_act_output_train;
			output_size = x->x_layers[x->x_num_layers-1].dense.x_output_size;
			break;
		default:
			break;
	}
	for (i = 0; i < x->x_num_in_samples; i++) {
		for (j = 0; j < output_size; j++) {
			t_float dvalue_clipped = output_train[i][j];
			/* clip data to prevent division by 0 */
			if (dvalue_clipped > upper_clip_bound) dvalue_clipped = upper_clip_bound;
			else if (dvalue_clipped < 1e-07) dvalue_clipped = 1e-07;
			x->x_loss_dinput[i][j] = -(x->x_target_vals[i][j] / dvalue_clipped - \
					(1.0 - x->x_target_vals[i][j]) / \
					(1.0 - dvalue_clipped)) / x->x_output_size;
			x->x_loss_dinput[i][j] /= x->x_num_in_samples;
		}
	}
}

static t_float regularization_loss(t_neuralnet *x)
{
	int i, j, k;
	t_float reg_loss = 0.0;
	t_float sum = 0.0;
	for (i = 0; i < x->x_num_layers; i++) {
		if (x->x_layers[i].type == 0 && x->x_layers[i].x_weight_regularizer_l1 > 0.0) {
			for (j = 0; j < x->x_layers[i].dense.x_input_size; j++) {
				for (k = 0; k < x->x_layers[i].dense.x_output_size; k++) {
					sum += fabs(x->x_layers[i].dense.x_weights[j][k]);
				}
			}
			reg_loss += x->x_layers[i].x_weight_regularizer_l1 * sum;
		}
	}
	sum = 0.0;
	for (i = 0; i < x->x_num_layers; i++) {
		if (x->x_layers[i].type == 0 && x->x_layers[i].x_weight_regularizer_l2 > 0.0) {
			for (j = 0; j < x->x_layers[i].dense.x_input_size; j++) {
				for (k = 0; k < x->x_layers[i].dense.x_output_size; k++) {
					sum += (x->x_layers[i].dense.x_weights[j][k] * x->x_layers[i].dense.x_weights[j][k]);
				}
			}
			reg_loss += x->x_layers[i].x_weight_regularizer_l2 * sum;
		}
	}
	sum = 0.0;
	for (i = 0; i < x->x_num_layers; i++) {
		if (x->x_layers[i].type == 0 && x->x_layers[i].x_bias_regularizer_l1 > 0.0) {
			for (j = 0; j < x->x_layers[i].dense.x_output_size; j++) {
				sum += fabs(x->x_layers[i].dense.x_biases[j]);
			}
			reg_loss += x->x_layers[i].x_bias_regularizer_l1 * sum;
		}
	}
	sum = 0.0;
	for (i = 0; i < x->x_num_layers; i++) {
		if (x->x_layers[i].type == 0 && x->x_layers[i].x_bias_regularizer_l2 > 0.0) {
			for (j = 0; j < x->x_layers[i].dense.x_output_size; j++) {
				sum += (x->x_layers[i].dense.x_biases[j] * x->x_layers[i].dense.x_biases[j]);
			}
			reg_loss += x->x_layers[i].x_bias_regularizer_l2 * sum;
		}
	}
	return reg_loss;
}

static void set_loss_function(t_neuralnet *x, t_symbol *func)
{
	int i, found_func = 0;
	for (i = 0; i < NUM_LOSS_FUNCS; i++) {
		if (strcmp(func->s_name, x->x_loss_func_str[i]) == 0) {
			x->x_loss_index = i;
			found_func = 1;
			break;
		}
	}
	if (!found_func) {
		pd_error(x, "%s: no such loss function", func->s_name);
	}
}

static void desired_loss(t_neuralnet *x, t_float f)
{
	/* desired loss is used to determine whether after training
	   the training dataset will be kept or freed, so we can retrain
	   the model in case of non-satisfactory results without having
	   to re-import the training dataset */
	if (f < 0.0) {
		pd_error(x, "desired loss must be positive");
		return;
	}
	x->x_desired_loss = f;
}

/************************** optimizer functions ************************/

static void optimizer_pre_update(t_neuralnet *x)
{
	if (x->x_decay) {
		x->x_current_learning_rate = x->x_learning_rate * (1.0 / \
				(1.0 + x->x_decay * (t_float)x->x_iterations));
	}
}

static void optimizer_adam_update(t_neuralnet *x, int index)
{
	int i, j;
	double weight_cache, bias_cache;
	for (i = 0; i < x->x_layers[index].dense.x_input_size; i++) {
		for (j = 0; j < x->x_layers[index].dense.x_output_size; j++) {
			x->x_layers[index].dense.x_weight_momentums[i][j] = x->x_beta_1 * \
														  x->x_layers[index].dense.x_weight_momentums[i][j] + \
														  (1.0 - x->x_beta_1) * \
														  x->x_layers[index].dense.x_dweights[i][j];
			/* x->x_iterations is 0 at first pass and we need to start with 1
			   hence the (x->x_iterations + 1) */
			x->x_layers[index].dense.x_weight_momentums_corrected[i][j] = x->x_layers[index].dense.x_weight_momentums[i][j] / \
																	(1.0 - pow(x->x_beta_1, (x->x_iterations + 1)));
			weight_cache = pow((double)x->x_layers[index].dense.x_dweights[i][j], 2);
			x->x_layers[index].dense.x_weight_cache[i][j] = x->x_beta_2 * \
													  x->x_layers[index].dense.x_weight_cache[i][j] + \
													  (1.0 - x->x_beta_2) * weight_cache;
			x->x_layers[index].dense.x_weight_cache_corrected[i][j] = x->x_layers[index].dense.x_weight_cache[i][j] / \
																(1.0 - pow(x->x_beta_2, (x->x_iterations + 1)));
			x->x_layers[index].dense.x_weights[i][j] += -x->x_current_learning_rate * \
												  x->x_layers[index].dense.x_weight_momentums_corrected[i][j] / \
												  (sqrt(x->x_layers[index].dense.x_weight_cache_corrected[i][j]) + \
												   x->x_epsilon);
			x->x_layers[index].dense.x_weights_transposed[j][i] = x->x_layers[index].dense.x_weights[i][j];
		}
	}
	for (i = 0; i < x->x_layers[index].dense.x_output_size; i++) {
		x->x_layers[index].dense.x_bias_momentums[i] = x->x_beta_1 * \
												 x->x_layers[index].dense.x_bias_momentums[i] + \
												 (1.0 - x->x_beta_1) * \
												 x->x_layers[index].dense.x_dbiases[i];
		x->x_layers[index].dense.x_bias_momentums_corrected[i] = x->x_layers[index].dense.x_bias_momentums[i] / \
														   (1.0 - pow(x->x_beta_1, (x->x_iterations + 1)));
		bias_cache = pow((double)x->x_layers[index].dense.x_dbiases[i], 2);
		x->x_layers[index].dense.x_bias_cache[i] = x->x_beta_2 * \
											 x->x_layers[index].dense.x_bias_cache[i] + \
											 (1.0 - x->x_beta_2) * bias_cache;
		x->x_layers[index].dense.x_bias_cache_corrected[i] = x->x_layers[index].dense.x_bias_cache[i] / \
													   (1.0 - pow(x->x_beta_2, (x->x_iterations + 1)));
		x->x_layers[index].dense.x_biases[i] += -x->x_current_learning_rate * \
										  x->x_layers[index].dense.x_bias_momentums_corrected[i] / \
										  (sqrt(x->x_layers[index].dense.x_bias_cache_corrected[i]) + \
										   x->x_epsilon);
	}
}

static void optimizer_sgd_update(t_neuralnet *x, int index)
{
	int i, j;
	t_float weight_updates, bias_updates;
	if (x->x_momentum) {
		for (i = 0; i < x->x_layers[index].dense.x_input_size; i++) {
			for (j = 0; j < x->x_layers[index].dense.x_output_size; j++) {
				weight_updates = x->x_momentum * \
								 x->x_layers[index].dense.x_weight_momentums[i][j] - \
								 x->x_current_learning_rate * \
								 x->x_layers[index].dense.x_dweights[i][j];
				x->x_layers[index].dense.x_weight_momentums[i][j] = weight_updates;
				x->x_layers[index].dense.x_weights[i][j] += weight_updates;
			}
		}
		for (i = 0; i < x->x_layers[index].dense.x_output_size; i++) {
			bias_updates = x->x_momentum * \
						   x->x_layers[index].dense.x_bias_momentums[i] - \
						   x->x_current_learning_rate * \
						   x->x_layers[index].dense.x_dbiases[i];
			x->x_layers[index].dense.x_bias_momentums[i] = bias_updates;
			x->x_layers[index].dense.x_biases[i] += bias_updates;
		}
	}
	else {
		for (i = 0; i < x->x_layers[index].dense.x_input_size; i++) {
			for (j = 0; j < x->x_layers[index].dense.x_output_size; j++) {
				weight_updates = -x->x_current_learning_rate * \
								 x->x_layers[index].dense.x_dweights[i][j];
				x->x_layers[index].dense.x_weights[i][j] += weight_updates;
			}
		}
		for (i = 0; i < x->x_layers[index].dense.x_output_size; i++) {
			bias_updates = -x->x_current_learning_rate * \
						   x->x_layers[index].dense.x_dbiases[i];
			x->x_layers[index].dense.x_biases[i] += bias_updates;
		}
	}
}

static void optimizer_adagrad_update(t_neuralnet *x, int index)
{
	int i, j;
	for (i = 0; i < x->x_layers[index].dense.x_input_size; i++) {
		for (j = 0; j < x->x_layers[index].dense.x_output_size; j++) {
			x->x_layers[index].dense.x_weight_cache[i][j] += (t_float)pow((double)x->x_layers[index].dense.x_dweights[i][j], 2);
			x->x_layers[index].dense.x_weights[i][j] += -x->x_current_learning_rate * \
												  x->x_layers[index].dense.x_dweights[i][j] / \
												  (sqrt(x->x_layers[index].dense.x_weight_cache[i][j]) + x->x_epsilon);
		}
	}
	for (i = 0; i < x->x_layers[index].dense.x_output_size; i++) {
		x->x_layers[index].dense.x_bias_cache[i] += (t_float)pow((double)x->x_layers[index].dense.x_dbiases[i], 2);
		x->x_layers[index].dense.x_biases[i] += -x->x_current_learning_rate * \
										  x->x_layers[index].dense.x_dbiases[i] / \
										  (sqrt(x->x_layers[index].dense.x_bias_cache[i]) + x->x_epsilon);
	}
}

static void optimizer_rms_prop_update(t_neuralnet *x, int index)
{
	int i, j;
	for (i = 0; i < x->x_layers[index].dense.x_input_size; i++) {
		for (j = 0; j < x->x_layers[index].dense.x_output_size; j++) {
			x->x_layers[index].dense.x_weight_cache[i][j] = x->x_rho * \
													  x->x_layers[index].dense.x_weight_cache[i][j] + \
													  (1.0 - x->x_rho) * \
													  (t_float)pow((double)x->x_layers[index].dense.x_dweights[i][j], 2);
			x->x_layers[index].dense.x_weights[i][j] += -x->x_current_learning_rate * \
												  x->x_layers[index].dense.x_dweights[i][j] / \
												  (sqrt(x->x_layers[index].dense.x_weight_cache[i][j]) + x->x_epsilon);
		}
	}
	for (i = 0; i < x->x_layers[index].dense.x_output_size; i++) {
		x->x_layers[index].dense.x_bias_cache[i] = x->x_rho * \
											 x->x_layers[index].dense.x_bias_cache[i] + \
											 (1.0 - x->x_rho) * \
											 (t_float)pow((double)x->x_layers[index].dense.x_dbiases[i], 2);
		x->x_layers[index].dense.x_biases[i] += -x->x_current_learning_rate * \
										  x->x_layers[index].dense.x_dbiases[i] / \
										  (sqrt(x->x_layers[index].dense.x_bias_cache[i]) + x->x_epsilon);
	}
}

static void optimizer_post_update(t_neuralnet *x)
{
	x->x_iterations += 1;
}

static void set_optimizer(t_neuralnet *x, t_symbol *func)
{
	int i, found_func = 0;
	for (i = 0; i < NUM_OPTIMIZER_FUNCS; i++) {
		if (strcmp(func->s_name, x->x_optimizer_func_str[i]) == 0) {
			x->x_optimizer_index = i;
			found_func = 1;
			break;
		}
	}
	if (!found_func) {
		pd_error(x, "%s: no such optimizer", func->s_name);
	}
}

static void set_learning_rate(t_neuralnet *x, t_float f)
{
	if (f < 0.0) {
		pd_error(x, "learning rate must be positive");
		return;
	}
	x->x_learning_rate = f;
}

static void set_decay(t_neuralnet *x, t_float f)
{
	if (f < 0.0) {
		pd_error(x, "decay must be positive");
		return;
	}
	x->x_decay = f;
}

static void set_beta1(t_neuralnet *x, t_float f)
{
	if (f < 0.0) {
		pd_error(x, "beta1 must be positive");
		return;
	}
	x->x_beta_1 = f;
}

static void set_beta2(t_neuralnet *x, t_float f)
{
	if (f < 0.0) {
		pd_error(x, "beta2 must be positive");
		return;
	}
	x->x_beta_2 = f;
}

static void set_epsilon(t_neuralnet *x, t_float f)
{
	if (f < 0.0) {
		pd_error(x, "epsilon must be positive");
		return;
	}
	x->x_epsilon = f;
}

static void set_rho(t_neuralnet *x, t_float f)
{
	if (f < 0.0) {
		pd_error(x, "rho must be positive");
		return;
	}
	x->x_rho = f;
}

static void set_momentum(t_neuralnet *x, t_float f)
{
	if (f < 0.0) {
		pd_error(x, "momentum must be positive");
		return;
	}
	x->x_momentum = f;
}

/************************ accuracy functions *************************/

static t_float standard_deviation(t_float **data, int rows, int cols)
{
	t_float sum = 0.0, mean, SD = 0.0;
	int i, j;
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			sum += data[i][j];
		}
	}
	mean = sum / (rows * cols);
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			SD += pow(data[i][j] - mean, 2);
		}
	}
	return sqrt(SD / (rows * cols));
}

static void set_accuracy_precision(t_neuralnet *x)
{
	x->x_acc_precision = standard_deviation(x->x_target_vals, x->x_num_in_samples,
			x->x_output_size) / x->x_accuracy_denominator;
}

static void set_accuracy_denominator(t_neuralnet *x, t_float f)
{
	if (f <= 0.0) {
		pd_error(x, "accuracy denominator must be positive");
		return;
	}
	x->x_accuracy_denominator = f;
	set_accuracy_precision(x);
}

static t_float get_accuracy(t_neuralnet *x)
{
	t_float **output = x->x_layers[x->x_num_layers-1].dense.x_act_output_train;
	t_float **target = x->x_target_vals;
	int output_size = x->x_layers[x->x_num_layers-1].dense.x_output_size;
	int i, j;
	t_float accuracy = 0.0;
	t_float denominator;
	switch (x->x_layers[x->x_num_layers-1].type) {
		case 0:
			output = x->x_layers[x->x_num_layers-1].dense.x_act_output_train;
			output_size = x->x_layers[x->x_num_layers-1].dense.x_output_size;
			break;
	}
	for (i = 0; i < x->x_num_in_samples; i++) {
		if (x->x_classification) {
			if (x->x_output_type == 1) {
				int max_ndx_pred = 0, max_ndx_target = 0;
				t_float max_val = -FLT_MAX;
				for (j = 0; j < output_size; j++) {
					/* get the index of the maximum confidence */
					if (output[i][j] >= max_val) {
						max_val = output[i][j];
						max_ndx_pred = j;
					}
					/* get the index of the target class */
					/* target is one-hot encoded */
					if (target[i][j] > 0.5) max_ndx_target = j;
				}
				accuracy += (max_ndx_pred == max_ndx_target ? 1.0 : 0.0);
			}
			/* for binary logistic regression we only need to compare the raw values */
			else if (x->x_output_type == 2) {
				for (j = 0; j < output_size; j++) {
					t_float comparison = 0.0;
					if ((output[i][j] > 0.5 && target[i][j] > 0.5) || \
							(output[i][j] < 0.5 && target[i][j] < 0.5)) {
						comparison = 1.0;
					}
					accuracy += comparison;
				}
			}
		}
		else {
			/* for regression we need to test if the difference between
			   output and target is smaller than the accuracy threshold */
			for (j = 0; j < output_size; j++) {
				accuracy += (((t_float)fabs(output[i][j] - target[i][j]) < x->x_acc_precision) ? 1.0 : 0.0);
			}
		}
	}
	/* since classification adds one for the predicted class and
	   0s for the rest of the classes, we must divide by the number
	   of samples only */
	if (x->x_classification) {
		denominator = x->x_num_in_samples;
	}
	/* in case of regression we must divide by the number of samples
	   times the number of output neurons of the network */
	else {
		denominator = x->x_num_in_samples * output_size;
	}
	accuracy /= denominator;
	return accuracy;
}

static void classification(t_neuralnet *x)
{
	x->x_classification = 1;
	x->x_output_type = 1;
}

static void binary_logistic_regression(t_neuralnet *x)
{
	x->x_classification = 1;
	x->x_output_type = 2;
}

static void regression(t_neuralnet *x)
{
	x->x_classification = 0;
	x->x_output_type = 0;
}

static void desired_accuracy(t_neuralnet *x, t_float f)
{
	if (f < 0.0) {
		pd_error(x, "desired accuracy must be positive");
		return;
	}
	x->x_desired_accuracy = f;
}

/******************** create and destroy network ********************/

static void add_layer(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv)
{
	int i;
	int input_size, output_size;
	t_symbol *dense_sym = gensym("dense");
	t_symbol *lstm_sym = gensym("lstm");
	t_symbol *gru_sym = gensym("gru");
	t_symbol *embedding_sym = gensym("embedding");

	t_symbol *arg_layer = atom_gensym(argv);
	(void)(s); /* unused */	

	if (arg_layer->s_name != dense_sym->s_name && arg_layer->s_name != lstm_sym->s_name && \
			arg_layer->s_name != gru_sym->s_name && arg_layer->s_name != embedding_sym->s_name) {
		pd_error(x, "%s: unkown layer type, must be one of the following: %s, %s, %s, %s",
			arg_layer->s_name, dense_sym->s_name, lstm_sym->s_name, gru_sym->s_name, embedding_sym->s_name);
		return;
	}

	argv++;

	if (arg_layer->s_name == dense_sym->s_name) {
		if (argc > 3) {
			pd_error(x, "dense layer takes maximum two arguments, number of inputs and number of outputs");
			return;
		}
		if (x->x_num_layers == 0 && argc != 3) {
			pd_error(x, "first dense layer needs both number of inputs and outputs");
			return;
		}
		if (x->x_num_layers > 0) {
			if ((x->x_layers[x->x_num_layers-1].type == 1 || x->x_layers[x->x_num_layers-1].type == 2) && \
					argc != 3) {
				pd_error(x, "a dense layer after an LSTM or GRU layer needs both number of inputs and outputs");
				return;
			}
			else if ((x->x_layers[x->x_num_layers-1].type == 0 || x->x_layers[x->x_num_layers-1].type == 3) && \
					argc == 3) {
				int prev_layer_outputs;
				if (x->x_layers[x->x_num_layers-1].type == 0) {
					prev_layer_outputs = x->x_layers[x->x_num_layers-1].dense.x_output_size;
				}
				else {
					prev_layer_outputs = x->x_layers[x->x_num_layers-1].embedding.x_output_dim;
				}
				if ((int)atom_getfloat(argv) != prev_layer_outputs) {
					pd_error(x, "number of inputs is not equal to number of outputs of previous layer");
					return;
				}
			}
		}
	}
	if (arg_layer->s_name == lstm_sym->s_name) {
		if (argc != 2) {
			pd_error(x, "LSTM layer takes one argument only, number of LSTM cells");
			return;
		}
	}
	if (arg_layer->s_name == gru_sym->s_name) {
		if (argc != 2) {
			pd_error(x, "GRU layer takes one argument only, number of GRU cells");
			return;
		}
	}
	if (arg_layer->s_name == embedding_sym->s_name) {
		if (x->x_num_layers == 0 && argc != 2) {
			pd_error(x, "first embedding layer needs both number of input and output dimensions");
			return;
		}
		if (x->x_layers[x->x_num_layers-1].type == 1 || x->x_layers[x->x_num_layers-1].type == 2) {
			pd_error(x, "cannot add embedding layer after LSTM or GRU layer");
			return;
		}
		if (x->x_layers[x->x_num_layers-1].type == 0 || x->x_layers[x->x_num_layers-1].type == 3) {
			int prev_layer_outputs;
			if (x->x_layers[x->x_num_layers-1].type == 0) {
				prev_layer_outputs = x->x_layers[x->x_num_layers-1].dense.x_output_size;
			}
			else {
				prev_layer_outputs = x->x_layers[x->x_num_layers-1].embedding.x_output_dim;
			}
			if (argc == 3 && (int)atom_getfloat(argv) != prev_layer_outputs) {
				pd_error(x, "number of input dimensions is not equal to outputs of previous layer");
				return;
			}
			else if (argc > 3) {
				pd_error(x, "embedding layer takes maximum two arguments");
				return;
			}
		}
	}

	x->x_num_layers++;
	if (x->x_layers == NULL) {
		x->x_layers = malloc(sizeof(t_layer) * x->x_num_layers);
		x->x_input_size = atom_getfloat(argv);
		if (argc == 3) x->x_output_size = atom_getfloat(argv+1);
		else x->x_output_size = x->x_input_size;
	}
	else {
		x->x_layers = (t_layer *)realloc(x->x_layers, sizeof(t_layer) * x->x_num_layers);
		if (argc == 3) x->x_output_size = atom_getfloat(argv+1);
		else x->x_output_size = atom_getfloat(argv);
	}
	if (arg_layer->s_name == dense_sym->s_name) {
		x->x_layers[x->x_num_layers-1].type = 0;
	}
	if (arg_layer->s_name == lstm_sym->s_name) {
		x->x_layers[x->x_num_layers-1].type = 1;
	}
	if (arg_layer->s_name == gru_sym->s_name) {
		x->x_layers[x->x_num_layers-1].type = 2;
	}
	if (arg_layer->s_name == embedding_sym->s_name) {
		x->x_layers[x->x_num_layers-1].type = 3;
	}

	if (x->x_layers[x->x_num_layers-1].type == 0) {
		if (argc == 3) {
			input_size = (int)atom_getfloat(argv);
			output_size = (int)atom_getfloat(argv+1);
		}
		else {
			if (x->x_layers[x->x_num_layers-2].type == 0) {
				input_size = x->x_layers[x->x_num_layers-2].dense.x_output_size;
			}
			else {
				input_size = x->x_layers[x->x_num_layers-2].embedding.x_output_dim;
			}
			output_size = (int)atom_getfloat(argv);
		}
	}
	layer_init_common_vars(x, x->x_num_layers-1);
	if (x->x_layers[x->x_num_layers-1].type == 0) {
		layer_dense_init(x, &x->x_layers[x->x_num_layers-1].dense, input_size, output_size);
	}
	if (!x->x_outvec_allocated) {
		/* output vector for outputting lists */
		x->x_outvec = (t_atom *)getbytes((t_float)x->x_output_size * sizeof(t_atom));
		for (i = 0; i < x->x_output_size; i++) {
			x->x_outvec[i].a_type = A_FLOAT;
		}
		x->x_outvec_allocated = 1;
	}
	else {
		x->x_outvec = (t_atom *)realloc(x->x_outvec, (t_float)x->x_output_size * sizeof(t_atom));
		for (i = 0; i < x->x_output_size; i++) {
			x->x_outvec[i].a_type = A_FLOAT;
		}
	}
	alloc_main_input(x, 1, 0);
	alloc_max_vals(x);
	x->x_net_created = 1;
}

static void create_net(t_neuralnet *x, int argc, t_atom *argv)
{
	int i;
	if (x->x_net_created) {
		destroy(x);
	}
	/* the first arguments is nr of input values
	   which don't have weights, biases, or an activation function
	   therefore, we embed the input to the first hidden layer */
	x->x_num_layers = argc - 1;
	x->x_input_size = atom_getfloat(argv);
	x->x_layers = malloc(sizeof(t_layer) * x->x_num_layers);
	for (i = 0; i < x->x_num_layers; i++) {
		x->x_layers[i].type = 0;
		layer_init_common_vars(x, i);
		layer_dense_init(x, &x->x_layers[i].dense, (int)atom_getfloat(argv+i), (int)atom_getfloat(argv+i+1));
	}
	x->x_layers_initialized = 1;
	/* x->x_output_size is stored for outlet_list() of the first outlet */
	x->x_output_size = (int)atom_getfloat(argv+argc-1);
	if (!x->x_outvec_allocated) {
		/* output vector for outputting lists */
		x->x_outvec = (t_atom *)getbytes((t_float)x->x_output_size * sizeof(t_atom));
		for (i = 0; i < x->x_output_size; i++) {
			x->x_outvec[i].a_type = A_FLOAT;
		}
		x->x_outvec_allocated = 1;
	}
	alloc_main_input(x, 1, 0);
	alloc_max_vals(x);
	x->x_net_created = 1;
}

static void create(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv)
{
	(void)(s); /* unused */	
	create_net(x, argc, argv);
}

/* deallocate basic network memory
   doesn't include training and testing allocated memory
   it's called through the "destroy" message and when
   the object is deleted */
static void destroy_dealloc(t_neuralnet *x)
{
	int i;
	if (x->x_layers_initialized) {
		for (i = 0; i < x->x_num_layers; i++) {
			if (x->x_layers[i].type == 0) dealloc_layer(&x->x_layers[i].dense);
		}
		x->x_layers_initialized = 0;
	}
	if (x->x_outvec_allocated) {
		freebytes(x->x_outvec, x->x_output_size * sizeof(t_atom));
		x->x_outvec_allocated = 0;
	}
	dealloc_main_input(x, x->x_old_num_in_samples);
	dealloc_max_vals(x);
}

static void destroy(t_neuralnet *x)
{
	destroy_dealloc(x);
	if (x->x_net_created) {
		free(x->x_layers);
	}
	init_object_variables(x);
	x->x_net_created = 0;
	x->x_net_trained = 0;
}

/* neuralnet_free is called on object deletion */
static void neuralnet_free(t_neuralnet *x)
{
	destroy_dealloc(x);
	if (x->x_net_created) {
		free(x->x_layers);
	}
	dealloc_transposed_input(x);
	dealloc_target(x, x->x_num_in_samples);
	dealloc_test_mem(x, x->x_num_in_samples);
	dealloc_train_mem(x, x->x_num_in_samples);
	clock_free(x->x_clock);
	clock_free(x->x_morph_clock);
}

/************************ morph between models ************************/

/* the following two functions are an adaptation of [line]'s code
   found in x_time.c from Pd's source code */
static void morph_tick(t_neuralnet *x)
{
	double timenow = clock_getlogicaltime();
	double msectogo = - clock_gettimesince(x->x_targettime);
	if (msectogo < 1E-9) {
		morph_copy_weights(x);
		dealloc_morph_mem(x);
	}
	else {
		morph_step(x, timenow);
		clock_delay(x->x_morph_clock,
				(x->x_grain > msectogo ? msectogo : x->x_grain));
	}
	forward_pass(x);
}

static void morph(t_neuralnet *x, t_symbol *s, t_float f)
{
	double timenow = clock_getlogicaltime();
	(void)(s); /* unused */	
	x->x_morph_ramp_dur = f;
	x->x_gotinlet = 1;
	if (x->x_gotinlet && x->x_morph_ramp_dur > 0.0) {
		alloc_morph_mem(x);
		if (timenow > x->x_targettime) morph_copy_set(x);
		else store_morph_step(x, timenow);
		x->x_prevtime = timenow;
		x->x_targettime = clock_getsystimeafter(x->x_morph_ramp_dur);
		x->x_morphing = 1;
		load(x, s);
		x->x_morphing = 0;
		morph_tick(x);
		x->x_gotinlet = 0;
		x->x_1overtimediff = 1. / (x->x_targettime - timenow);
		clock_delay(x->x_morph_clock,
				(x->x_grain > x->x_morph_ramp_dur ? x->x_morph_ramp_dur : x->x_grain));
	}
	else {
		clock_unset(x->x_morph_clock);
		morph_copy_weights(x);
		forward_pass(x);
	}
	x->x_gotinlet = 0;
}

static void morph_step(t_neuralnet *x, double timenow)
{
	int i, j, k;
	for (i = 0; i < x->x_num_layers; i++) {
		if (x->x_layers[i].type == 0) {
			for (j = 0; j < x->x_layers[i].dense.x_input_size; j++) {
				for (k = 0; k < x->x_layers[i].dense.x_output_size; k++) {
					x->x_layers[i].dense.x_weights[j][k] = x->x_layers[i].dense.x_set_weights[j][k] +
						x->x_1overtimediff * (timenow - x->x_prevtime) *
						(x->x_layers[i].dense.x_target_weights[j][k] -
						 x->x_layers[i].dense.x_set_weights[j][k]);
				}
			}
			for (j = 0; j < x->x_layers[i].dense.x_output_size; j++) {
				x->x_layers[i].dense.x_biases[j] = x->x_layers[i].dense.x_set_biases[j] +
					x->x_1overtimediff * (timenow - x->x_prevtime) *
					(x->x_layers[i].dense.x_target_biases[j] - x->x_layers[i].dense.x_set_biases[j]);
			}
		}
	}
	for (i = 0; i < x->x_input_size; i++) {
		x->x_max_in_vals[i] = x->x_set_max_in_vals[i] +
			x->x_1overtimediff * (timenow - x->x_prevtime) *
			(x->x_target_max_in_vals[i] - x->x_set_max_in_vals[i]);
	}
	for (i = 0; i < x->x_output_size; i++) {
		x->x_max_out_vals[i] = x->x_set_max_out_vals[i] +
			x->x_1overtimediff * (timenow - x->x_prevtime) *
			(x->x_target_max_out_vals[i] - x->x_set_max_out_vals[i]);
	}
}

static void store_morph_step(t_neuralnet *x, double timenow)
{
	int i, j, k;
	for (i = 0; i < x->x_num_layers; i++) {
		if (x->x_layers[i].type == 0) {
			for (j = 0; j < x->x_layers[i].dense.x_input_size; j++) {
				for (k = 0; k < x->x_layers[i].dense.x_output_size; k++) {
					x->x_layers[i].dense.x_set_weights[j][k] = x->x_layers[i].dense.x_set_weights[j][k] +
						x->x_1overtimediff * (timenow - x->x_prevtime) *
						(x->x_layers[i].dense.x_target_weights[j][k] -
						 x->x_layers[i].dense.x_set_weights[j][k]);
				}
			}
			for (j = 0; j < x->x_layers[i].dense.x_output_size; j++) {
				x->x_layers[i].dense.x_set_biases[j] = x->x_layers[i].dense.x_set_biases[j] +
					x->x_1overtimediff * (timenow - x->x_prevtime) *
					(x->x_layers[i].dense.x_target_biases[j] - x->x_layers[i].dense.x_set_biases[j]);
			}
		}
	}
	for (i = 0; i < x->x_input_size; i++) {
		x->x_set_max_in_vals[i] = x->x_set_max_in_vals[i] +
			x->x_1overtimediff * (timenow - x->x_prevtime) *
			(x->x_target_max_in_vals[i] - x->x_set_max_in_vals[i]);
	}
	for (i = 0; i < x->x_output_size; i++) {
		x->x_set_max_out_vals[i] = x->x_set_max_out_vals[i] +
			x->x_1overtimediff * (timenow - x->x_prevtime) *
			(x->x_target_max_out_vals[i] - x->x_set_max_out_vals[i]);
	}
}

static void morph_copy_set(t_neuralnet *x)
{
	int i, j, k;
	for (i = 0; i < x->x_num_layers; i++) {
		if (x->x_layers[i].type == 0) {
			for (j = 0; j < x->x_layers[i].dense.x_input_size; j++) {
				for (k = 0; k < x->x_layers[i].dense.x_output_size; k++) {
					x->x_layers[i].dense.x_set_weights[j][k] = x->x_layers[i].dense.x_weights[j][k];
				}
			}
			for (j = 0; j < x->x_layers[i].dense.x_output_size; j++) {
				x->x_layers[i].dense.x_set_biases[j] = x->x_layers[i].dense.x_biases[j];
			}
		}
	}
	for (i = 0; i < x->x_input_size; i++) {
		x->x_set_max_in_vals[i] = x->x_max_in_vals[i];
	}
	for (i = 0; i < x->x_output_size; i++) {
		x->x_set_max_out_vals[i] = x->x_max_out_vals[i];
	}
}

static void morph_copy_weights(t_neuralnet *x)
{
	int i, j, k;
	for (i = 0; i < x->x_num_layers; i++) {
		if (x->x_layers[i].type == 0) {
			for (j = 0; j < x->x_layers[i].dense.x_input_size; j++) {
				for (k = 0; k < x->x_layers[i].dense.x_output_size; k++) {
					x->x_layers[i].dense.x_weights[j][k] = x->x_layers[i].dense.x_target_weights[j][k];
				}
			}
			for (j = 0; j < x->x_layers[i].dense.x_output_size; j++) {
				x->x_layers[i].dense.x_biases[j] = x->x_layers[i].dense.x_target_biases[j];
			}
		}
	}
	for (i = 0; i < x->x_input_size; i++) {
		x->x_max_in_vals[i] = x->x_target_max_in_vals[i];
	}
	for (i = 0; i < x->x_output_size; i++) {
		x->x_max_out_vals[i] = x->x_target_max_out_vals[i];
	}
}

/********************* forward and backward passes ********************/

static void forward_pass(t_neuralnet *x)
{
	int i;
	int input_size = 1;
	t_float **prev_output = x->x_input;
	if (x->x_is_training || x->x_is_validating) input_size = x->x_num_in_samples;
	if (x->x_layers[0].type == 0) {
		/* make a forward pass from the input to the first hidden layer */
		layer_dense_forward(prev_output, &x->x_layers[0].dense, input_size, x->x_is_training, x->x_is_validating);
		activation_forward(x, input_size, 0);
	}
	/* the rest of the forward passes can be done inside a loop */
	for (i = 1; i < x->x_num_layers; i++) {
		if (x->x_layers[i].type == 0) {
			if (x->x_layers[i-1].type == 0) {
				if (x->x_is_training || x->x_is_validating) prev_output = x->x_layers[i-1].dense.x_act_output_train;
				else prev_output = x->x_layers[i-1].dense.x_act_output;
			}
			layer_dense_forward(prev_output, &x->x_layers[i].dense, input_size, x->x_is_training, x->x_is_validating);
			activation_forward(x, input_size, i);
		}
	}
	if (x->x_is_training || x->x_is_validating) {
		t_float cur_acc, cur_loss;
		x->x_loss = loss_forward(x);
		x->x_loss += regularization_loss(x);
		cur_loss = x->x_loss;
		if (x->x_layers[x->x_num_layers-1].type == 0) x->x_accuracy = get_accuracy(x);
		/* we're updating the best weights only if we're training
		   as we'll make a last forward pass when the training has finished
		   and we want to prove that we did choose the best weights */
		if (x->x_is_training) {
			cur_acc = max(x->x_accuracy, x->x_prev_accuracy);
			if (x->x_loss > 1e-04) {
				cur_loss = min(x->x_loss, x->x_prev_loss);
			}
			if (cur_acc != x->x_prev_accuracy || cur_loss != x->x_prev_loss) {
				x->x_prev_accuracy = x->x_accuracy;
				x->x_prev_loss = x->x_loss;
			}
		}
		outlet_float(x->x_loss_out, x->x_loss);
		outlet_float(x->x_acc_out, x->x_accuracy);
	}
}

static void back_propagate(t_neuralnet *x)
{
	int i, j, k;
	/* dummy assignment to prev_layer_output to avoid compiler warning */
	t_float **prev_layer_output = x->x_layers[x->x_num_layers-1].dense.x_act_output_transposed;
	t_float **dinput = x->x_loss_dinput;
	loss_backward(x);
	/* the first backward pass for the activation functions
	   and the layers is done separately as the input to the
	   activation function of the last layer (the first for the
	   back propagation) is the output derivative of the loss function */
	if (x->x_layers[x->x_num_layers-1].type == 0) {
		activation_backward(x, x->x_loss_dinput, x->x_num_layers-1);
		switch (x->x_layers[x->x_num_layers-2].type) {
			case 0:
				prev_layer_output = x->x_layers[x->x_num_layers-2].dense.x_act_output_transposed;
				break;
		}
		layer_dense_backward(&x->x_layers[x->x_num_layers-1].dense, prev_layer_output,
			x->x_layers[x->x_num_layers-1].x_weight_regularizer_l1,
			x->x_layers[x->x_num_layers-1].x_weight_regularizer_l2,
			x->x_layers[x->x_num_layers-1].x_bias_regularizer_l1,
			x->x_layers[x->x_num_layers-1].x_bias_regularizer_l2,
			x->x_num_in_samples);
	}
	/* the rest of the back propagation can be done inside a loop */
	for (i = x->x_num_layers - 2; i > 0; i--) {
		if (x->x_layers[i].type == 0) {
			switch (x->x_layers[i+1].type) {
				case 0:
					dinput = x->x_layers[i+1].dense.x_dinput;
					break;
			}
			activation_backward(x, dinput, i);
			switch (x->x_layers[i-1].type) {
				case 0:
					prev_layer_output = x->x_layers[i-1].dense.x_act_output_transposed;
					break;
			}
			layer_dense_backward(&x->x_layers[i].dense, prev_layer_output,
				x->x_layers[i].x_weight_regularizer_l1,
				x->x_layers[i].x_weight_regularizer_l2,
				x->x_layers[i].x_bias_regularizer_l1,
				x->x_layers[i].x_bias_regularizer_l2,
				x->x_num_in_samples);
		}
	}
	/* the last backward pass is done outside the loop because
	   the first layer (the last for the back propagation) needs the
	   transposed input of the network instead of the transposed output
	   of the previous activation function */
	if (x->x_layers[0].type == 0) {
		switch (x->x_layers[1].type) {
			case 0:
				dinput = x->x_layers[1].dense.x_dinput;
				break;
		}
		activation_backward(x, dinput, 0);
		layer_dense_backward(&x->x_layers[0].dense, x->x_input_transposed,
			x->x_layers[0].x_weight_regularizer_l1,
			x->x_layers[0].x_weight_regularizer_l2,
			x->x_layers[0].x_bias_regularizer_l1,
			x->x_layers[0].x_bias_regularizer_l2,
			x->x_num_in_samples);
	}
	
	/* check if any layer has its weights or biases set to be written to an array during training */
	for (i = 0; i < x->x_num_layers; i++) {
		if (x->x_layers[i].type == 0) {
			for (j = 0; j < x->x_layers[i].dense.x_input_size; j++) {
				if (x->x_layers[i].dense.x_store_weights_during_training[j]) {
					int array_size;
					t_garray *a;
					t_word *vec;
					if (!(a = (t_garray *)pd_findbyclass(x->x_layers[i].dense.x_weights_arrays[j], garray_class)))
						pd_error(x, "%s: no such array", x->x_layers[i].dense.x_weights_arrays[j]->s_name);
					else if (!garray_getfloatwords(a, &array_size, &vec))
						pd_error(x, "%s: bad template for neuralnet", x->x_layers[i].dense.x_weights_arrays[j]->s_name);
					else if (array_size != x->x_layers[i].dense.x_output_size)
						pd_error(x, "%s: size not equal to number of outputs in layer", x->x_layers[i].dense.x_weights_arrays[j]->s_name);
					else {
						for (k = 0; k < array_size; k++) {
							vec[k].w_float = x->x_layers[i].dense.x_weights[i][k] * x->x_layers[i].dense.x_weights_scale;
						}
						/* update the graphic array */
						garray_redraw(a);
					}
				}
			}
			if (x->x_layers[i].dense.x_store_biases_during_training) {
				int array_size;
				t_garray *a;
				t_word *vec;
				if (!(a = (t_garray *)pd_findbyclass(x->x_layers[i].dense.x_biases_array, garray_class)))
					pd_error(x, "%s: no such array", x->x_layers[i].dense.x_biases_array->s_name);
				else if (!garray_getfloatwords(a, &array_size, &vec))
					pd_error(x, "%s: bad template for neuralnet", x->x_layers[i].dense.x_biases_array->s_name);
				else if (array_size != x->x_layers[i].dense.x_output_size)
					pd_error(x, "%s: size not equal to number of outputs in layer", x->x_layers[i].dense.x_biases_array->s_name);
				else {
					for (j = 0; j < array_size; j++) {
						vec[j].w_float = x->x_layers[i].dense.x_biases[j] * x->x_layers[i].dense.x_biases_scale;
					}
					/* update the graphic array */
					garray_redraw(a);
				}
			}
		}
	}
}

/************ training, validating and predicting functions *************/

static void train_net(t_neuralnet *x)
{
	int i, j;
	/* use the if test below instead of a for loop
	   so we can use a clock with a delay between each epoch
	   and not clog Pd */
	if (x->x_epoch_count < x->x_epochs) {
		if (x->x_batch_count < x->x_batch_steps) {
			forward_pass(x);
			back_propagate(x);
			optimizer_pre_update(x);
			for (i = 0; i < x->x_num_layers; i++) {
				if (x->x_layers[i].type == 0) x->x_optimizer_funcs[x->x_optimizer_index](x, i);
			}
			optimizer_post_update(x);
			x->x_batch_count++;
			if (x->x_batch_size > 0) {
				/* it's very likely that the last batch will have
				   a training set that's not entirely full,
				   so we have to update x->x_num_in_samples
				   otherwise the loops in the forward and backward steps
				   will go past the number of samples and Pd will crash */
				if (x->x_batch_count == (x->x_batch_steps-1)) {
					x->x_num_in_samples = (x->x_old_num_in_samples - x->x_percentage) - \
										  (x->x_batch_count * x->x_batch_size);
				}
				/* because the incrementing of x->x_batch_count happens above
				   we must check if we're still within bounds */
				if (x->x_batch_count < x->x_batch_steps) {
					/* copy the next batch slice to the arrays */
					for (i = 0; i < x->x_num_in_samples; i++) {
						for (j = 0; j < x->x_input_size; j++) {
							x->x_input[i][j] = x->x_batch_input[i+(x->x_batch_count*x->x_batch_size)][j];
							x->x_input_transposed[j][i] = x->x_batch_transposed[j][i+(x->x_batch_count*x->x_batch_size)];
						}
						for (j = 0; j < x->x_output_size; j++) {
							x->x_target_vals[i][j] = x->x_batch_target[i+(x->x_batch_count*x->x_batch_size)][j];
						}
					}
				}
				outlet_float(x->x_batch_step_out, (t_float)x->x_batch_count);
			}
		}
		else if (x->x_batch_count >= x->x_batch_steps) {
			if (x->x_save_every_x_epochs) {
				if ((x->x_epoch_count % x->x_num_epochs_to_save) == 0 && \
					x->x_epoch_count >= x->x_start_saving_after_epoch) {
					save_during_training(x);
				}
			}
			x->x_batch_count = 0;
			x->x_epoch_count++;
			x->x_epoch_old++;
			if (x->x_batch_size > 0) {
				/* pause so the next training data can be loaded */
				/*pause_training(x);*/
				x->x_num_in_samples = x->x_batch_size;
			}
			outlet_float(x->x_epoch_out, (t_float)x->x_epoch_count);
		}
		/* small delay to avoid freezing Pd */
		clock_delay(x->x_clock, x->x_train_del);
	}
	else {
		int dealloc_size = x->x_num_in_samples;
		/* desired_loss and desired_accuracy make use of training pause
		   to not free the memory, though the epoch count doesn't need to
		   be changed as it will have reached the end */
		if (x->x_desired_loss < FLT_MAX) {
			if (x->x_loss > x->x_desired_loss) {
				x->x_is_paused = 1;
			}
		}
		if (x->x_desired_accuracy > 0.0) {
			if (x->x_accuracy < x->x_desired_accuracy) {
				x->x_is_paused = 1;
			}
		}
		/* to pause training we max the epoch_count variable
		   which will result in the else chuck we're now in being called
		   but we don't want to free the memory because we
		   haven't aborted training */
		if (!x->x_is_paused) {
			x->x_epoch_count = 0;
			x->x_epoch_old = 0;
			x->x_is_training = 0;
			x->x_save_every_x_epochs = 0;
			x->x_save_every_x_epochs_counter = 0;
			//restore_weights_and_biases(x);
			if (x->x_batch_size > 0) {
				dealloc_size = x->x_batch_size;
			}
			if (x->x_percentage == 0) {
				/* deallocate the main input, its transposed version
				   and the target array only if we're not keeping any data
				   from the training set for testing */
				dealloc_main_input(x, dealloc_size);
				dealloc_target(x, dealloc_size);
				/* in case of dropout, first deallocate the memory */
				for (i = 0; i < x->x_num_layers; i++) {
					if (x->x_layers[i].type == 0) {
						if (x->x_layers[i].dense.x_dropout_rate > 0.0) {
							dealloc_dropout_mem(&x->x_layers[i].dense, dealloc_size);
						}
					}
				}
				/* then reset x_num_in_samples */
				x->x_num_in_samples = x->x_old_num_in_samples;
				/* in case we trained in batches */
				if (x->x_batch_size > 0) {
					dealloc_batch_mem(x);
					x->x_batch_size = 0;
				}
			}
			else {
				/* in case of dropout, first deallocate the memory */
				for (i = 0; i < x->x_num_layers; i++) {
					if (x->x_layers[i].type == 0) {
						if (x->x_layers[i].dense.x_dropout_rate > 0.0) {
							dealloc_dropout_mem(&x->x_layers[i].dense, dealloc_size);
						}
					}
				}
				if (x->x_batch_size == 0) {
					/* then restore the original size for correct deallocation */
					x->x_num_in_samples += x->x_percentage;
					dealloc_size = x->x_num_in_samples;
				}
			}
			dealloc_transposed_input(x);
			dealloc_test_mem(x, dealloc_size);
			dealloc_train_mem(x, dealloc_size);
			/* reset variables for the forward pass and memory allocation.
			   Only if we're not using part of the training data for testing
			   should we allocate the main input array, as in case we're using
			   part of the data, this array has not been deallocated yet */
			if (x->x_percentage == 0) {
				x->x_num_in_samples = 0;
				x->x_old_num_in_samples = 0;
				alloc_main_input(x, 1, 0);
			}
			x->x_net_trained = 1;
		}
	}
}

static void train(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv)
{
	int i;
	(void)(s); /* unused */	
	if (!x->x_net_created) {
		pd_error(x, "network has not been created");
		return;
	}
	x->x_is_training = 1;
	x->x_is_validating = 0;
	x->x_is_predicting = 0;
	/* test_mem is also used in training so we allocate both */
	alloc_test_mem(x);
	alloc_train_mem(x);
	/* check if we must normalize input and output
	   in case these values exceed the range between -1 and 1 */
	if (x->x_must_normalize_input) {
		norm_input(x);
	}
	/* normalize only if we're not outputting class indexes */
	if (x->x_must_normalize_output &&
			!(x->x_layers[x->x_num_layers-1].x_activation_index == SOFTMAX_INDEX))
	{
		norm_output(x);
	}
	/* set the accuracy precision before starting to train */
	set_accuracy_precision(x);
	/* if we want to keep some of the data for testing
	   we pass this as an argument to the "train" message */
	if (argc > 0) {
		int percentage;
		if (argc > 1) {
			pd_error(x, "train takes up to one argument only");
			return;
		}
		if (argv->a_type != A_FLOAT) {
			pd_error(x, "argument must be float");
			return;
		}
		if (x->x_batch_size > 0) {
			pd_error(x, "when training in batches, the percentage of the data percentage for validating is set in the set_batch_size message");
			return;
		}
		percentage = (int)argv->a_w.w_float;
		if (percentage > 0) {
			set_percentage(x, percentage);
			x->x_num_in_samples -= x->x_percentage;
		}
		else {
			pd_error(x, "data percentage must be greater than 0");
			return;
		}
	}
	/* check if a layer uses the softmax activation function
	   because we'll need to populate the eye matrix first */
	for (i = 0; i < x->x_num_layers; i++) {
		if (x->x_layers[i].type == 0) {
			if (x->x_layers[i].dense.x_allocate_softmax) {
				populate_eye(&x->x_layers[i].dense);
			}
			if (x->x_layers[i].dense.x_dropout_rate > 0.0) {
				int err = alloc_dropout_mem(&x->x_layers[i].dense, x->x_num_in_samples);
				if (err > 0) {
					switch (err) {
						case 1:
							pd_error(x, "dropout output mem error");
							break;
						case 2:
							pd_error(x, "dropout binary mask mem error");
							break;
						default:
							break;
					}
					return;
				}
			}
		}
	}
	train_net(x);
}

/* this is called if we want to train the network from scratch
   without providing a new training dataset */
static void retrain(t_neuralnet *x)
{
	if (x->x_train_mem_allocated) {
		x->x_epoch_count = 0;
		x->x_batch_count = 0;
		x->x_is_paused = 0;
		/* reset the weights before restarting training */
		set_weight_coeff(x, x->x_weight_coeff);
		init_biases(x);
		train_net(x);
	}
	else {
		pd_error(x, "training data has not been kept in memory, cannot retrain");
	}
}

/* this is called if we want to traing the network more
   without providing a new training dataset */
static void keep_training(t_neuralnet *x)
{
	if (x->x_train_mem_allocated) {
		x->x_epoch_count = 0;
		x->x_batch_count = 0;
		x->x_is_paused = 0;
		train_net(x);
	}
	else {
		pd_error(x, "training data has not been kept in memory, cannot keep training");
	}
}

static void release_mem(t_neuralnet *x)
{
	if (x->x_train_mem_allocated) {
		x->x_is_paused = 0;
		x->x_desired_loss = FLT_MAX;
		x->x_desired_accuracy = 0.0;
		train_net(x);
	}
	else {
		pd_error(x, "training data has not been kept in memory, cannot release memory");
	}
}

static void set_percentage(t_neuralnet *x, int percentage)
{
	x->x_percentage = (percentage * x->x_num_in_samples) / 100;
}

static void set_train_del(t_neuralnet *x, t_float f)
{
	if (f < 0.0) {
		pd_error(x, "delay time must be positive");
		return;
	}
	x->x_train_del = f;
}

static void pause_training(t_neuralnet *x)
{
	x->x_epoch_count = x->x_epochs;
	x->x_is_paused = 1;
}

static void resume_training(t_neuralnet *x)
{
	x->x_epoch_count = x->x_epoch_old;
	x->x_is_paused = 0;
	/* to protect from crashing in case "resume_training" is called
	   after an "abort_training" message", check if the training
	   memory is still allocated */
	if (x->x_train_mem_allocated) {
		train_net(x);
	}
}

static void abort_training(t_neuralnet *x)
{
	x->x_epoch_count = x->x_epochs;
}

static void set_batch_size(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv)
{
	int size = 0;
	int percentage;
	(void)(s); /* unused */	
	if (argc > 0) {
		size = (int)atom_getfloat(argv);
		if (argc > 2) {
			pd_error(x, "set_batch_size takes up to two arguments");
			return;
		}
		else if (argc == 2) {
			percentage = (int)atom_getfloat(argv+1);
			if (percentage > 0) {
				/* we don't update x->x_num_in_samples because it is
				   being updated at the end of this function based on the batch size */
				set_percentage(x, percentage);
			}
			else {
				pd_error(x, "data percentage must be greater than 0");
				return;
			}
		}
	}
	else {
		pd_error(x, "set_batch_size takes at least one argument");
	}
	if (x->x_num_in_samples == 0) {
		pd_error(x, "the training dataset must be imported before setting the batch size");
		return;
	}
	x->x_batch_size = size;
	x->x_batch_steps = x->x_num_in_samples / x->x_batch_size;
	/* in case there are some remaining samples we add one more step */
	if ((x->x_batch_steps * x->x_batch_size) < x->x_num_in_samples) {
		x->x_batch_steps++;
	}
	/* need to allocate new arrays to hold all the samples
	   then reallocate the input, transposed and target arrays to fit one batch
	   and make the appropriate copies of a batch slice each time */
	alloc_batch_mem(x);
	/* update x_num_in_samples to fit one batch only */
	x->x_num_in_samples = x->x_batch_size;
	post("batch size is set to %d with %d batch steps", (int)x->x_batch_size, x->x_batch_steps);
}

static void set_epochs(t_neuralnet *x, t_float epochs)
{
	if (epochs < 0.0) {
		pd_error(x, "epochs must be positive");
		return;
	}
	x->x_epochs = (int)epochs;
}

static void validate(t_neuralnet *x)
{
	int i, j;
	int dealloc_size = x->x_old_num_in_samples;
	t_float **input;
	t_float **target;
	int remaining;
	x->x_is_validating = 1;
	x->x_is_training = 0;
	x->x_is_predicting = 0;
	/* check if we're testing with a percentage of the training data */
	if (x->x_percentage > 0) {
		if (x->x_batch_size == 0) {
			input = x->x_input;
			target = x->x_target_vals;
			/* in this case we need to move the last values to the beginning */
			remaining = x->x_num_in_samples - x->x_percentage;
		}
		else {
			input = x->x_batch_input;
			target = x->x_batch_target;
			remaining = x->x_old_num_in_samples - x->x_percentage;
			alloc_main_input(x, x->x_percentage, x->x_batch_size);
			alloc_target(x, x->x_percentage, x->x_batch_size);
			dealloc_size = x->x_percentage;
		}
		/* update the number of input samples first */
		x->x_num_in_samples = x->x_percentage;
		for (i = 0; i < x->x_percentage; i++) {
			for (j = 0; j < x->x_input_size; j++) {
				x->x_input[i][j] = input[i+remaining][j];
			}
			for (j = 0; j < x->x_output_size; j++) {
				x->x_target_vals[i][j] = target[i+remaining][j];
			}
		}
	}
	/* now we're allocating the test memory only
	   with the update number of input samples */
	alloc_test_mem(x);
	forward_pass(x);
	x->x_is_validating = 0;
	/* first deallocate the test memory */
	dealloc_test_mem(x, x->x_num_in_samples);
	/* and deallocate the input array and its transposed version
	   because these have been allocated with the initial sample size
	   or the batch size, in case of batch training */
	dealloc_main_input(x, dealloc_size);
	dealloc_target(x, dealloc_size);
	if (x->x_batch_size > 0) {
		dealloc_batch_mem(x);
	}
	/* reset variables */
	x->x_num_in_samples = 0;
	x->x_old_num_in_samples = 0;
	x->x_percentage = 0;
	x->x_batch_size = 0;
	alloc_main_input(x, 1, 0);
}

static void predict(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv)
{
	t_garray *a;
	int i;
	t_word *vec;
	int array_size;
	(void)(s); /* unused */	
	if (x->x_net_trained) {
		if (argc > 0) {
			if (argc != x->x_input_size) {
				pd_error(x, "number of inputs is not equal to number of neurons in first layer");
				return;
			}
			for (i = 0; i < argc; i++) {
				if (argv->a_type == A_FLOAT) {
					x->x_input[0][i] = argv->a_w.w_float;
					x->x_input[0][i] /= x->x_max_in_vals[i];
					argv++;
				}
				else {
					pd_error(x, "predict method takes floats only");
					return;
				}
			}
		}
		/* if there are no arguments then we read from an array */
		else {
			if (!x->x_pred_from_added) {
				pd_error(x, "no predict input array added");
				return;
			}
			if (!(a = (t_garray *)pd_findbyclass(x->x_predict_from, garray_class))) {
				pd_error(x, "%s: no such array", x->x_predict_from->s_name);
			}
			else if (!garray_getfloatwords(a, &array_size, &vec)) {
				pd_error(x, "%s: bad template for neuralnet", x->x_predict_from->s_name);
			}
			else {
				if (array_size != x->x_input_size) {
					pd_error(x, "array size is not equal to number of neurons in first layer");
					return;
				}
				for (i = 0; i < array_size; i++) {
					x->x_input[0][i] = vec[i].w_float;
					x->x_input[0][i] /= x->x_max_in_vals[i];
				}
			}
		}
		x->x_is_predicting = 1;
		forward_pass(x);
	}
	else {
		post("neural network hasn't been trained");
	}
}

static void predict_from(t_neuralnet *x, t_symbol *s)
{
	const char from_inlet[] = "inlet";
	if (strcmp(s->s_name, from_inlet) == 0) {
		x->x_pred_from_added = 0;
	}
	else {
		x->x_predict_from = s;
		x->x_pred_from_added = 1;
	}
}

static void predict_to(t_neuralnet *x, t_symbol *s)
{
	const char to_outlet[] = "outlet";
	if (strcmp(s->s_name, to_outlet) == 0) {
		x->x_pred_to_added = 0;
	}
	else {
		x->x_predict_to = s;
		x->x_pred_to_added = 1;
	}
}

static void output_during_training(t_neuralnet *x, t_float f)
{
	x->x_output_during_training = (int)f;
}

static void store_weights_during_training(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv)
{
	int array_size;
	int layer_nr;
	int input_nr;
	t_symbol *array_name;
	t_garray *a;
	t_word *vec;
	(void)(s); /* unused */	
	if (argc != 3) {
		pd_error(x, "store_weights_during_training takes 3 arguments");
		return;
	}
	if (argv[0].a_type != A_FLOAT || argv[1].a_type != A_FLOAT || argv[2].a_type != A_SYMBOL) {
		pd_error(x, "first two arguments to store_weights_during_training must be floats and third must be a symbol");
		return;
	}
	layer_nr = atom_getfloat(argv++);
	input_nr = atom_getfloat(argv++);
	array_name = atom_gensym(argv);
	if (layer_nr >= x->x_num_layers) {
		pd_error(x, "layer number greater than number of layers");
		return;
	}
	if (x->x_layers[layer_nr].type != 0) {
		pd_error(x, "storing weights is only possible for dense layers");
		return;
	}
	if (input_nr >= x->x_layers[layer_nr].dense.x_input_size) {
		pd_error(x, "input number greater than number of inputs in layer");
		return;
	}
	if (!(a = (t_garray *)pd_findbyclass(array_name, garray_class))) {
		pd_error(x, "%s: no such array", array_name->s_name);
		return;
	}
	else if (!garray_getfloatwords(a, &array_size, &vec)) {
		pd_error(x, "%s: bad template for neuralnet", array_name->s_name);
		return;
	}
	else if (array_size != x->x_layers[layer_nr].dense.x_output_size) {
		pd_error(x, "%s: size not equal to number of outputs in layer", array_name->s_name);
		return;
	}
	x->x_layers[layer_nr].dense.x_weights_arrays[input_nr] = array_name;
	x->x_layers[layer_nr].dense.x_store_weights_during_training[input_nr] = 1;
}

static void store_biases_during_training(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv)
{
	int array_size;
	t_garray *a;
	t_word *vec;
	int layer_nr;
	t_symbol *array_name;
	(void)(s); /* unused */
	if (argc != 2) {
		pd_error(x, "store_biases_during_training takes 2 arguments");
		return;
	}
	if (argv[0].a_type != A_FLOAT || argv[1].a_type != A_SYMBOL) {
		pd_error(x, "first argument to store_biases_during_training must be a float and second must be a symbol");
		return;
	}
	layer_nr = atom_getfloat(argv++);
	array_name = atom_gensym(argv);
	if (layer_nr >= x->x_num_layers) {
		pd_error(x, "layer number greater than number of layers");
		return;
	}
	if (x->x_layers[layer_nr].type != 0) {
		pd_error(x, "storing biases is only possible for dense layers");
		return;
	}
	if (!(a = (t_garray *)pd_findbyclass(array_name, garray_class))) {
		pd_error(x, "%s: no such array", array_name->s_name);
		return;
	}
	else if (!garray_getfloatwords(a, &array_size, &vec)) {
		pd_error(x, "%s: bad template for neuralnet", array_name->s_name);
		return;
	}
	else if (array_size != x->x_layers[layer_nr].dense.x_output_size) {
		pd_error(x, "%s: size not equal to number of outputs in layer", array_name->s_name);
		return;
	}
	x->x_layers[layer_nr].dense.x_biases_array = array_name;
	x->x_layers[layer_nr].dense.x_store_biases_during_training = 1;
}

static void weights_scale_coeff(t_neuralnet *x, t_float layer_nr_f, t_float coeff)
{
	int layer_nr = (int)layer_nr_f;
	if (layer_nr > x->x_num_layers) {
		pd_error(x, "layer number greater than number of layers");
		return;
	}
	else if (layer_nr < 0) {
		pd_error(x, "layer number can't be negative");
		return;
	}
	if (x->x_layers[layer_nr].type != 0) {
		pd_error(x, "weights scale coefficient can be set only for dense layers");
		return;
	}
	x->x_layers[layer_nr].dense.x_weights_scale = coeff;
}

static void biases_scale_coeff(t_neuralnet *x, t_float layer_nr_f, t_float coeff)
{
	int layer_nr = (int)layer_nr_f;
	if (layer_nr > x->x_num_layers) {
		pd_error(x, "layer number greater than number of layers");
		return;
	}
	else if (layer_nr < 0) {
		pd_error(x, "layer number can't be negative");
		return;
	}
	if (x->x_layers[layer_nr].type != 0) {
		pd_error(x, "biases scale coefficient can be set only for dense layers");
		return;
	}
	x->x_layers[layer_nr].dense.x_biases_scale = coeff;
}

/************ memory allocation, deallocation, reallocation ***********/

/* x_old_num_in_samples is used in deallocation and it's
   updated after all allocations are done, as allocations are
   first checking if memory is already allocated, in which case
   it deallocates it first, based on the previous number of samples */
static void update_old_num_in_samples(t_neuralnet *x)
{
	x->x_old_num_in_samples = x->x_num_in_samples;
}

static void alloc_test_mem(t_neuralnet *x)
{
	int i, j;
	if (!x->x_test_mem_allocated) {
		for (i = 0; i < x->x_num_layers; i++) {
			if (x->x_layers[i].type == 0) {
				x->x_layers[i].dense.x_output_train = (t_float **)malloc(sizeof(t_float*) * x->x_num_in_samples);
				x->x_layers[i].dense.x_act_output_train = (t_float **)malloc(sizeof(t_float*) * x->x_num_in_samples);
				for (j = 0; j < x->x_num_in_samples; j++) {
					x->x_layers[i].dense.x_output_train[j] = (t_float *)malloc(sizeof(t_float) * x->x_layers[i].dense.x_output_size);
					if (x->x_layers[i].dense.x_output_train[j] == NULL) {
						pd_error(x, "output_train[%d][%d] mem error", i, j);
						return;
					}
					x->x_layers[i].dense.x_act_output_train[j] = (t_float *)malloc(sizeof(t_float) * x->x_layers[i].dense.x_output_size);
					if (x->x_layers[i].dense.x_act_output_train[j] == NULL) {
						pd_error(x, "act_output_train[%d][%d] mem error", i, j);
						return;
					}
				}
			}
		}
		x->x_test_mem_allocated = 1;
	}
}

static void dealloc_test_mem(t_neuralnet *x, int num_samples)
{
	int i, j;
	if (x->x_test_mem_allocated) {
		for (i = 0; i < x->x_num_layers; i++) {
			if (x->x_layers[i].type == 0) {
				for (j = 0; j < num_samples; j++) {
					if(x->x_layers[i].dense.x_output_train[j] != NULL)
						free(x->x_layers[i].dense.x_output_train[j]);
					if(x->x_layers[i].dense.x_act_output_train[j] != NULL)
						free(x->x_layers[i].dense.x_act_output_train[j]);
				}
				if (x->x_layers[i].dense.x_output_train != NULL)
					free(x->x_layers[i].dense.x_output_train);
				if (x->x_layers[i].dense.x_act_output_train != NULL)
					free(x->x_layers[i].dense.x_act_output_train);
			}
		}
	}
	x->x_test_mem_allocated = 0;
}

static void alloc_morph_mem(t_neuralnet *x)
{
	int i, j;
	if (!x->x_morph_mem_allocated) {
		dealloc_morph_mem(x);
	}
	for (i = 0; i < x->x_num_layers; i++) {
		if (x->x_layers[i].type == 0) {
			x->x_layers[i].dense.x_set_weights = (t_float **)malloc(sizeof(t_float*) * x->x_layers[i].dense.x_input_size);
			x->x_layers[i].dense.x_target_weights = (t_float **)malloc(sizeof(t_float*) * x->x_layers[i].dense.x_input_size);
			for (j = 0; j < x->x_layers[i].dense.x_input_size; j++) {
				x->x_layers[i].dense.x_set_weights[j] = (t_float *)malloc(sizeof(t_float) * x->x_layers[i].dense.x_output_size);
				if (x->x_layers[i].dense.x_set_weights[j] == NULL) {
					pd_error(x, "morph set weights mem error");
					return;
				}
				x->x_layers[i].dense.x_target_weights[j] = (t_float *)malloc(sizeof(t_float) * x->x_layers[i].dense.x_output_size);
				if (x->x_layers[i].dense.x_target_weights[j] == NULL) {
					pd_error(x, "morph target weights mem error");
					return;
				}
			}
			x->x_layers[i].dense.x_set_biases = (t_float *)malloc(sizeof(t_float) * x->x_layers[i].dense.x_output_size);
			if (x->x_layers[i].dense.x_set_biases == NULL) {
				pd_error(x, "morph set biases %d mem error", i);
				return;
			}
			x->x_layers[i].dense.x_target_biases = (t_float *)malloc(sizeof(t_float) * x->x_layers[i].dense.x_output_size);
			if (x->x_layers[i].dense.x_target_biases == NULL) {
				pd_error(x, "morph target biases %d mem error", i);
				return;
			}
		}
	}
	x->x_set_max_in_vals = (t_float *)malloc(sizeof(t_float) * x->x_input_size);
	if (x->x_set_max_in_vals == NULL) {
		pd_error(x, "set_max_in_vals mem error");
		return;
	}
	x->x_set_max_out_vals = (t_float *)malloc(sizeof(t_float) * x->x_output_size);
	if (x->x_set_max_out_vals == NULL) {
		pd_error(x, "set_max_out_vals mem error");
		return;
	}
	x->x_target_max_in_vals = (t_float *)malloc(sizeof(t_float) * x->x_input_size);
	if (x->x_target_max_in_vals == NULL) {
		pd_error(x, "target_max_in_vals mem error");
		return;
	}
	x->x_target_max_out_vals = (t_float *)malloc(sizeof(t_float) * x->x_output_size);
	if (x->x_target_max_out_vals == NULL) {
		pd_error(x, "target_max_out_vals mem error");
		return;
	}
	x->x_morph_mem_allocated = 1;
}

static void dealloc_morph_mem(t_neuralnet *x)
{
	int i, j;
	if (x->x_morph_mem_allocated) {
		for (i = 0; i < x->x_num_layers; i++) {
			if (x->x_layers[i].type == 0) {
				for (j = 0; j < x->x_layers[i].dense.x_input_size; j++) {
					if(x->x_layers[i].dense.x_set_weights[j] != NULL)
						free(x->x_layers[i].dense.x_set_weights[j]);
					if(x->x_layers[i].dense.x_target_weights[j] != NULL)
						free(x->x_layers[i].dense.x_target_weights[j]);
				}
				if (x->x_layers[i].dense.x_set_weights != NULL)
					free(x->x_layers[i].dense.x_set_weights);
				if (x->x_layers[i].dense.x_target_weights != NULL)
					free(x->x_layers[i].dense.x_target_weights);
				if (x->x_layers[i].dense.x_set_biases != NULL)
					free(x->x_layers[i].dense.x_set_biases);
				if (x->x_layers[i].dense.x_target_biases != NULL)
					free(x->x_layers[i].dense.x_target_biases);
			}
		}
		if (x->x_set_max_in_vals != NULL)
			free(x->x_set_max_in_vals);
		if (x->x_target_max_in_vals != NULL)
			free(x->x_target_max_in_vals);
		if (x->x_set_max_out_vals != NULL)
			free(x->x_set_max_out_vals);
		if (x->x_target_max_out_vals != NULL)
			free(x->x_target_max_out_vals);
		x->x_morph_mem_allocated = 0;
	}
}

static void alloc_train_mem(t_neuralnet *x)
{
	int i;
	int err = 0;
	int output_size = 0; /* assingment to 0 to avoid compiler warning */
	switch (x->x_layers[x->x_num_layers-1].type) {
		case 0:
			output_size = x->x_layers[x->x_num_layers-1].dense.x_output_size;
			break;
		default:
			break;
	}
	if (!x->x_train_mem_allocated) {
		for (i = 0; i < x->x_num_layers; i++) {
			if (x->x_layers[i].type == 0) {
				err = alloc_dense_train_mem(&x->x_layers[i].dense, x->x_num_in_samples);
				if (err > 0) {
					pd_error(x, "train memory error");
				}
			}
		}
		/* the derivative inputs of the loss function is
		   size_of_data*size_of_network_output */
		x->x_loss_dinput = (t_float **)malloc(sizeof(t_float*) * x->x_num_in_samples);
		for (i = 0; i < x->x_num_in_samples; i++) {
			x->x_loss_dinput[i] = (t_float *)malloc(sizeof(t_float) * output_size);
			if (x->x_loss_dinput == NULL) {
				pd_error(x, "loss_dinput[%d][%d] mem error", x->x_num_in_samples, i);
				return;
			}
		}
		x->x_train_mem_allocated = 1;
	}
}

static void dealloc_train_mem(t_neuralnet *x, int num_samples)
{
	int i;
	if (x->x_train_mem_allocated) {
		for (i = 0; i < x->x_num_layers; i++) {
			if (x->x_layers[i].type == 0) {
				dealloc_dense_train_mem(&x->x_layers[i].dense, num_samples);
			}
		}
		for (i = 0; i < num_samples; i++) {
			if(x->x_loss_dinput[i] != NULL)
				free(x->x_loss_dinput[i]);
		}
		if (x->x_loss_dinput != NULL)
			free(x->x_loss_dinput);
	}
	x->x_train_mem_allocated = 0;
}

static void dealloc_dropout_mem(t_layer_dense *layer, int num_samples)
{
	int i;
	if (layer->x_dropout_allocated) {
		for (i = 0; i < num_samples; i++) {
			free(layer->x_dropout_output[i]);
			free(layer->x_binary_mask[i]);
		}
		free(layer->x_dropout_output);
		free(layer->x_binary_mask);
		layer->x_dropout_allocated = 0;
	}
}

static void alloc_main_input(t_neuralnet *x, int num_samples, int old_num_samples)
{
	int i;
	if (x->x_main_input_allocated) {
		dealloc_main_input(x, old_num_samples);
	}
	x->x_input = (t_float **)malloc(sizeof(t_float*) * num_samples);
	for (i = 0; i < num_samples; i++) {
		x->x_input[i] = (t_float *)malloc(sizeof(t_float) * x->x_input_size);
		if (x->x_input[i] == NULL) {
			pd_error(x, "main input mem error");
			return;
		}
	}
	x->x_main_input_allocated = 1;
}

static void dealloc_main_input(t_neuralnet *x, int old_num_samples)
{
	int i;
	if (x->x_main_input_allocated) {
		for (i = 0; i < old_num_samples; i++) {
			if (x->x_input[i] != NULL)
				free(x->x_input[i]);
		}
		if (x->x_input != NULL)
			free(x->x_input);
	}
	x->x_main_input_allocated = 0;
}

static void realloc_main_input(t_neuralnet *x)
{
	/*t_float **temp;*/
	x->x_input = (t_float **)realloc(x->x_input, sizeof(t_float*) * x->x_num_in_samples);
	if (x->x_input == NULL) {
		pd_error(x, "dynamic mem error");
		return;
	}
	/*x->x_input = temp;*/
	x->x_input[x->x_num_in_samples-1] = (t_float *)malloc(sizeof(t_float) * x->x_input_size);
}

static void alloc_transposed_input(t_neuralnet *x)
{
	int i;
	if (x->x_transposed_allocated) {
		dealloc_transposed_input(x);
	}
	x->x_input_transposed = (t_float **)malloc(sizeof(t_float*) * x->x_input_size);
	for (i = 0; i < x->x_input_size; i++) {
		x->x_input_transposed[i] = (t_float *)malloc(sizeof(t_float) * x->x_num_in_samples);
		if (x->x_input_transposed[i] == NULL) {
			pd_error(x, "main transposed input mem error");
			return;
		}
	}
	x->x_transposed_allocated = 1;
}

static void dealloc_transposed_input(t_neuralnet *x)
{
	int i;
	if (x->x_transposed_allocated) {
		for (i = 0; i < x->x_input_size; i++) {
			if (x->x_input_transposed[i] != NULL)
				free(x->x_input_transposed[i]);
		}
		if (x->x_input_transposed != NULL)
			free(x->x_input_transposed);
	}
	x->x_transposed_allocated = 0;
}

static void realloc_transposed_input(t_neuralnet *x)
{
	int i;
	/*t_float *temp;*/
	for (i = 0; i < x->x_input_size; i++) {
		x->x_input_transposed[i] = (t_float *)realloc(x->x_input_transposed[i],
				sizeof(t_float) * x->x_num_in_samples);
		if (x->x_input_transposed[i] == NULL) {
			pd_error(x, "dynamic transposed mem error");
			return;
		}
		/*x->x_input_transposed[i] = temp;*/
	}
}

static void alloc_target(t_neuralnet *x, int num_samples, int old_num_samples)
{
	int i;
	if (x->x_target_vals_allocated) {
		dealloc_target(x, old_num_samples);
	}
	x->x_target_vals = (t_float **)malloc(sizeof(t_float*) * num_samples);
	for (i = 0; i < num_samples; i++) {
		x->x_target_vals[i] = (t_float *)malloc(sizeof(t_float) * x->x_output_size);
		if (x->x_target_vals[i] == NULL) {
			pd_error(x, "target values mem error");
			return;
		}
	}
	x->x_target_vals_allocated = 1;
}

static void dealloc_target(t_neuralnet *x, int old_num_samples)
{
	int i;
	if (x->x_target_vals_allocated) {
		for (i = 0; i < old_num_samples; i++) {
			if (x->x_target_vals[i] != NULL)
				free(x->x_target_vals[i]);
		}
		if (x->x_target_vals != NULL)
			free(x->x_target_vals);
	}
	x->x_target_vals_allocated = 0;
}

static void realloc_target(t_neuralnet *x)
{
	/*t_float **temp;*/
	x->x_target_vals = (t_float **)realloc(x->x_target_vals, sizeof(t_float*) * x->x_num_in_samples);
	if (x->x_target_vals == NULL) {
		pd_error(x, "dynamic mem error");
		return;
	}
	/*x->x_target_vals = temp;*/
	x->x_target_vals[x->x_num_in_samples-1] = (t_float *)malloc(sizeof(t_float) * x->x_output_size);
}

static void alloc_max_vals(t_neuralnet *x)
{
	int i;
	if (x->x_max_vals_allocated) {
		dealloc_max_vals(x);
	}
	x->x_max_in_vals = (t_float *)malloc(sizeof(t_float) * x->x_input_size);
	if (x->x_max_in_vals == NULL) {
		pd_error(x, "max_in_vals mem error");
		return;
	}
	for (i = 0; i < x->x_input_size; i++) {
		x->x_max_in_vals[i] = 1.0;
	}
	x->x_max_out_vals = (t_float *)malloc(sizeof(t_float) * x->x_output_size);
	if (x->x_max_out_vals == NULL) {
		pd_error(x, "max_out_vals mem error");
		return;
	}
	for (i = 0; i < x->x_output_size; i++) {
		x->x_max_out_vals[i] = 1.0;
	}
	x->x_max_vals_allocated = 1;
}

static void dealloc_max_vals(t_neuralnet *x)
{
	if (x->x_max_vals_allocated) {
		free(x->x_max_in_vals);
		free(x->x_max_out_vals);
	}
	x->x_max_vals_allocated = 0;
}

static void alloc_batch_mem(t_neuralnet *x)
{
	int i, j;
	x->x_batch_input = (t_float **)malloc(sizeof(t_float*) * x->x_num_in_samples);
	for (i = 0; i < x->x_num_in_samples; i++) {
		x->x_batch_input[i] = (t_float *)malloc(sizeof(t_float) * x->x_input_size);
		if (x->x_batch_input[i] == NULL) {
			pd_error(x, "batch input copy mem error");
			return;
		}
	}
	x->x_batch_transposed = (t_float **)malloc(sizeof(t_float*) * x->x_input_size);
	for (i = 0; i < x->x_input_size; i++) {
		x->x_batch_transposed[i] = (t_float *)malloc(sizeof(t_float) * x->x_num_in_samples);
		if (x->x_batch_transposed[i] == NULL) {
			pd_error(x, "batch transposed input copy mem error");
			return;
		}
	}
	x->x_batch_target = (t_float **)malloc(sizeof(t_float*) * x->x_num_in_samples);
	for (i = 0; i < x->x_num_in_samples; i++) {
		x->x_batch_target[i] = (t_float *)malloc(sizeof(t_float) * x->x_output_size);
		if (x->x_batch_target[i] == NULL) {
			pd_error(x, "batch target copy mem error");
			return;
		}
	}
	/* populate the batch training arrays */
	for (i = 0; i < x->x_num_in_samples; i++) {
		for (j = 0; j < x->x_input_size; j++) {
			x->x_batch_input[i][j] = x->x_input[i][j];
			x->x_batch_transposed[j][i] = x->x_input_transposed[j][i];
		}
		for (j = 0; j < x->x_output_size; j++) {
			x->x_batch_target[i][j] = x->x_target_vals[i][j];
		}
	}
	/* then reallocate the main arrays to fit one batch only */
	x->x_input = (t_float **)realloc(x->x_input, sizeof(t_float*) * x->x_batch_size);
	x->x_target_vals = (t_float **)realloc(x->x_target_vals, sizeof(t_float*) * x->x_batch_size);
	for (i = 0; i < x->x_input_size; i++) {
		x->x_input_transposed[i] = (t_float *)realloc(x->x_input_transposed[i],
				sizeof(t_float) * x->x_batch_size);
		if (x->x_input_transposed[i] == NULL) {
			pd_error(x, "dynamic transposed mem error");
			return;
		}
	}
}

static void dealloc_batch_mem(t_neuralnet *x)
{
	int i;
	for (i = 0; i < x->x_old_num_in_samples; i++) {
		if (x->x_batch_input[i] != NULL)
			free(x->x_batch_input[i]);
		if (x->x_batch_target[i] != NULL)
			free(x->x_batch_target[i]);
	}
	if (x->x_batch_input != NULL)
		free(x->x_batch_input);
	if (x->x_batch_target != NULL)
		free(x->x_batch_target);
	for (i = 0; i < x->x_input_size; i++) {
		if (x->x_batch_transposed[i] != NULL)
			free(x->x_batch_transposed[i]);
	}
	if (x->x_batch_transposed != NULL)
		free(x->x_batch_transposed);
}

/* funcitons for the dense layer */

/* free the memory allocated by a layer */
static void dealloc_layer(t_layer_dense *layer)
{
	int i;
	for (i = 0; i < layer->x_input_size; i++) {
		free(layer->x_weights[i]);
	}
	free(layer->x_weights);
	free(layer->x_weights_arrays);
	free(layer->x_store_weights_during_training);
	free(layer->x_output[0]);
	free(layer->x_output);
	free(layer->x_act_output[0]);
	free(layer->x_act_output);
	free(layer->x_biases);
}

static int alloc_dense_train_mem(t_layer_dense *layer, int num_samples)
{
	int i, j;
	layer->x_act_output_transposed = (t_float **)malloc(sizeof(t_float*) * layer->x_output_size);
	layer->x_dinput = (t_float **)malloc(sizeof(t_float*) * num_samples);
	layer->x_act_dinput = (t_float **)malloc(sizeof(t_float*) * num_samples);
	layer->x_dweights = (t_float **)malloc(sizeof(t_float*) * layer->x_input_size);
	layer->x_weights_copy = (t_float **)malloc(sizeof(t_float*) * layer->x_input_size);
	layer->x_dbiases = (t_float *)malloc(sizeof(t_float) * layer->x_output_size);
	if (layer->x_dbiases == NULL) {
		return 1;
	}
	layer->x_biases_copy = (t_float *)malloc(sizeof(t_float) * layer->x_output_size);
	if (layer->x_biases_copy == NULL) {
		return 1;
	}
	layer->x_weight_momentums = (t_float **)malloc(sizeof(t_float*) * layer->x_input_size);
	layer->x_weight_cache = (t_float **)malloc(sizeof(t_float*) * layer->x_input_size);
	layer->x_weight_momentums_corrected = (t_float **)malloc(sizeof(t_float*) * layer->x_input_size);
	layer->x_weight_cache_corrected = (t_float **)malloc(sizeof(t_float*) * layer->x_input_size);
	layer->x_weights_transposed = (t_float **)malloc(sizeof(t_float*) * layer->x_output_size);
	if (layer->x_allocate_softmax) {
		layer->x_eye = (t_float **)malloc(sizeof(t_float*) * layer->x_output_size);
		layer->x_dot_product = (t_float **)malloc(sizeof(t_float*) * layer->x_output_size);
		layer->x_jac_mtx = (t_float **)malloc(sizeof(t_float*) * layer->x_output_size);
	}
	layer->x_bias_momentums = (t_float *)malloc(sizeof(t_float) * layer->x_output_size);
	if (layer->x_bias_momentums == NULL) {
		return 1;
	}
	layer->x_bias_cache = (t_float *)malloc(sizeof(t_float) * layer->x_output_size);
	if (layer->x_bias_cache == NULL) {
		return 1;
	}
	layer->x_bias_momentums_corrected = (t_float *)malloc(sizeof(t_float) * layer->x_output_size);
	if (layer->x_bias_momentums_corrected == NULL) {
		return 1;
	}
	layer->x_bias_cache_corrected = (t_float *)malloc(sizeof(t_float) * layer->x_output_size);
	if (layer->x_bias_cache_corrected == NULL) {
		return 1;
	}
	for (i = 0; i < num_samples; i++) {
		layer->x_dinput[i] = (t_float *)malloc(sizeof(t_float) * layer->x_input_size);
		if (layer->x_dinput[i] == NULL) {
			return 1;
		}
		layer->x_act_dinput[i] = (t_float *)malloc(sizeof(t_float) * layer->x_output_size);
		if (layer->x_act_dinput[i] == NULL) {
			return 1;
		}
	}
	for (i = 0; i < layer->x_input_size; i++) {
		layer->x_weights_copy[i] = (t_float *)malloc(sizeof(t_float) * layer->x_output_size);
		if (layer->x_weights_copy[i] == NULL) {
			return 1;
		}
		layer->x_dweights[i] = (t_float *)malloc(sizeof(t_float) * layer->x_output_size);
		if (layer->x_dweights[i] == NULL) {
			return 1;
		}
		layer->x_weight_momentums[i] = (t_float *)malloc(sizeof(t_float) * layer->x_output_size);
		if (layer->x_weight_momentums[i] == NULL) {
			return 1;
		}
		layer->x_weight_cache[i] = (t_float *)malloc(sizeof(t_float) * layer->x_output_size);
		if (layer->x_weight_cache[i] == NULL) {
			return 1;
		}
		layer->x_weight_momentums_corrected[i] = (t_float *)malloc(sizeof(t_float) * layer->x_output_size);
		if (layer->x_weight_momentums_corrected[i] == NULL) {
			return 1;
		}
		layer->x_weight_cache_corrected[i] = (t_float *)malloc(sizeof(t_float) * layer->x_output_size);
		if (layer->x_weight_cache_corrected[i] == NULL) {
			return 1;
		}
		for (j = 0; j < layer->x_output_size; j++) {
			layer->x_weight_momentums[i][j] = 0.0;
			layer->x_weight_cache[i][j] = 0.0;
			layer->x_weight_momentums_corrected[i][j] = 0.0;
			layer->x_weight_cache_corrected[i][j] = 0.0;
		}
	}
	for (i = 0; i < layer->x_output_size; i++) {
		layer->x_act_output_transposed[i] = (t_float *)malloc(sizeof(t_float) * num_samples);
		if (layer->x_act_output_transposed[i] == NULL) {
			return 1;
		}
		layer->x_weights_transposed[i] = (t_float *)malloc(sizeof(t_float) * layer->x_input_size);
		if (layer->x_weights_transposed[i] == NULL) {
			return 1;
		}
		if (layer->x_allocate_softmax) {
			layer->x_eye[i] = (t_float *)malloc(sizeof(t_float) * layer->x_output_size);
			if (layer->x_eye[i] == NULL) {
				return 1;
			}
			layer->x_dot_product[i] = (t_float *)malloc(sizeof(t_float) * layer->x_output_size);
			if (layer->x_dot_product[i] == NULL) {
				return 1;
			}
			layer->x_jac_mtx[i] = (t_float *)malloc(sizeof(t_float) * layer->x_output_size);
			if (layer->x_jac_mtx[i] == NULL) {
				return 1;
			}
		}
		for (j = 0; j < layer->x_input_size; j++) {
			layer->x_weights_transposed[i][j] = layer->x_weights[j][i];
		}
		layer->x_bias_momentums[i] = 0.0;
		layer->x_bias_cache[i] = 0.0;
	}
	return 0;
}

static void dealloc_dense_train_mem(t_layer_dense *layer, int num_samples)
{
	int i;
	for (i = 0; i < layer->x_input_size; i++) {
		if(layer->x_weights_copy[i] != NULL)
			free(layer->x_weights_copy[i]);
		if(layer->x_dweights[i] != NULL)
			free(layer->x_dweights[i]);
		if(layer->x_weight_momentums[i] != NULL)
			free(layer->x_weight_momentums[i]);
		if(layer->x_weight_cache[i] != NULL)
			free(layer->x_weight_cache[i]);
		if(layer->x_weight_momentums_corrected[i] != NULL)
			free(layer->x_weight_momentums_corrected[i]);
		if(layer->x_weight_cache_corrected[i] != NULL)
			free(layer->x_weight_cache_corrected[i]);
	}
	if (layer->x_weights_copy != NULL)
		free(layer->x_weights_copy);
	if(layer->x_biases_copy != NULL)
		free(layer->x_biases_copy);
	if(layer->x_dbiases != NULL)
		free(layer->x_dbiases);
	if(layer->x_bias_momentums != NULL)
		free(layer->x_bias_momentums);
	if(layer->x_bias_cache != NULL)
		free(layer->x_bias_cache);
	if(layer->x_bias_momentums_corrected != NULL)
		free(layer->x_bias_momentums_corrected);
	if(layer->x_bias_cache_corrected != NULL)
		free(layer->x_bias_cache_corrected);
	for (i = 0; i < num_samples; i++) {
		if(layer->x_dinput[i] != NULL)
			free(layer->x_dinput[i]);
		if(layer->x_act_dinput[i] != NULL)
			free(layer->x_act_dinput[i]);
	}
	for (i = 0; i < layer->x_output_size; i++) {
		if(layer->x_act_output_transposed[i] != NULL)
			free(layer->x_act_output_transposed[i]);
		if(layer->x_weights_transposed[i] != NULL)
			free(layer->x_weights_transposed[i]);
		if (layer->x_allocate_softmax) {
			if (layer->x_eye[i] != NULL)
				free(layer->x_eye[i]);
			if (layer->x_dot_product[i] != NULL)
				free(layer->x_dot_product[i]);
			if (layer->x_jac_mtx[i] != NULL)
				free(layer->x_jac_mtx[i]);
		}
	}
	if (layer->x_act_output_transposed != NULL)
		free(layer->x_act_output_transposed);
	if (layer->x_dinput != NULL)
		free(layer->x_dinput);
	if (layer->x_act_dinput != NULL)
		free(layer->x_act_dinput);
	if (layer->x_allocate_softmax) {
		if (layer->x_eye != NULL)
			free(layer->x_eye);
		if (layer->x_dot_product != NULL)
			free(layer->x_dot_product);
		if (layer->x_jac_mtx != NULL)
			free(layer->x_jac_mtx);
		layer->x_allocate_softmax = 0;
	}
	if (layer->x_dweights != NULL)
		free(layer->x_dweights);
	if (layer->x_weight_momentums != NULL)
		free(layer->x_weight_momentums);
	if (layer->x_weight_cache != NULL)
		free(layer->x_weight_cache);
	if (layer->x_weight_momentums_corrected != NULL)
		free(layer->x_weight_momentums_corrected);
	if (layer->x_weight_cache_corrected != NULL)
		free(layer->x_weight_cache_corrected);
	if (layer->x_weights_transposed != NULL)
		free(layer->x_weights_transposed);
}

static int alloc_dropout_mem(t_layer_dense *layer, int num_samples)
{
	int i;
	if (!layer->x_dropout_allocated) {
		layer->x_dropout_output = (t_float **)malloc(sizeof(t_float*) * num_samples);
		layer->x_binary_mask = (t_float **)malloc(sizeof(t_float*) * num_samples);
		for (i = 0; i < num_samples; i++) {
			layer->x_dropout_output[i] = (t_float *)malloc(sizeof(t_float) * layer->x_input_size);
			if (layer->x_dropout_output[i] == NULL) {
				return 1;
			}
		}
		for (i = 0; i < num_samples; i++) {
			layer->x_binary_mask[i] = (t_float *)malloc(sizeof(t_float) * layer->x_input_size);
			if (layer->x_binary_mask[i] == NULL) {
				return 2;
			}
		}
		layer->x_dropout_allocated = 1;
	}
	return 0;
}

/******************** data normalization functions *******************/

/* the following four functions are used when input exceeds the -1 to 1 range
   The first two are called internally and the last two with the respective messages*/
static void norm_input(t_neuralnet *x)
{
	int i, j;
	for (i = 0; i < x->x_num_in_samples; i++) {
		for (j = 0; j < x->x_input_size; j++) {
			x->x_input[i][j] /= x->x_max_in_vals[j];
			x->x_input_transposed[j][i] = x->x_input[i][j];
		}
	}
	x->x_must_normalize_input = 0;
}

static void norm_output(t_neuralnet *x)
{
	int i, j;
	for (i = 0; i < x->x_num_in_samples; i++) {
		for (j = 0; j < x->x_output_size; j++) {
			x->x_target_vals[i][j] /= x->x_max_out_vals[j];
		}
	}
	x->x_must_normalize_output = 0;
}

static void normalize_input(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv)
{
	int i;
	t_atom *argv_local = argv;
	(void)(s); /* unused */
	if (argc > 0) {
		if (argc != x->x_input_size) {
			pd_error(x, "%d arguments for %d inputs", argc, x->x_input_size);
			return;
		}
		for (i = 0; i < argc; i++) {
			x->x_max_in_vals[i] = argv_local->a_w.w_float;
			argv_local++;
		}
	}
	norm_input(x);
}

static void normalize_output(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv)
{
	int i;
	(void)(s); /* unused */
	if (argc > 0) {
		if (argc != x->x_output_size) {
			pd_error(x, "arguments must as many as number of network inputs");
			return;
		}
		for (i = 0; i < argc; i++) {
			x->x_max_out_vals[i] = argv->a_w.w_float;
			argv++;
		}
	}
	norm_output(x);
}

/**************** manual data importing and shuffling ******************/

static void add(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv)
{
	int is_one_hot = 0;
	int vert_arrays = 0;
	(void)(s); /* unused */
	/* if we receive no arguments, it means we'll be adding arrays vertically */
	if (argc == 0) {
		if (!x->x_arrays_ver_added) {
			pd_error(x, "no arrays have been added");
			return;
		}
		vert_arrays = 1;
	}
	else {
		if (argc != (x->x_input_size + x->x_output_size)) {
			if (x->x_classification && argc == (x->x_input_size+1)) {
				is_one_hot = 1;
			}
			else {
				pd_error(x, "incorrect argument number: %d\nexpected %d or %d", argc, \
						(x->x_input_size + x->x_output_size), (x->x_input_size + 1));
				return;
			}
		}
	}
	x->x_num_in_samples++;
	if (x->x_num_in_samples == 1) {
		/* main input has been allocated on the creation of the network */
		alloc_transposed_input(x);
		alloc_target(x, x->x_num_in_samples, x->x_old_num_in_samples);
	}
	else {
		realloc_main_input(x);
		realloc_transposed_input(x);
		realloc_target(x);
	}
	/* update the x->x_old_num_in_samples variable
	   as we have already done the necessary deallocations */
	update_old_num_in_samples(x);
	if (vert_arrays) {
		get_data_in_arrays_ver(x);
		get_data_out_arrays_ver(x);
	}
	else {
		get_list_data(x, argv, is_one_hot);
	}
}

static void get_list_data(t_neuralnet *x, t_atom *argv, int is_one_hot)
{
	int i;
	/* store the current input to the last indexes of the arrays
	   and test whether normalizing is necessary */
	for (i = 0; i < x->x_input_size; i++) {
		x->x_input[x->x_num_in_samples-1][i] = atom_getfloat(argv+i);
		if (x->x_input[x->x_num_in_samples-1][i] > 1.0 || x->x_input[x->x_num_in_samples-1][i] < -1.0) {
			if (fabs(x->x_input[x->x_num_in_samples-1][i]) > x->x_max_in_vals[i]) {
				x->x_max_in_vals[i] = fabs(x->x_input[x->x_num_in_samples-1][i]);
			}
			x->x_must_normalize_input = 1;
		}
		x->x_input_transposed[i][x->x_num_in_samples-1] = x->x_input[x->x_num_in_samples-1][i];
	}
	if (is_one_hot) {
		int col;
		/* convert labelled data to one-hot encoded */
		for (i = 0; i < x->x_output_size; i++) {
			x->x_target_vals[x->x_num_in_samples-1][i] = 0.0;
		}
		col = (int)atom_getfloat(argv+x->x_input_size);
		if (col < 0) {
			pd_error(x, "class index can't be negative");
			/* remove one sample */
			x->x_num_in_samples--;
			if (x->x_num_in_samples < 0) x->x_num_in_samples = 0;
			return;
		}
		x->x_target_vals[x->x_num_in_samples-1][col] = 1.0;
	}
	else {
		for (i = 0; i < x->x_output_size; i++) {
			x->x_target_vals[x->x_num_in_samples-1][i] = atom_getfloat(argv+i+x->x_input_size);
			if (x->x_target_vals[x->x_num_in_samples-1][i] > 1.0 || \
					x->x_target_vals[x->x_num_in_samples-1][i] < -1.0) {
				if (fabs(x->x_target_vals[x->x_num_in_samples-1][i]) > x->x_max_out_vals[i]) {
					x->x_max_out_vals[i] = fabs(x->x_target_vals[x->x_num_in_samples-1][i]);
				}
				x->x_must_normalize_output = 1;
			}
		}
	}
}

static void add_arrays(t_neuralnet *x, t_symbol *in_array, t_symbol *out_array)
{
	int array_size;
	int num_samples = 0;
	array_size = check_data_in_arrays_ver(x, in_array);
	/* in case an error occured, the function above returns 0 */
	if (array_size > 0) num_samples += array_size;
	else return;
	array_size = check_data_out_arrays_ver(x, out_array);
	if (array_size > 0) num_samples += array_size;
	else return;
	if (num_samples != (x->x_input_size + x->x_output_size)) {
		pd_error(x, "incorrect argument number: %d\nexpected %d or %d", num_samples, \
				(x->x_input_size + x->x_output_size), (x->x_input_size + 1));
		return;
	}
	x->x_arrays_ver_added = 1;
}

static void shuffle(int *array, size_t n)
{
	if (n > 1) {
		size_t i;
		for (i = 0; i < n - 1; i++) {
			size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
			int t = array[j];
			array[j] = array[i];
			array[i] = t;
		}
	}
}

static void shuffle_train_set(t_neuralnet *x)
{
	int i, j;
	/* create a local array that will hold the random indexes */
	int rand_indexes[x->x_num_in_samples];
	/* create copies of the input and target arrays */
	t_float **main_input_copy, **target_copy;
	main_input_copy = (t_float **)malloc(sizeof(t_float*) * x->x_num_in_samples);
	target_copy = (t_float **)malloc(sizeof(t_float*) * x->x_num_in_samples);
	for (i = 0; i <  x->x_num_in_samples; i++) {
		main_input_copy[i] = (t_float *)malloc(sizeof(t_float) * x->x_input_size);
		if (main_input_copy[i] == NULL) {
			pd_error(x, "shuffle main mem error");
			return;
		}
		target_copy[i] = (t_float *)malloc(sizeof(t_float) * x->x_output_size);
		if (target_copy[i] == NULL) {
			pd_error(x, "shuffle main mem error");
			return;
		}
	}
	/* populate the random indexes array before randomizing it */
	for (i = 0; i < x->x_num_in_samples; i++) {
		rand_indexes[i] = i;
	}
	shuffle(rand_indexes, x->x_num_in_samples);
	/* populate the copies of the input and target */
	for (i = 0; i < x->x_num_in_samples; i++) {
		for (j = 0; j < x->x_input_size; j++) {
			main_input_copy[i][j] = x->x_input[i][j];
		}
		for (j = 0; j < x->x_output_size; j++) {
			target_copy[i][j] = x->x_target_vals[i][j];
		}
	}
	/* finally move values according to random indexes */
	for (i = 0; i < x->x_input_size; i++) {
		for (j = 0; j < x->x_num_in_samples; j++) {
			x->x_input[j][i] = main_input_copy[rand_indexes[j]][i];
			x->x_input_transposed[i][j] = x->x_input[j][i];
		}
	}
	for (i = 0; i < x->x_output_size; i++) {
		for (j = 0; j < x->x_num_in_samples; j++) {
			x->x_target_vals[j][i] = target_copy[rand_indexes[j]][i];
		}
	}
}

/************************ data array functions *********************/

static void data_in_arrays(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv)
{
	t_garray *a;
	int i, j;
	t_word *vec;
	int array_size;
	(void)(s); /* unused */

	if (argc != x->x_input_size) {
		pd_error(x, "number of test inputs MUST be equal to number of inputs in first layer");
		return;
	}

	for (i = 0; i < argc; i++) {
		x->x_arrayname = atom_gensym(argv+i);
		if (!(a = (t_garray *)pd_findbyclass(x->x_arrayname, garray_class))) {
			pd_error(x, "%s: no such array", x->x_arrayname->s_name);
		}
		else if (!garray_getfloatwords(a, &array_size, &vec)) {
			pd_error(x, "%s: bad template for neuralnet", x->x_arrayname->s_name);
		}
		else {
			if (i == 0) {
				x->x_num_in_samples = array_size;
				alloc_main_input(x, x->x_num_in_samples, x->x_old_num_in_samples);
				alloc_transposed_input(x);
			}
			if (array_size != x->x_num_in_samples) {
				pd_error(x, "all test input arrays MUST have equal size");
				return;
			}
			/* copy the array with cast to float because the layers work with floats */
			for (j = 0; j < x->x_num_in_samples; j++) {
				x->x_input[j][i] = x->x_input_transposed[i][j] = vec[j].w_float;
				if (x->x_input[j][i] > 1.0 || x->x_input[j][i] < -1.0) {
					if (fabs(x->x_input[j][i]) > x->x_max_in_vals[i]) {
						x->x_max_in_vals[i] = fabs(x->x_input[j][i]);
					}
					x->x_must_normalize_input = 1;
				}
			}
		}
	}
}

static void data_out_arrays(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv)
{
	t_garray *a;
	int i, j, k;
	t_word *vec;
	int array_size;
	(void)(s); /* unused */

	if (argc != x->x_output_size) {
		/* if we're in classification mode and the target values
		   are labelled categorically, no error should be output  */
		if (!x->x_classification) {
			pd_error(x, "number of test outputs MUST be equal to number of outputs in last layer");
			return;
		}
	}

	for (i = 0; i < argc; i++) {
		x->x_arrayname = atom_gensym(argv+i);
		if (!(a = (t_garray *)pd_findbyclass(x->x_arrayname, garray_class))) {
			pd_error(x, "%s: no such array", x->x_arrayname->s_name);
		}
		else if (!garray_getfloatwords(a, &array_size, &vec)) {
			pd_error(x, "%s: bad template for neuralnet", x->x_arrayname->s_name);
		}
		else {
			if (array_size != x->x_num_in_samples) {
				pd_error(x, "all data in and data out arrays MUST be of equal size");
				return;
			}
			if (i == 0) {
				alloc_target(x, x->x_num_in_samples, x->x_old_num_in_samples);
				/* after all allocations (and possible deallocations) have been
				   done, update the x_old_num_in_samples variable */
				update_old_num_in_samples(x);
			}
			/* if we're in classification mode and target is labelled
			   categorically, turn it to one-hot encoded */
			if (x->x_classification) {
				for (j = 0; j < x->x_num_in_samples; j++) {
					/* first fill the whole row with zeros */
					for (k = 0; k < x->x_output_size; k++) {
						x->x_target_vals[j][k] = 0.0;
					}
					/* and then write the one in the index of the input value */
					x->x_target_vals[j][(int)vec[j].w_float] = 1.0;
				}
			}
			else {
				for (j = 0; j < x->x_num_in_samples; j++) {
					x->x_target_vals[j][i] = vec[j].w_float;
					if (x->x_target_vals[j][i] > 1.0 || x->x_target_vals[j][i] < -1.0) {
						if (fabs(x->x_target_vals[j][i]) > x->x_max_out_vals[i]) {
							x->x_max_out_vals[i] = fabs(x->x_target_vals[j][i]);
						}
						x->x_must_normalize_output = 1;
					}
				}
			}
		}
	}
}

/* the following four functions are called by add() in case
   we're loading arrays vertically instead of a list of values */
static int check_data_in_arrays_ver(t_neuralnet *x, t_symbol *s)
{
	t_garray *a;
	t_word *vec;
	int array_size;

	x->x_in_vert_array = s;
	if (!(a = (t_garray *)pd_findbyclass(x->x_in_vert_array, garray_class))) {
		pd_error(x, "%s: no such array", x->x_in_vert_array->s_name);
		array_size = 0;
	}
	else if (!garray_getfloatwords(a, &array_size, &vec)) {
		pd_error(x, "%s: bad template for neuralnet", x->x_in_vert_array->s_name);
		array_size = 0;
	}
	else {
		if (array_size != x->x_input_size) {
			pd_error(x, "array size MUST be equal to number of inputs in first layer");
			array_size = 0;
		}
	}
	return array_size;
}

static void get_data_in_arrays_ver(t_neuralnet *x)
{
	t_garray *a;
	int i;
	t_word *vec;
	int array_size;

	a = (t_garray *)pd_findbyclass(x->x_in_vert_array, garray_class);
	garray_getfloatwords(a, &array_size, &vec);
	/* copy the array with cast to float because the layers work with floats */
	for (i = 0; i < array_size; i++) {
		x->x_input[x->x_num_in_samples-1][i] = x->x_input_transposed[i][x->x_num_in_samples-1] = vec[i].w_float;
		if (x->x_input[x->x_num_in_samples-1][i] > 1.0 || x->x_input[x->x_num_in_samples-1][i] < -1.0) {
			if (fabs(x->x_input[x->x_num_in_samples-1][i]) > x->x_max_in_vals[i]) {
				x->x_max_in_vals[i] = fabs(x->x_input[x->x_num_in_samples-1][i]);
			}
			x->x_must_normalize_input = 1;
		}
	}
}

static int check_data_out_arrays_ver(t_neuralnet *x, t_symbol *s)
{
	t_garray *a;
	t_word *vec;
	int array_size;

	x->x_out_vert_array = s;
	if (!(a = (t_garray *)pd_findbyclass(x->x_out_vert_array, garray_class))) {
		pd_error(x, "%s: no such array", x->x_out_vert_array->s_name);
		array_size = 0;
	}
	else if (!garray_getfloatwords(a, &array_size, &vec)) {
		pd_error(x, "%s: bad template for neuralnet", x->x_out_vert_array->s_name);
		array_size = 0;
	}
	else {
		if (array_size != x->x_output_size) {
			pd_error(x, "array size MUST be equal to the number of neurons on the network's output layer");
			array_size = 0;
		}
	}
	return array_size;
}

static void get_data_out_arrays_ver(t_neuralnet *x)
{
	t_garray *a;
	int i;
	t_word *vec;
	int array_size;

	a = (t_garray *)pd_findbyclass(x->x_out_vert_array, garray_class);
	garray_getfloatwords(a, &array_size, &vec);
	for (i = 0; i < array_size; i++) {
		x->x_target_vals[x->x_num_in_samples-1][i] = vec[i].w_float;
		if (x->x_target_vals[x->x_num_in_samples-1][i] > 1.0 || x->x_target_vals[x->x_num_in_samples-1][i] < -1.0) {
			if (fabs(x->x_target_vals[x->x_num_in_samples-1][i]) > x->x_max_out_vals[i]) {
				x->x_max_out_vals[i] = fabs(x->x_target_vals[x->x_num_in_samples-1][i]);
			}
			x->x_must_normalize_output = 1;
		}
	}
}

/****************** saving and loading models ******************/

static const char *get_full_path(t_neuralnet *x, const char *a)
{
	char *net_path;
	size_t size, size_trimmed;
	int add_suffix = 0;
	/* if we provide a relative path */
	if (!starts_with(a, "/")) {
		int offset = 0;
		int copy_slash = 0;
		if (starts_with(a, "./")) {
			size_trimmed = strlen(a)-1;
			offset = 1;
		}
		else {
			size_trimmed = strlen(a)+1;
			copy_slash = 1;
		}
		if (!ends_with(a, ".ann")) {
			add_suffix = 1;
			size_trimmed += 4;
		}
		size = strlen(canvas_getdir(x->x_canvas)->s_name) + size_trimmed;
		size++; /* for NULL-termination */
		net_path = malloc(size);
		strcpy(net_path, canvas_getdir(x->x_canvas)->s_name);
		if (copy_slash) strcat(net_path, "/");
		strncat(net_path, a+offset, size_trimmed);
		if (add_suffix) strcat(net_path, ".ann");
	}
	else {
		size = strlen(a);
		if (!ends_with(a, ".ann")) {
			add_suffix = 1;
			size += 4;
		}
		size++; /* for NULL-termination */
		net_path = malloc(size);
		strcpy(net_path, a);
		if (add_suffix) strcat(net_path, ".ann");
	}
	return net_path;
}

static void save(t_neuralnet *x, t_symbol *s)
{
	save_net(x, s->s_name);
}

static void save_during_training(t_neuralnet *x)
{
	int num_digits = get_num_digits(x->x_save_every_x_epochs_counter);
	char file_name_extension[num_digits+1];
	const char *net_path = get_full_path(x, x->x_save_every_x_epochs_name->s_name);
	/* below we want to copy the full path without the .ann extension
	   so we subtract 4 but we add 1 for the terminating character
	   so we end up subtracting 3 */
	int file_name_length = strlen(net_path) + num_digits - 3;
	char file_name[file_name_length];
	size_t str_length = strlen(net_path);
	/* convert counter value to string */
	sprintf(file_name_extension, "%d", x->x_save_every_x_epochs_counter);
	strncpy(file_name, net_path, str_length-4);
	strcat(file_name, file_name_extension);
	/* the .ann extension will be added in save() */
	save_net(x, file_name);
	x->x_save_every_x_epochs_counter++;
}

static void save_net(t_neuralnet *x, const char *s)
{
	FILE *fp;
	int i, j, k;
	const char *net_path = get_full_path(x, s);

	fp = fopen(net_path, "w+");

	fprintf(fp, "##network creation data\n");
	for (i = 0; i < x->x_num_layers; i++) {
		if (x->x_layers[i].type == 0) {
			if (!i) {
				fprintf(fp, "%d\n", x->x_layers[i].dense.x_input_size);
			}
			fprintf(fp, "%d\n", x->x_layers[i].dense.x_output_size);
		}
	}
	fprintf(fp, "\n##mode\n");
	fprintf(fp, "%d\n", (int)x->x_classification);
	fprintf(fp, "\n##output type\n");
	fprintf(fp, "%d\n", (int)x->x_output_type);
	fprintf(fp, "\n##activation functions\n");
	for (i = 0; i < x->x_num_layers; i++) {
		fprintf(fp, "#layer-%d\n", i);
		fprintf(fp, "%d\n", x->x_layers[i].x_activation_index);
	}
	fprintf(fp, "\n##max in values\n");
	for (i = 0; i < x->x_input_size; i++) {
		fprintf(fp, "#max_val_ndx-%d\n", i);
		fprintf(fp, "%f\n", x->x_max_in_vals[i]);
	}
	fprintf(fp, "\n##max out values\n");
	for (i = 0; i < x->x_output_size; i++) {
		fprintf(fp, "#max_val_ndx-%d\n", i);
		fprintf(fp, "%f\n", x->x_max_out_vals[i]);
	}
	fprintf(fp, "\n##layer weights\n");
	for (i = 0; i < x->x_num_layers; i++) {
		fprintf(fp, "#layer-%d\n", i);
		if (x->x_layers[i].type == 0) {
			for (j = 0; j < x->x_layers[i].dense.x_input_size; j++) {
				fprintf(fp, "#row-%d\n", j);
				for (k = 0; k < x->x_layers[i].dense.x_output_size; k++) {
					fprintf(fp, "%f", x->x_layers[i].dense.x_weights[j][k]);
					/* avoid writing a white space at the end so that
					   when the file is read, it doesn't read an additional
					   zero value with strtok() */
					if (k < x->x_layers[i].dense.x_output_size - 1) {
						fprintf(fp, " ");
					}
				}
				fprintf(fp, "\n");
			}
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "##layer biases\n");
	for (i = 0; i < x->x_num_layers; i++) {
		fprintf(fp, "#layer-%d\n", i);
		if (x->x_layers[i].type == 0) {
			for (j = 0; j < x->x_layers[i].dense.x_output_size; j++) {
				fprintf(fp, "%f", x->x_layers[i].dense.x_biases[j]);
				/* again, avoid writing a white space at the end of the line */
				if (j < x->x_layers[i].dense.x_output_size - 1) {
					fprintf(fp, " ");
				}
			}
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
	post("saved model: %s", net_path);
}

/* copied from the second example found here
https://solarianprogrammer.com/2019/04/03/
c-programming-read-file-lines-fgets-getline-implement-portable-getline/ */
static int get_max_line_length(t_neuralnet *x, const char *net_path)
{
	char chunk[100];
	/* Store the chunks of text into a line buffer */
	size_t len = sizeof(chunk);
	char *line = malloc(len);
	FILE *fp = fopen(net_path, "r");
	if(fp == NULL) {
		pd_error(x, "NULL file");
		return 0;
	}
	if(line == NULL) {
		pd_error(x, "Unable to allocate memory for the line buffer.");
		return 0;
	}
	/* "Empty" the string */
	line[0] = '\0';

	while(fgets(chunk, sizeof(chunk), fp) != NULL) {
		/* Resize the line buffer if necessary */
		size_t len_used = strlen(line);
		size_t chunk_used = strlen(chunk);

		if(len - len_used < chunk_used) {
			len += 100;
			if((line = realloc(line, len)) == NULL) {
				pd_error(x, "Unable to reallocate memory for the line buffer.");
				free(line);
				return 0;
			}
		}

		/* Copy the chunk to the end of the line buffer */
		strncpy(line + len_used, chunk, len - len_used);
		len_used += chunk_used;
		/* Check if line contains '\n', if yes process the line of text */
		if(line[len_used - 1] == '\n') {
			/* "Empty" the line buffer */
			line[0] = '\0';
		}
	}
	free(line);
	fclose(fp);
	return (int)len;
}

static int starts_with(const char *a, const char *b)
{
	if (strncmp(a, b, strlen(b)) == 0) return 1;
	return 0;
}

static int ends_with(const char *a, const char *b)
{
	/* turn off compiler warning for strncpy() line */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-truncation"
	char *end_of_str;
	end_of_str = malloc(strlen(b));
	strncpy(end_of_str, a+(strlen(a)-strlen(b)), strlen(b));
#pragma GCC diagnostic pop
	if (strncmp(end_of_str, b, strlen(b)) == 0) return 1;
	return 0;
}

static int extract_int(const char *a)
{
	const char *num = strpbrk(a, "0123456789");
	return atoi(num);
}

static void load(t_neuralnet *x, t_symbol *s)
{
	int data_type = -1, old_data_type = -1; /* old_data_type is used to accumulate data types correctly */
	int layer_nr = 0, layer_row = 0;
	int max_val_ndx = 0;
	int num_layers_counter = 0;
	int num_neurons[20];
	int ndx = 0;
	int i;
	int ending_layer = 0;
	int num_layers;
	int min_nr_of_neurons = INT_MAX; /* for detecting the latent space */
	int min_nr_neurons_ndx = 0; /* and the index of the latent space */
	int num_data_in_file = 0;
	int min_num_data_in_file = 8;
	t_atom *neuron_v;
	/*const int max_line_length = get_max_line_length(x, s);*/
	char *line; /*[max_line_length];*/
	/* the following two variable are used to split a string with white spaces */
	char *delim = " ";
	char *token;
	FILE *fp;
	const char *net_path = get_full_path(x, s->s_name);

	int max_line_length = get_max_line_length(x, net_path);
	line = malloc(max_line_length * sizeof(char));

	fp = fopen(net_path, "r");
	if (fp == NULL) {
		/* no need to print an error here, this is done in get_max_line_length() */
		return;
	}

	while (fgets(line, max_line_length, fp)) {
		if (starts_with(line, "##")) {
			/* lines starting with ## denote a block of data */
			if (starts_with(line, "##network creation data")) {
				data_type = 0;
			}
			/* mode is the second data chunk in an .ann file, so as soon as we see this line
			   we start calculating the layers */
			else if (starts_with(line, "##mode")) {
				data_type = 1;
				if (!x->x_morphing) {
					if (x->x_net_io_set) {
						num_layers = x->x_num_layers_from_args;
						ending_layer = num_layers + x->x_first_layer - 1;
					}
					else {
						if (x->x_is_encoder) {
							x->x_first_layer = 0;
							num_layers = min_nr_neurons_ndx + 1;
							ending_layer = min_nr_neurons_ndx;
						}
						else if (x->x_is_decoder) {
							x->x_first_layer = min_nr_neurons_ndx;
							ending_layer = num_layers_counter - 1;
							num_layers = num_layers_counter - min_nr_neurons_ndx;
						}
						else {
							ending_layer = num_layers_counter - 1;
							num_layers = num_layers_counter;
						}
					}
					neuron_v = (t_atom *)getbytes((t_float)num_layers * sizeof(t_atom));
					for (i = 0; i < num_layers; i++) {
						neuron_v[i].a_type = A_FLOAT;
						neuron_v[i].a_w.w_float = (t_float)num_neurons[i+x->x_first_layer];
					}
					create_net(x, num_layers, neuron_v);
				}
			}
			else if (starts_with(line, "##output type")) {
				data_type = 2;
			}
			else if (starts_with(line, "##activation functions")) {
				data_type = 3;
			}
			else if (starts_with(line, "##max in values")) {
				data_type = 4;
			}
			else if (starts_with(line, "##max out values")) {
				data_type = 5;
			}
			else if (starts_with(line, "##layer weights")) {
				data_type = 6;
			}
			else if (starts_with(line, "##layer biases")) {
				data_type = 7;
			}
			else if (starts_with(line, "##autoenc type")) {
				data_type = 8;
			}
		}
		else if (starts_with(line, "#")) {
			if (starts_with(line, "#max_val_ndx")) {
				max_val_ndx = extract_int(line);
			}
			else if (starts_with(line, "#layer")) {
				layer_nr = extract_int(line);
				/* in case of weights we're zeroing ndx twice
				   but in the case of biases we're only zeroing here */
				ndx = 0;
			}
			else if (starts_with(line, "#row")) {
				layer_row = extract_int(line);
				/* necessary zeroing for starting over at a new row */
				ndx = 0;
			}
		}
		else {
			/* don't read empty lines */
			if (strlen(line) > 1) {
				switch (data_type) {
					case 0:
						num_neurons[ndx] = extract_int(line);
						if (num_neurons[ndx] < min_nr_of_neurons) {
							/* store the minimum number of neurons as this is the latent space
							   in case we create an encoder or a decoder */
							min_nr_of_neurons = num_neurons[ndx];
							min_nr_neurons_ndx = ndx;
						}
						ndx++;
						num_layers_counter = ndx;
						if (old_data_type != data_type) {
							num_data_in_file++;
							old_data_type = data_type;
						}
						break;
					case 1:
						/* classification or regression */
						if (!x->x_morphing) x->x_classification = extract_int(line);
						if (old_data_type != data_type) {
							num_data_in_file++;
							old_data_type = data_type;
						}
						break;
					case 2:
						/* output type (regression, classification, binary logistic regression) */
						if (!x->x_morphing) x->x_output_type = extract_int(line);
						if (old_data_type != data_type) {
							num_data_in_file++;
							old_data_type = data_type;
						}
						break;
					case 3:
						/* activation functions */
						if (layer_nr >= x->x_first_layer && layer_nr < ending_layer) {
							int this_layer = layer_nr - x->x_first_layer;
							if (!x->x_morphing) x->x_layers[this_layer].x_activation_index = extract_int(line);
						}
						if (old_data_type != data_type) {
							num_data_in_file++;
							old_data_type = data_type;
						}
						break;
					case 4:
						/* max in values */
						if (x->x_is_whole_net || x->x_first_layer == 0) {
							if (x->x_morphing) x->x_target_max_in_vals[max_val_ndx] = extract_int(line);
							else x->x_max_in_vals[max_val_ndx] = extract_int(line);
						}
						if (old_data_type != data_type) {
							num_data_in_file++;
							old_data_type = data_type;
						}
						break;
					case 5:
						/* max out values */
						if (x->x_is_whole_net || ending_layer == num_layers_counter-1) {
							if (x->x_morphing) x->x_target_max_out_vals[max_val_ndx] = extract_int(line);
							else x->x_max_out_vals[max_val_ndx] = extract_int(line);
						}
						if (old_data_type != data_type) {
							num_data_in_file++;
							old_data_type = data_type;
						}
						break;
					case 6:
						/* weights */
						token = strtok(line, delim);
						while (token != NULL) {
							if (layer_nr >= x->x_first_layer && layer_nr < ending_layer) {
								int this_layer = layer_nr - x->x_first_layer;
								if (x->x_morphing) {
									x->x_layers[this_layer].dense.x_target_weights[layer_row][ndx++] = atof(token);
								}
								else {
									x->x_layers[this_layer].dense.x_weights[layer_row][ndx++] = atof(token);
								}
							}
							token = strtok(NULL, delim);
						}
						if (old_data_type != data_type) {
							num_data_in_file++;
							old_data_type = data_type;
						}
						break;
					case 7:
						/* biases */
						token = strtok(line, delim);
						while (token != NULL) {
							if (layer_nr >= x->x_first_layer && layer_nr < ending_layer) {
								int this_layer = layer_nr - x->x_first_layer;
								if (x->x_morphing) {
									x->x_layers[this_layer].dense.x_target_biases[ndx++] = atof(token);
								}
								else {
									x->x_layers[this_layer].dense.x_biases[ndx++] = atof(token);
								}
							}
							token = strtok(NULL, delim);
						}
						if (old_data_type != data_type) {
							num_data_in_file++;
							old_data_type = data_type;
						}
						break;
					default:
						break;
				}
			}
		}
	}
	if (num_data_in_file == min_num_data_in_file) {
		if (!x->x_morphing) {
			freebytes(neuron_v, num_layers * sizeof(t_atom));
			neuron_v = NULL;
			post("loaded model: %s", net_path);
		}
		x->x_net_trained = 1;
	}
	else if (num_data_in_file != min_num_data_in_file) {
		pd_error(x, "incomplete file with %d number of data", num_data_in_file);
		destroy(x);
	}
	fclose(fp);
	fp = NULL;
}

static void save_every_x_epochs(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv)
{
	(void)(s); /* unused */
	if (argc < 2) {
		pd_error(x, "save_every_x_epochs takes at least 2 arguments");
		return;
	}
	if (argc > 3) {
		pd_error(x, "save_every_x_epochs takes at most 3 arguments");
		return;
	}
	if (argc == 2) {
		if (argv[0].a_type != A_FLOAT || argv[2].a_type != A_SYMBOL) {
			pd_error(x, "if two arguments are provided to save_every_x_epochs, then first must be a float and second must be a symbol");
			return;
		}
	}
	else if (argv[0].a_type != A_FLOAT || argv[1].a_type != A_FLOAT || argv[2].a_type != A_SYMBOL) {
		pd_error(x, "if three arguments are provided to save_every_x_epochs, then first two must be floats and third must be a symbol");
		return;
	}
	x->x_num_epochs_to_save = (int)atom_getfloat(argv++);
	if (argc == 2) x->x_save_every_x_epochs_name = atom_gensym(argv);
	else {
		x->x_start_saving_after_epoch = (int)atom_getfloat(argv++);
		x->x_save_every_x_epochs_name = atom_gensym(argv);
	}
	x->x_save_every_x_epochs = 1;
}

/************************ miscellaneous ***************************/

static int get_num_digits(int n)
{
	int r = 1;
	while (n > 9) {
        n /= 10;
        r++;
    }
    return r;
}

/********************* object initialization **********************/

static void init_object_variables(t_neuralnet *x)
{
	x->x_num_layers = 0;
	/* accuracy variables */
	x->x_acc_precision = 0.0;
	x->x_accuracy_denominator = 250.0;
	x->x_prev_accuracy = 0.0;
	x->x_desired_accuracy = 0.0;
	/* training and validating state variables */
	x->x_epochs = 1000;
	x->x_epoch_count = 0;
	x->x_epoch_old = 0;
	x->x_batch_steps = 1; /* default value so training works in train_net() */
	x->x_batch_count = 0; /* this is compared to the above so non-batch training works */
	x->x_batch_size = 0;
	x->x_is_paused = 0;
	x->x_is_training = 0;
	x->x_is_validating = 0;
	x->x_is_predicting = 0;
	x->x_output_during_training = 0;
	x->x_classification = 0;
	x->x_num_in_samples = 0;
	x->x_old_num_in_samples = 0;
	x->x_percentage = 0;
	x->x_train_del = EPOCH_DEL;
	x->x_net_trained = 0;
	/* optimizer variables */
	x->x_optimizer_index = 0;
	x->x_current_learning_rate = 0.005;
	x->x_learning_rate = 0.005;
	x->x_decay = 0.001;
	x->x_beta_1 = 0.9;
	x->x_beta_2 = 0.999;
	x->x_epsilon = 1.0e-7;
	x->x_rho = 0.9;
	x->x_momentum = 0.5;
	/* variable that determines the type of output
	   whether regression value, class, or binary cross-entropy list */
	x->x_output_type = 0;
	/* loss variables */
	x->x_prev_loss = FLT_MAX;
	x->x_loss_index = 0;
	x->x_desired_loss = FLT_MAX;
	/* confidence variables */
	x->x_confidence_thresh = 0.0;
	x->x_is_confident = 1; /* this is always 1 unless we set a threshold */
	x->x_confidences = 0;
	/* memory allocation state booleans */
	x->x_main_input_allocated = 0;
	x->x_transposed_allocated = 0;
	x->x_target_vals_allocated = 0;
	x->x_train_mem_allocated = 0;
	x->x_test_mem_allocated = 0;
	x->x_max_vals_allocated = 0;
	x->x_outvec_allocated = 0;
	x->x_copy_mem_allocated = 0;
	x->x_morph_mem_allocated = 0;
	x->x_net_created = 0;
	/* data normalization variables */
	x->x_must_normalize_input = 0;
	x->x_must_normalize_output = 0;
	/* morphing variables */
	x->x_morphing = 0;
	x->x_grain = DEFAULTLINEGRAIN;
	x->x_1overtimediff = 1.0;
	x->x_gotinlet = 0;
	x->x_targettime = x->x_prevtime = clock_getlogicaltime();
	/* misc variables */
	x->x_weight_coeff = 0.1;
	x->x_test_optimal_weights = 0;
	x->x_layers_initialized = 0;
	x->x_pred_to_added = 0;
	x->x_leaky_relu_coeff = 0.01;
	/* safety booleans to avoid crashes when adding data
	   or trying to predict without prior adding of arrays */
	x->x_arrays_ver_added = 0;
	x->x_pred_from_added = 0;
	/* number of neurons in input and output layers to create the correct portion of a network */
	x->x_first_layer = 0;
	x->x_num_layers_from_args = 0;
	/* boolean to determine if we are loading the whole net or part of it */
	x->x_is_whole_net = 0;
	/* determine whether we create an encoder or a decoder of an autoencoder */
	x->x_is_encoder = 0;
	x->x_is_decoder = 0;
	x->x_save_every_x_epochs_counter = 0;
	x->x_save_every_x_epochs = 0;
	x->x_num_epochs_to_save = 0;
	x->x_start_saving_after_epoch = 0;
}

static void parse_args(t_neuralnet *x, int argc, t_atom *argv)
{
	int i;
	int load_net = 0;
	int num_float_args = 0;
	int setting_first_layer = 0;
	int setting_num_layers = 0;

	t_symbol *first_layer_sym = gensym("-first_layer");
	t_symbol *num_layers_sym = gensym("-num_layers");
	t_symbol *encoder_sym = gensym("encoder");
	t_symbol *decoder_sym = gensym("decoder");

	t_symbol *net_name;

	/* first check the arguments and their types to determine what needs to be done */
	for (i = 0; i < argc; i++) {
		if (argv->a_type == A_SYMBOL) {
			if (atom_gensym(argv) == first_layer_sym) {
				setting_first_layer = 1;
			}
			else if (atom_gensym(argv) == num_layers_sym) {
				setting_num_layers = 1;
			}
			else if (atom_gensym(argv) == encoder_sym) {
				x->x_is_encoder = 1;
				x->x_is_decoder = 0;
				x->x_is_whole_net = 0;
			}
			else if (atom_gensym(argv) == decoder_sym) {
				x->x_is_decoder = 1;
				x->x_is_encoder = 0;
				x->x_is_whole_net = 0;
			}
			else {
				if (i == 0) {
					net_name = atom_gensym(argv);
					load_net = 1;
				}
			}
		}
		else if (argv->a_type == A_FLOAT) {
			if (setting_first_layer) {
				x->x_first_layer = (int)atom_getfloat(argv);
				setting_first_layer = 0;
			}
			else if (setting_num_layers) {
				x->x_num_layers_from_args = (int)atom_getfloat(argv);
				x->x_net_io_set = 1;
				setting_num_layers = 0;
			}
			num_float_args++; /* in case we want to create a new, unsaved network, see end of function */
		}
		argv++;
	}
	if (load_net) {
		if (!x->x_net_io_set && !x->x_is_encoder && !x->x_is_decoder) x->x_is_whole_net = 1;
		else if (x->x_first_layer == 1) x->x_num_layers_from_args++;
		load(x, net_name);
	}
	else if (num_float_args > 0) {
		/* first we must reset the argv pointer position */
		argv -= argc;
		create_net(x, num_float_args, argv);
	}
	/* reset some object variables */
	x->x_net_io_set = 0;
}

static void *neuralnet_new(t_symbol *s, int argc, t_atom *argv)
{
	t_neuralnet *x = (t_neuralnet *)pd_new(neuralnet_class);
	(void)(s); /* unused */

	/* set a random seed based on time */
	srand(time(0));

	x->x_canvas = canvas_getcurrent();
	x->x_out = outlet_new(&x->x_obj, &s_list);
	x->x_non_conf_out = outlet_new(&x->x_obj, &s_list);
	x->x_epoch_out = outlet_new(&x->x_obj, &s_float);
	x->x_batch_step_out = outlet_new(&x->x_obj, &s_float);
	x->x_acc_out = outlet_new(&x->x_obj, &s_float);
	x->x_loss_out = outlet_new(&x->x_obj, &s_float);
	x->x_clock = clock_new(x, (t_method)train_net);
	x->x_morph_clock = clock_new(x, (t_method)morph_tick);

	init_object_variables(x);

	parse_args(x, argc, argv);

	x->x_act_funcs[0] = linear_forward;
	x->x_act_funcs[1] = sigmoid_forward;
	x->x_act_funcs[2] = bipolar_sigmoid_forward;
	x->x_act_funcs[3] = relu_forward;
	x->x_act_funcs[4] = leaky_relu_forward;
	x->x_act_funcs[5] = rect_softplus_forward;
	x->x_act_funcs[6] = tanh_forward;
	x->x_act_funcs[7] = softmax_forward;

	x->x_act_back[0] = linear_backward;
	x->x_act_back[1] = sigmoid_backward;
	x->x_act_back[2] = bipolar_sigmoid_backward;
	x->x_act_back[3] = relu_backward;
	x->x_act_back[4] = leaky_relu_backward;
	x->x_act_back[5] = rect_softplus_backward;
	x->x_act_back[6] = tanh_backward;
	x->x_act_back[7] = softmax_backward;

	x->x_loss_funcs[0] = mse_forward;
	x->x_loss_funcs[1] = mae_forward;
	x->x_loss_funcs[2] = categ_x_entropy_loss_forward;
	x->x_loss_funcs[3] = bin_x_entropy_loss_forward;

	x->x_loss_back[0] = mse_backward;
	x->x_loss_back[1] = mae_backward;
	x->x_loss_back[2] = categ_x_entropy_loss_backward;
	x->x_loss_back[3] = bin_x_entropy_loss_backward;

	x->x_optimizer_funcs[0] = optimizer_adam_update;
	x->x_optimizer_funcs[1] = optimizer_sgd_update;
	x->x_optimizer_funcs[2] = optimizer_adagrad_update;
	x->x_optimizer_funcs[3] = optimizer_rms_prop_update;

	x->x_act_func_str[0] = "linear";
	x->x_act_func_str[1] = "sigmoid";
	x->x_act_func_str[2] = "bipolar_sigmoid";
	x->x_act_func_str[3] = "relu";
	x->x_act_func_str[4] = "leaky_relu";
	x->x_act_func_str[5] = "rect_softplus";
	x->x_act_func_str[6] = "tanh";
	x->x_act_func_str[7] = "softmax";

	x->x_loss_func_str[0] = "mse";
	x->x_loss_func_str[1] = "mae";
	x->x_loss_func_str[2] = "cat_x_entropy";
	x->x_loss_func_str[3] = "bin_x_entropy";

	x->x_optimizer_func_str[0] = "adam";
	x->x_optimizer_func_str[1] = "sgd";
	x->x_optimizer_func_str[2] = "adagrad";
	x->x_optimizer_func_str[3] = "rms_prop";

	return(x);
}

void neuralnet_setup(void)
{
	neuralnet_class = class_new(gensym("neuralnet"),
			(t_newmethod)neuralnet_new, (t_method)neuralnet_free,
			sizeof(t_neuralnet), 0, A_GIMME, 0);
	class_addmethod(neuralnet_class, (t_method)add_layer,
			gensym("addlayer"), A_GIMME, 0);
	class_addmethod(neuralnet_class, (t_method)data_in_arrays,
			gensym("data_in_arrays"), A_GIMME, 0);
	class_addmethod(neuralnet_class, (t_method)data_out_arrays,
			gensym("data_out_arrays"), A_GIMME, 0);
	class_addmethod(neuralnet_class, (t_method)train,
			gensym("train"), A_GIMME, 0);
	class_addmethod(neuralnet_class, (t_method)validate,
			gensym("validate"), 0, 0);
	class_addmethod(neuralnet_class, (t_method)set_batch_size,
			gensym("set_batch_size"), A_GIMME, 0);
	class_addmethod(neuralnet_class, (t_method)predict,
			gensym("predict"), A_GIMME, 0);
	class_addmethod(neuralnet_class, (t_method)output_during_training,
			gensym("output_during_training"), A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)set_epochs,
			gensym("set_epochs"), A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)set_activation_function,
			gensym("set_activation_function"), A_GIMME, 0);
	class_addmethod(neuralnet_class, (t_method)create,
			gensym("create"), A_GIMME, 0);
	class_addmethod(neuralnet_class, (t_method)destroy,
			gensym("destroy"), 0, 0);
	class_addmethod(neuralnet_class, (t_method)set_weight_regularizer1,
			gensym("set_weight_reg1"), A_DEFFLOAT, A_DEFFLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)set_weight_regularizer2,
			gensym("set_weight_reg2"), A_DEFFLOAT, A_DEFFLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)set_bias_regularizer1,
			gensym("set_bias_reg1"), A_DEFFLOAT, A_DEFFLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)set_bias_regularizer2,
			gensym("set_bias_reg2"), A_DEFFLOAT, A_DEFFLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)load,
			gensym("load"), A_SYMBOL, 0);
	class_addmethod(neuralnet_class, (t_method)save_every_x_epochs,
			gensym("save_every_x_epochs"), A_GIMME, 0);
	class_addmethod(neuralnet_class, (t_method)save,
			gensym("save"), A_SYMBOL, 0);
	class_addmethod(neuralnet_class, (t_method)add,
			gensym("add"), A_GIMME, 0);
	class_addmethod(neuralnet_class, (t_method)add_arrays,
			gensym("add_arrays"), A_SYMBOL, A_SYMBOL, 0);
	class_addmethod(neuralnet_class, (t_method)normalize_input,
			gensym("normalize_input"), A_GIMME, 0);
	class_addmethod(neuralnet_class, (t_method)normalize_output,
			gensym("normalize_output"), A_GIMME, 0);
	class_addmethod(neuralnet_class, (t_method)shuffle_train_set,
			gensym("shuffle_train_set"), 0, 0);
	class_addmethod(neuralnet_class, (t_method)set_weight_coeff,
			gensym("set_weight_coeff"), A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)set_loss_function,
			gensym("set_loss_function"), A_SYMBOL, 0);
	class_addmethod(neuralnet_class, (t_method)set_optimizer,
			gensym("set_optimizer"), A_SYMBOL, 0);
	class_addmethod(neuralnet_class, (t_method)set_confidences,
			gensym("confidences"), A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)confidence_thresh,
			gensym("confidence_thresh"), A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)set_accuracy_denominator,
			gensym("set_accuracy_denominator"), A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)set_learning_rate,
			gensym("set_learning_rate"), A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)set_decay,
			gensym("set_decay"), A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)set_beta1,
			gensym("set_beta1"), A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)set_beta2,
			gensym("set_beta2"), A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)set_epsilon,
			gensym("set_epsilon"), A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)set_rho,
			gensym("set_rho"), A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)set_momentum,
			gensym("set_momentum"), A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)set_dropout,
			gensym("set_dropout"), A_FLOAT, A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)pause_training,
			gensym("pause_training"), 0, 0);
	class_addmethod(neuralnet_class, (t_method)resume_training,
			gensym("resume_training"), 0, 0);
	class_addmethod(neuralnet_class, (t_method)abort_training,
			gensym("abort_training"), 0, 0);
	class_addmethod(neuralnet_class, (t_method)set_train_del,
			gensym("set_train_del"), A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)classification,
			gensym("classification"), 0, 0);
	class_addmethod(neuralnet_class, (t_method)binary_logistic_regression,
			gensym("binary_logistic_regression"), 0, 0);
	class_addmethod(neuralnet_class, (t_method)regression,
			gensym("regression"), 0, 0);
	class_addmethod(neuralnet_class, (t_method)predict_from,
			gensym("predict_from"), A_SYMBOL, 0);
	class_addmethod(neuralnet_class, (t_method)store_weights_during_training,
			gensym("store_weights_during_training"), A_GIMME, 0);
	class_addmethod(neuralnet_class, (t_method)store_biases_during_training,
			gensym("store_biases_during_training"), A_GIMME, 0);
	class_addmethod(neuralnet_class, (t_method)weights_scale_coeff,
			gensym("weights_scale_coeff"), A_FLOAT, A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)biases_scale_coeff,
			gensym("biases_scale_coeff"), A_FLOAT, A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)predict_to,
			gensym("predict_to"), A_SYMBOL, 0);
	class_addmethod(neuralnet_class, (t_method)morph,
			gensym("morph"), A_SYMBOL, A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)desired_loss,
			gensym("desired_loss"), A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)desired_accuracy,
			gensym("desired_accuracy"), A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)retrain,
			gensym("retrain"), 0, 0);
	class_addmethod(neuralnet_class, (t_method)keep_training,
			gensym("keep_training"), 0, 0);
	class_addmethod(neuralnet_class, (t_method)release_mem,
			gensym("release_mem"), 0, 0);
	class_addmethod(neuralnet_class, (t_method)set_seed,
			gensym("seed"), A_FLOAT, 0);
	class_addmethod(neuralnet_class, (t_method)set_leaky_relu_coeff,
			gensym("set_leaky_relu_coeff"), A_FLOAT, 0);
}
