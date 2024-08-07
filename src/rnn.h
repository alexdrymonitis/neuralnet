
#ifndef RNN_H
#define RNN_H

#include <stdlib.h>

#ifdef WINDOWS

#else
#include <unistd.h>
#endif

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>
#include <limits.h>

#include "m_pd.h"

#define	OPTIMIZE_ADAM 0
#define OPTIMIZE_GRADIENT_DESCENT 1

#define RNN_MAX_LAYERS 10
#define DEFAULT_NUM_LAYERS 3
#define DEFAULT_NUM_NEURONS 68

#define ITERATIONS 100000000
#define NO_EPOCHS 0

#define	SET_MAX_CHARS 1000

typedef struct rnn_model_parameters_t
{
	/* For progress monitoring */
	t_float loss_moving_avg;
	/* For gradient descent */
	t_float learning_rate;
	t_float momentum;
	t_float lambda;
	t_float softmax_temp;
	t_float beta1;
	t_float beta2;
	int gradient_clip;
	int gradient_fit;
	int optimizer;
	int model_regularize;
	int stateful;
	int decrease_lr;
	t_float learning_rate_decrease;

	/* How many layers */
	unsigned int layers;
	int cell_type;
	/* How many neurons each layer has */
	unsigned int *neurons;

	/* Output configuration for interactivity */
	long print_progress_iterations;
	int  print_progress;
	int  print_progress_to_file;
	int  print_progress_number_of_chars;
	char *print_sample_output_to_file_name;
	char *print_sample_output_to_file_arg;
	int  store_progress_every_x_iterations;
	char *store_progress_file_name;
	const char *store_network_every_file_name;
	int  store_network_every;

	/* General parameters */
	unsigned int mini_batch_size;
	t_float gradient_clip_limit;
	unsigned long iterations;
	unsigned long epochs;
} rnn_model_parameters_t;

typedef struct rnn_model_t
{
	unsigned int X; /**< Number of input nodes */
	unsigned int N; /**< Number of neurons */
	unsigned int Y; /**< Number of output nodes */
	unsigned int S; /**< rnn_model_t.X + rnn_model_t.N */

	/* Parameters */
	rnn_model_parameters_t * params;

	/* The model */
	t_float* Wi;
	t_float* Wf;
	t_float* Wc;
	t_float* Wo;
	t_float* Wy;
	t_float* bi;
	t_float* bf;
	t_float* bc;
	t_float* bo;
	t_float* by;

	/* cache */
	t_float* dldh;
	t_float* dldho;
	t_float* dldhf;
	t_float* dldhi;
	t_float* dldhc;
	t_float* dldc;

	t_float* dldXi;
	t_float* dldXo;
	t_float* dldXf;
	t_float* dldXc;

	/* Gradient descent momentum */
	t_float* Wfm;
	t_float* Wim;
	t_float* Wcm;
	t_float* Wom;
	t_float* Wym;
	t_float* bfm;
	t_float* bim;
	t_float* bcm;
	t_float* bom;
	t_float* bym;

} rnn_model_t;

typedef struct rnn_values_cache_t
{
	t_float* probs;
	t_float* probs_before_sigma;
	t_float* c;
	t_float* h;
	t_float* c_old;
	t_float* h_old;
	t_float* X;
	t_float* hf;
	t_float* hi;
	t_float* ho;
	t_float* hc;
	t_float* tanh_c_cache;
} rnn_values_cache_t;

typedef struct rnn_values_state_t
{
	t_float* c;
	t_float* h;
} rnn_values_state_t;

typedef struct rnn_values_next_cache_t
{
	t_float* dldh_next;
	t_float* dldc_next;
	t_float* dldY_pass;
} rnn_values_next_cache_t;

typedef struct set_t
{
	char values[SET_MAX_CHARS];
	int free[SET_MAX_CHARS];
} set_t;

/* this is the data structure of the Pd object */
typedef struct _layer_rnn
{
	rnn_model_t **x_model_layers;
	rnn_model_parameters_t x_params;
	/* variables for shuffling training data */
	int x_seq_length;
	int x_num_batches;
	int x_last_shuffled_ndx;
	int x_last_batch_size;
	int *x_shuffled_ndxs;
	/* the following data are used in training and then freed */
	rnn_values_state_t **x_stateful_d_next;
	rnn_values_cache_t ***x_cache_layers;
	rnn_values_cache_t ***x_caches_layer2;
	rnn_values_next_cache_t **x_d_next_layers;
	rnn_model_t **x_gradient_layers;
	rnn_model_t **x_gradient_layers_entry;
	rnn_model_t **x_M_layers;
	rnn_model_t **x_R_layers;
	unsigned int x_training_points;
	t_float x_loss;
	t_float *x_first_layer_input;
	t_float *x_first_layer_input2;
	/* end of data used in training */
	int x_clock_delay;
	set_t x_set;
	int x_set_max_chars;
	int x_text; /* set whether we're processing text or numbers */
	int x_send_progress_to_outlet;
	int x_net_structure_set; /* boolean to determine whether we have set a network structure through arguments/messages */
	unsigned long x_iter_count;
	unsigned long x_epoch_count;
	int x_generate_size;
	int x_stop_byte;
	int x_stop_byte_set;
	int x_store_network_every_counter;
} t_layer_rnn;

#endif
