#include "m_pd.h"
#include "float.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "limits.h"
#include "dense.h"
#include "rnn.h"
#include "embedding.h"

#define RAND_HIGH_RANGE (0.10)
#define RAND_MIN_RANGE (-0.10)
#define INIT_BIASES (0.0)
#define NUM_ACT_FUNCS 8
#define SOFTMAX_INDEX NUM_ACT_FUNCS - 1
#define NUM_LOSS_FUNCS 4
#define NUM_OPTIMIZER_FUNCS 4
#define CAT_X_ENT_NDX 2
#define BIN_X_ENT_NDX 3
#define EPOCH_DEL (20.0)
#define DEFAULTLINEGRAIN 20

/* initial neural struc so it can be included
   in the function pointers below, because this pointer contains
   the t_neuralnet struct before it has been defined */
struct _neuralnet;
/* define a pointer to the activation functions
   so we can call them dynamically */
typedef t_float (*act_funcs)(struct _neuralnet *x, int input_size, int index, int out);
typedef void (*act_back)(struct _neuralnet *x, t_float **dvalues, int index);
/* do the same for the loss functions, both forward and backward */
typedef t_float (*loss_funcs)(struct _neuralnet *x);
typedef void (*loss_back)(struct _neuralnet *x);
/* and for the optimizers */
typedef void (*optimizer_funcs)(struct _neuralnet *x, int index);

typedef struct _layer
{
	int type;
	/* activation function index (activation functions are called
	   via a function pointer array) */
	int x_activation_index;
	/* regularization variables */
	t_float x_weight_regularizer_l1;
	t_float x_weight_regularizer_l2;
	t_float x_bias_regularizer_l1;
	t_float x_bias_regularizer_l2;
	union {
		t_layer_dense dense;
		t_layer_rnn rnn;
		t_layer_embedding embedding;
	};
} t_layer;

typedef struct _neuralnet
{
	/* object structure */
	t_object x_obj;
	t_canvas *x_canvas; /* to get the directory of the containing patch */
	t_outlet *x_out;
	t_outlet *x_non_conf_out; /* output when not really confident */
	t_outlet *x_loss_out;
	t_outlet *x_acc_out;
	t_outlet *x_epoch_out;
	t_outlet *x_batch_step_out;
	t_clock *x_clock;
	/* arguments to explicitly set a portion of a trained network */
	int x_first_layer;
	int x_num_layers_from_args;
	/* network states */
	int x_is_whole_net;
	int x_net_io_set;
	/* network input vectors */
	t_float **x_input;
	t_float **x_input_transposed;
	/* network precition output vector */
	t_atom *x_outvec;
	/* network output target vector */
	t_float **x_target_vals;
	/* loss back propagation vector */
	t_float **x_loss_dinput;
	/* input and output scaling vectors */
	t_float *x_max_in_vals;
	t_float *x_max_out_vals;
	/* morphing vectors */
	t_float *x_set_max_in_vals;
	t_float *x_target_max_in_vals;
	t_float *x_set_max_out_vals;
	t_float *x_target_max_out_vals;
	/* batch training vectors */
	t_float **x_batch_input;
	t_float **x_batch_transposed;
	t_float **x_batch_target;
	/* network layers vector */
	t_layer *x_layers;
	/* optimizer variables */
	t_float x_current_learning_rate;
	t_float x_learning_rate;
	t_float x_beta_1;
	t_float x_beta_2;
	t_float x_epsilon;
	t_float x_decay;
	t_float x_momentum;
	t_float x_rho;
	int x_iterations;
	/* accuracy variables */
	t_float x_accuracy;
	t_float x_acc_precision;
	t_float x_prev_accuracy;
	t_float x_prev_loss;
	t_float x_accuracy_denominator;
	t_float x_desired_accuracy;
	/* activation functions */
	act_funcs x_act_funcs[NUM_ACT_FUNCS];
	act_back x_act_back[NUM_ACT_FUNCS];
	const char *x_act_func_str[NUM_ACT_FUNCS];
	int x_output_type; /* if output is regression, classes, or binary cross-entropy */
	t_float x_leaky_relu_coeff;
	/* loss functions */
	loss_funcs x_loss_funcs[NUM_LOSS_FUNCS];
	loss_back x_loss_back[NUM_LOSS_FUNCS];
	const char *x_loss_func_str[NUM_LOSS_FUNCS];
	t_float x_desired_loss;
	t_float x_loss;
	int x_loss_index;
	/* optimizer functions */
	optimizer_funcs x_optimizer_funcs[NUM_OPTIMIZER_FUNCS];
	const char *x_optimizer_func_str[NUM_OPTIMIZER_FUNCS];
	int x_optimizer_index;
	/* user settable variables */
	t_float x_weight_coeff;
	t_float x_confidence_thresh;
	int x_epochs;
	int x_epoch_count;
	int x_epoch_old; /* for resuming a paused training session */
	int x_is_paused;
	t_int x_batch_size;
	int x_percentage;
	t_int x_train_del;
	int x_net_trained;
	t_int x_classification; /* used to determine accuracy function */
	/* variables set by various functions */
	t_symbol *x_arrayname; /* for input and output train and test data */
	t_symbol *x_in_vert_array; /* vertical array input */
	t_symbol *x_out_vert_array; /* vertical array output */
	t_symbol *x_predict_from; /* for input prediction data */
	t_symbol *x_predict_to; /* for output prediction data */
	t_symbol *x_save_every_x_epochs_name; /* for saving every x epochs during training */
	int x_save_every_x_epochs_counter;
	int x_save_every_x_epochs; /* boolean to determine whether we'll save every x epochs */
	int x_num_epochs_to_save; /* the number every which we'll save during training */
	int x_start_saving_after_epoch; /* offset to not start saving from the beginning */
	int x_pred_to_array;
	int x_num_layers;
	int x_input_size;
	int x_output_size;
	int x_num_in_samples;
	int x_old_num_in_samples;
	int x_is_training;
	int x_is_validating;
	int x_is_predicting;
	int x_output_during_training;
	int x_layers_initialized;
	int x_batch_count;
	int x_batch_steps;
	/* memory allocation booleans */
	int x_main_input_allocated;
	int x_transposed_allocated;
	int x_target_vals_allocated;
	int x_train_mem_allocated;
	int x_test_mem_allocated;
	int x_max_vals_allocated;
	int x_outvec_allocated;
	int x_copy_mem_allocated;
	int x_morph_mem_allocated;
	int x_net_created;
	/* normalization */
	int x_must_normalize_input;
	int x_must_normalize_output;
	int x_test_optimal_weights;
	/* confidence variables */
	int x_confidences;
	int x_is_confident;
	/* morphing variables */
	t_clock *x_morph_clock;
	int x_morphing;
	double x_targettime;
	double x_prevtime;
	double x_morph_ramp_dur;
	double x_1overtimediff;
	t_float x_grain;
	int x_gotinlet;
	/* safety booleans to avoid crashes when adding data
	   or trying to predict without prior adding of arrays */
	int x_arrays_ver_added;
	int x_pred_from_added;
	int x_pred_to_added;
	/* detect whether we create an encoder or decoder of an autoencoder */
	int x_is_encoder;
	int x_is_decoder;
} t_neuralnet;

/********** all t_neuralnet object variables initialization ***********/
static void init_object_variables(t_neuralnet *x);
static void init_biases(t_neuralnet *x);

/**************** parsing arguments on creation of network *************/
static void parse_args(t_neuralnet *x, int argc, t_atom *argv); 

/**************************** layer funcitons **************************/
/* forward pass for one layer */
//static void layer_forward(t_float **previous_layer_output,
//		t_layer_dense *this_layer, int input_size,
//		int is_training, int is_testing);
/* backward pass for one layer (used in back propagation) */
//static void layer_backward(t_layer *layer, t_layer_dense *layer_dense, t_float **previous_layer_output, int dot_loop_size);
static void layer_init_common_vars(t_neuralnet *x, int layer_nr);
static void layer_dense_init(t_neuralnet *x, t_layer_dense *layer, int input_size, int output_size);
static void set_dropout(t_neuralnet *x, t_float which_layer, t_float rate);
static void set_weight_coeff(t_neuralnet *x, t_float f);
/* return a uniformly distributed random value */
static t_float rand_gen();
static t_float normal_random();
static void init_layer_weights(t_neuralnet *x, t_layer_dense *layer);
static void init_layer_biases(t_layer_dense *layer);
static void populate_eye(t_layer_dense *layer);
//static void dropout_forward(t_layer_dense *layer, t_float **prev_layer_output, int rows);

/********************* memory allocation functions *********************/
/* memory for the test set (also used for the train set) */
static void alloc_test_mem(t_neuralnet *x);
/* memory for the train set */
static void alloc_train_mem(t_neuralnet *x);
/* morphing memory allocation */
static void alloc_morph_mem(t_neuralnet *x);
/* main input memory, used for training, testing, and predicting */
static void alloc_main_input(t_neuralnet *x, int num_samples, int old_num_samples);
/* memory for the transposed input used in a dot product function */
static void alloc_transposed_input(t_neuralnet *x);
/* memory for the target values, used for training */
static void alloc_target(t_neuralnet *x, int num_samples, int old_num_samples);
/* memory for the maximum input and output values, used for scaling
   input and output to a range between 0 and 1 or -1 and 1 */
static void alloc_max_vals(t_neuralnet *x);
/* memory allocated for batch training */
static void alloc_batch_mem(t_neuralnet *x);
/* functions for the dense layer */
static int alloc_dropout_mem(t_layer_dense *layer, int num_samples);
static int alloc_dense_train_mem(t_layer_dense *layer, int num_samples);
/******************** memory deallocation functions *******************/
static void dealloc_train_mem(t_neuralnet *x, int num_samples);
static void dealloc_morph_mem(t_neuralnet *x);
static void dealloc_test_mem(t_neuralnet *x, int num_samples);
static void dealloc_main_input(t_neuralnet *x, int old_num_samples);
static void dealloc_target(t_neuralnet *x, int old_num_samples);
static void dealloc_transposed_input(t_neuralnet *x);
static void dealloc_max_vals(t_neuralnet *x);
static void dealloc_dropout_mem(t_layer_dense *layer, int num_samples);
static void dealloc_batch_mem(t_neuralnet *x);
/* functions for the dense layer */
static void dealloc_layer(t_layer_dense *layer);
static void dealloc_dense_train_mem(t_layer_dense *layer, int num_samples);
/***************** memory reallocation functions **********************/
/* used for dynamically growing vectors when inputing training data manually */
static void realloc_main_input(t_neuralnet *x);
static void realloc_transposed_input(t_neuralnet *x);
static void realloc_target(t_neuralnet *x);

/******************* data normalization functions **********************/
/* normalization functions to bring input and output to
   a range between 0 and 1 or -1 and 1
   the first two are internal, the last two are called via messages */
static void norm_input(t_neuralnet *x);
static void norm_output(t_neuralnet *x);
static void normalize_input(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv);
static void normalize_output(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv);

/**************************** loss funcitons **************************/
/* forward pass for the loss function */
static t_float loss_forward(t_neuralnet *x);
/* backward pass for the loss function (used in back propagation) */
static void loss_backward(t_neuralnet *x);
/* regularization loss in case it is used */
static t_float regularization_loss(t_neuralnet *x);

static t_float mse_forward(t_neuralnet *x);
static t_float mae_forward(t_neuralnet *x);
static t_float categ_x_entropy_loss_forward(t_neuralnet *x);
static t_float bin_x_entropy_loss_forward(t_neuralnet *x);
static void mse_backward(t_neuralnet *x);
static void mae_backward(t_neuralnet *x);
static void categ_x_entropy_loss_backward(t_neuralnet *x);
static void bin_x_entropy_loss_backward(t_neuralnet *x);
static void set_loss_function(t_neuralnet *x, t_symbol *s);
/* the following four functions are used to determine whether
   the loss function will take regularization into account or not */
static void set_weight_regularizer1(t_neuralnet *x, t_float layer, t_float reg);
static void set_weight_regularizer2(t_neuralnet *x, t_float layer, t_float reg);
static void set_bias_regularizer1(t_neuralnet *x, t_float layer, t_float reg);
static void set_bias_regularizer2(t_neuralnet *x, t_float layer, t_float reg);
/* function to set a desired loss
   when done training, if loss is above this value, the memory
   won't be freed and the network will be trainable without re-importing
   the training dataset */
static void desired_loss(t_neuralnet *x, t_float f);

/************************ accuracy functions ***************************/
/* modified version of code copied from
https://www.programiz.com/c-programming/examples/standard-deviation */
static t_float standard_deviation(t_float **data, int rows, int cols);
/* set the accuracy precision camparison value */
static void set_accuracy_precision(t_neuralnet *x);
static void set_accuracy_denominator(t_neuralnet *x, t_float f);
/* function to return the currect accuracy */
static t_float get_accuracy(t_neuralnet *x);
/* function to set a desired accuracy
   when done training, if accuracy is below this value, the memory
   won't be freed and the network will be trainable without re-importing
   the training dataset */
static void desired_accuracy(t_neuralnet *x, t_float f);

/********************* confidence functions ***********************/
/* set a confidence threshold below which the network
   won't output a predicted class, defaults to 0 */
static void confidence_thresh(t_neuralnet *x, t_float f);
/* set whether the object should output
   the confidences or the predicted class */
static void set_confidences(t_neuralnet *x, t_float f);

/*********************** activation functions ***********************/
/* forward pass for one activation function
   calls whichever activation function has been set to each layer */
static void activation_forward(t_neuralnet *x, int input_size, int index);
/* backward pass for one activation function (used in back propagation) */
static void activation_backward(t_neuralnet *x, t_float **dvalues, int index);

static t_float sigmoid_forward(t_neuralnet *x, int input_size, int index, int out);
static t_float bipolar_sigmoid_forward(t_neuralnet *x, int input_size, int index, int out);
static t_float relu_forward(t_neuralnet *x, int input_size, int index, int out);
static t_float leaky_relu_forward(t_neuralnet *x, int input_size, int index, int out);
static t_float rect_softplus_forward(t_neuralnet *x, int input_size, int index, int out);
static t_float tanh_forward(t_neuralnet *x, int input_size, int index, int out);
static t_float linear_forward(t_neuralnet *x, int input_size, int index, int out);
static t_float softmax_forward(t_neuralnet *x, int input_size, int index, int out);
/* backward activation functions */
static void sigmoid_backward(t_neuralnet *x, t_float **dvalues, int index);
static void bipolar_sigmoid_backward(t_neuralnet *x, t_float **dvalues, int index);
static void relu_backward(t_neuralnet *x, t_float **dvalue, int index);
static void leaky_relu_backward(t_neuralnet *x, t_float **dvalue, int index);
static void rect_softplus_backward(t_neuralnet *x, t_float **dvalue, int index);
static void tanh_backward(t_neuralnet *x, t_float **dvalue, int index);
static void linear_backward(t_neuralnet *x, t_float **dvalues, int index);
static void softmax_backward(t_neuralnet *x, t_float **dvalues, int index);
/* set the activation function for one layer */
static void set_activation_function(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv);
/* common function for all forward passes to set values to the output list */
static void set_vec_val(t_neuralnet *x, t_atom *vec, t_float val, int ndx);

/*********************** optimizer functions **************************/
static void optimizer_pre_update(t_neuralnet *x);
static void optimizer_adam_update(t_neuralnet *x, int index);
static void optimizer_sgd_update(t_neuralnet *x, int index);
static void optimizer_adagrad_update(t_neuralnet *x, int index);
static void optimizer_rms_prop_update(t_neuralnet *x, int index);
static void optimizer_post_update(t_neuralnet *x);
static void set_optimizer(t_neuralnet *x, t_symbol *func);
static void set_learning_rate(t_neuralnet *x, t_float f);
static void set_decay(t_neuralnet *x, t_float f);
static void set_beta1(t_neuralnet *x, t_float f);
static void set_beta2(t_neuralnet *x, t_float f);
static void set_epsilon(t_neuralnet *x, t_float f);
static void set_rho(t_neuralnet *x, t_float f);
static void set_momentum(t_neuralnet *x, t_float f);

/************************ min - max functions *************************/
/* statndard min and max functions */
static t_float min(t_float a, t_float b) { return (a < b) ? a : b; }
static t_float max(t_float a, t_float b) { return (a > b) ? a : b; }

/************** network creation and destruction functions *************/
/* network creation function to be called with the "create" message
   calls create_net() internatlly */
static void create(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv);
static void create_net(t_neuralnet *x, int argc, t_atom *argv);
/* destroy the network */
static void destroy_dealloc(t_neuralnet *x);
static void destroy(t_neuralnet *x);
/* free the allocated memory, called when the object is deleted
   or the patch is closed */
static void neuralnet_free(t_neuralnet *x);
/* update the number of input samples
   used to properly deallocate various vectors without causing a crash */
static void update_old_num_in_samples(t_neuralnet *x);

/************ training and testing data sets functions *****************/
/* add training data manually */
static void add(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv);
/* get data as a list, not as array names */
static void get_list_data(t_neuralnet *x, t_atom *argv, int is_one_hot);
/* add vertical input and output arrays */
static void add_arrays(t_neuralnet *x, t_symbol *in_array, t_symbol *out_array);
/* shuffle the training data, for better fitting
   calls the shuffle() function internally */
static void shuffle_train_set(t_neuralnet *x);
/* Arrange the N elements of ARRAY in random order.
   Only effective if N is much smaller than RAND_MAX;
   if this may not be the case, use a better random
   number generator.
   copied from https://benpfaff.org/writings/clc/shuffle.html
   used for shuffling training data for better fitting the model */
static void shuffle(int *array, size_t n);
/* set arrays horizontally for input and output */
static void data_in_arrays(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv);
static void data_out_arrays(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv);
/* set arrays vertically for input and output
   these are called from inside add() */
static int check_data_in_arrays_ver(t_neuralnet *x, t_symbol *s);
static void get_data_in_arrays_ver(t_neuralnet *x);
static int check_data_out_arrays_ver(t_neuralnet *x, t_symbol *s);
static void get_data_out_arrays_ver(t_neuralnet *x);

/* give prediction based on input, used after training */
static void predict(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv);
static void predict_from(t_neuralnet *x, t_symbol *s);
static void predict_to(t_neuralnet *x, t_symbol *s);
/* make a full forward pass */
static void forward_pass(t_neuralnet *x);
/* make a full back propagation */
static void back_propagate(t_neuralnet *x);
/* internal function to train the network */
static void train_net(t_neuralnet *x);
/* function called via the "train" message */
static void train(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv);
static void validate(t_neuralnet *x);
static void retrain(t_neuralnet *x);
static void keep_training(t_neuralnet *x);
static void release_mem(t_neuralnet *x);
static void set_train_del(t_neuralnet *x, t_float f);
static void set_percentage(t_neuralnet *x, int percentage);

/**************** model saving and loading functions *****************/
static const char *get_full_path(t_neuralnet *x, const char *a);
/* the following two functions are called from the load() function */
static int starts_with(const char *a, const char *b);
static int ends_with(const char *a, const char *b);
static int extract_int(const char *a);
/* separate save and save_net functions to easily call save() with a Pd message
   and save_net() from other places within the code */
static void save(t_neuralnet *x, t_symbol *s);
static void save_during_training(t_neuralnet *x);
static void save_net(t_neuralnet *x, const char *s);
static int get_max_line_length(t_neuralnet *x, const char *net_path);
static void load(t_neuralnet *x, t_symbol *s);
static void save_every_x_epochs(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv);

/***************** morphing between models functions *****************/
static void morph_step(t_neuralnet *x, double timenow);
static void store_morph_step(t_neuralnet *x, double timenow);
static void morph_copy_set(t_neuralnet *x);
static void morph_copy_weights(t_neuralnet *x);
static void morph(t_neuralnet *x, t_symbol *s, t_float f);

/**************************** misc functions *************************/
static int get_num_digits(int n); /* for appending to incrementing .ann file names */
/* set number of epochs */
static void set_epochs(t_neuralnet *x, t_float epochs);
static void set_batch_size(t_neuralnet *x, t_symbol *s, int argc, t_atom *argv);
/* copy and restore best weights, based on performance during training */
static void copy_weights_and_biases(t_neuralnet *x);
static void restore_weights_and_biases(t_neuralnet *x);\

static void classification(t_neuralnet *x);
static void binary_logistic_regression(t_neuralnet *x);
static void regression(t_neuralnet *x);
static void set_seed(t_neuralnet *x, t_float f);
