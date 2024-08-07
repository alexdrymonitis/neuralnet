typedef struct _layer_embedding
{
	int vocab_size;
	int x_embedding_dim;
	int x_output_dim;
	float *embedding_matrix;
	float *gradients;
} t_layer_embedding;

void embedding_layer_update_embedding_matrix(t_layer_embedding *x);
void embedding_layer_set_vocab_size(t_layer_embedding *x, t_floatarg f);
void embedding_layer_set_embedding_dim(t_layer_embedding *x, t_floatarg f);
void embedding_layer_compute(t_layer_embedding *x, t_floatarg index);
int embedding_layer_update(t_layer_embedding *x, t_floatarg index, t_symbol *s, int argc, t_atom *argv);
void embedding_layer_free(t_layer_embedding *x);
