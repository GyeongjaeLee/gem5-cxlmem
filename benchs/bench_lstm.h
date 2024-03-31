#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


//Input file
#define NAME_INPUT_BICYCLE  "alexnet_weights/bicycle227.bin"
#define NAME_INPUT_CAT      "alexnet_weights/cat227.bin"
#define NAME_INPUT_CHEETAH  "alexnet_weights/cheetah227.bin"
#define NAME_INPUT_FOX      "alexnet_weights/fox227.bin"


//Weight and bias name
#define NAME_EMBB_W		"lstm_weights/lstm_1_ui"
#define NAME_EMBB_B		"lstm_weights/lstm_1_ui"

#define NAME_LSTM_1_UF		"lstm_weights/lstm_1_uf"
#define NAME_LSTM_1_UC		"lstm_weights/lstm_1_uc"
#define NAME_LSTM_1_UI		"lstm_weights/lstm_1_ui"
#define NAME_LSTM_1_UO		"lstm_weights/lstm_1_uo"

#define NAME_LSTM_1_WF		"lstm_weights/lstm_1_wf"
#define NAME_LSTM_1_WC		"lstm_weights/lstm_1_wc"
#define NAME_LSTM_1_WI		"lstm_weights/lstm_1_wi"
#define NAME_LSTM_1_WO		"lstm_weights/lstm_1_wo"

#define NAME_LSTM_2_UF		"lstm_weights/lstm_2_uf"
#define NAME_LSTM_2_UC		"lstm_weights/lstm_2_uc"
#define NAME_LSTM_2_UI		"lstm_weights/lstm_2_ui"
#define NAME_LSTM_2_UO		"lstm_weights/lstm_2_uo"

#define NAME_LSTM_2_WF		"lstm_weights/lstm_2_wf"
#define NAME_LSTM_2_WC		"lstm_weights/lstm_2_wc"
#define NAME_LSTM_2_WI		"lstm_weights/lstm_2_wi"
#define NAME_LSTM_2_WO		"lstm_weights/lstm_2_wo"

#define NAME_FC_W	    			"lstm_weights/fc_w"
#define NAME_FC_B    				"lstm_weights/fc_b"

#define NAME_R_RAND			"lstm_weights/r_rand"



//Weight length
#define LENGTH_EMBB_W      	  160000
#define LENGTH_EMBB_B    		  32

#define LENGTH_LSTM_1_UF  	  19968
#define LENGTH_LSTM_1_UC		  19968
#define LENGTH_LSTM_1_UI		  19968
#define LENGTH_LSTM_1_UO	    19968

#define LENGTH_LSTM_1_WF	  262144
#define LENGTH_LSTM_1_WC      262144
#define LENGTH_LSTM_1_WI      262144
#define LENGTH_LSTM_1_WO      262144

#define LENGTH_LSTM_2_UF      262144 
#define LENGTH_LSTM_2_UC      262144		
#define LENGTH_LSTM_2_UI      262144		
#define LENGTH_LSTM_2_UO      262144	

#define LENGTH_LSTM_2_WF      262144	
#define LENGTH_LSTM_2_WC      262144
#define LENGTH_LSTM_2_WI      262144
#define LENGTH_LSTM_2_WO      262144

#define LENGTH_FC_W    		    317440
#define LENGTH_FC_B  		    	62

#define LENGTH_R_RAND		2000


#define INPUT_BATCH_1 "FILE_NAME"
#define INPUT_BATCH_2 "FILE_NAME"
#define INPUT_BATCH_3 "FILE_NAME"
#define INPUT_BATCH_4 "FILE_NAME"
#define INPUT_BATCH_5 "FILE_NAME"



/*
//Input name
#define SEED1					"Othello"
#define SEED2					"Hamlet"
#define SEED3					"Lear"
#define SEED4					"Macbeth"
*/


//@@@@@@@@@@@@@@@@@@@@@@@@ struct definition @@@@@@@@@@@@@@@@@@@@@@@@@@


typedef struct layer {
	int batch;
	int width;
	int sequence;
	float **data;
} layer;




//@@@@@@@@@@@@@@@@@@@@@@@ function declaration @@@@@@@@@@@@@@@@@@@@@@@@

void lstm_layer(layer *input, layer *output, float **cell_memory, float **cell_output, float *Uf, float *Wf, float *Uc, float *Wc, float *Ui, float *Wi, float *Uo, float *Wo);
void forward(float **input, float **output, int input_width, int output_width, float *weight, int sequence, int cell);
void fc_layer(layer *input, layer *output, float *weight, float *bias, int no_relu);

void load_input(layer *input, const char *file_name[]);
void load_file(float **w_input, const char *file_name, int f_length);


void my_tanh(float **input, float **output, int output_width, int sequence);
void my_tanh_cell(float **input, float **output, int output_width, int sequence);
void sigmoid(float **input, float **output, int output_width, int sequence);

void element_mul(float **input_a, float **input_b, float **output, int output_width, int sequence);
void element_add(float **input_a, float **input_b, float **output, int output_width, int sequence);
void element_cell_add(float **input_a, float **input_b, float **output, int output_width, int sequence);
void element_cell_mul(float **input_a, float **input_b, float **output, int output_width, int sequence);
void element_cell_mul2(float **input_a, float **input_b, float **output, int output_width, int sequence);


void evening(layer *input, layer *output);
void softmax(layer *input, layer *output);

void save_symbol(layer *input, layer *output, int sequence, float **symbol_data);

void print_symbol(float **symbol_data);
void generate_sample_input(layer *input);




