#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Macro function
#define MAX(a,b) \
({ __typeof__ (a) _a = (a); \
__typeof__ (b) _b = (b); \
_a > _b ? _a : _b; })
  
#define MIN(a,b) \
({ __typeof__ (a) _a = (a); \
__typeof__ (b) _b = (b); \
_a < _b ? _a : _b; })

// Input and weight definition
#define N_INPUT_BICYCLE     "squeezenet_weights/bicycle.bin"
#define N_INPUT_CAT         "squeezenet_weights/cat.bin"
#define N_INPUT_CHEETAH     "squeezenet_weights/cheetah.bin"
#define N_INPUT_FOX         "squeezenet_weights/fox.bin"
#define IMG_WIDTH           227
#define IMG_CHANNEL         3
#define BATCH_SIZE          4
#define LABEL_FILE_NAME     "synset_words.txt"

// SqueezeNet weight name
#define N_CONV_W01          "squeezenet_weights/conv1_w"

#define N_FIRE2_S1_W        "squeezenet_weights/fire2_s1_w"
#define N_FIRE2_E1_W        "squeezenet_weights/fire2_e1_w"
#define N_FIRE2_E3_W        "squeezenet_weights/fire2_e3_w"

#define N_FIRE3_S1_W        "squeezenet_weights/fire3_s1_w"
#define N_FIRE3_E1_W        "squeezenet_weights/fire3_e1_w"
#define N_FIRE3_E3_W        "squeezenet_weights/fire3_e3_w"

#define N_FIRE4_S1_W        "squeezenet_weights/fire4_s1_w"
#define N_FIRE4_E1_W        "squeezenet_weights/fire4_e1_w"
#define N_FIRE4_E3_W        "squeezenet_weights/fire4_e3_w"

#define N_FIRE5_S1_W        "squeezenet_weights/fire5_s1_w"
#define N_FIRE5_E1_W        "squeezenet_weights/fire5_e1_w"
#define N_FIRE5_E3_W        "squeezenet_weights/fire5_e3_w"

#define N_FIRE6_S1_W        "squeezenet_weights/fire6_s1_w"
#define N_FIRE6_E1_W        "squeezenet_weights/fire6_e1_w"
#define N_FIRE6_E3_W        "squeezenet_weights/fire6_e3_w"

#define N_FIRE7_S1_W        "squeezenet_weights/fire7_s1_w"
#define N_FIRE7_E1_W        "squeezenet_weights/fire7_e1_w"
#define N_FIRE7_E3_W        "squeezenet_weights/fire7_e3_w"

#define N_FIRE8_S1_W        "squeezenet_weights/fire8_s1_w"
#define N_FIRE8_E1_W        "squeezenet_weights/fire8_e1_w"
#define N_FIRE8_E3_W        "squeezenet_weights/fire8_e3_w"

#define N_FIRE9_S1_W        "squeezenet_weights/fire9_s1_w"
#define N_FIRE9_E1_W        "squeezenet_weights/fire9_e1_w"
#define N_FIRE9_E3_W        "squeezenet_weights/fire9_e3_w"

#define N_CONV_W10          "squeezenet_weights/conv10_w"

// SqueezeNet bias name
#define N_CONV_B01          "squeezenet_weights/conv1_b"

#define N_FIRE2_S1_B        "squeezenet_weights/fire2_s1_b"
#define N_FIRE2_E1_B        "squeezenet_weights/fire2_e1_b"
#define N_FIRE2_E3_B        "squeezenet_weights/fire2_e3_b"

#define N_FIRE3_S1_B        "squeezenet_weights/fire3_s1_b"
#define N_FIRE3_E1_B        "squeezenet_weights/fire3_e1_b"
#define N_FIRE3_E3_B        "squeezenet_weights/fire3_e3_b"

#define N_FIRE4_S1_B        "squeezenet_weights/fire4_s1_b"
#define N_FIRE4_E1_B        "squeezenet_weights/fire4_e1_b"
#define N_FIRE4_E3_B        "squeezenet_weights/fire4_e3_b"

#define N_FIRE5_S1_B        "squeezenet_weights/fire5_s1_b"
#define N_FIRE5_E1_B        "squeezenet_weights/fire5_e1_b"
#define N_FIRE5_E3_B        "squeezenet_weights/fire5_e3_b"

#define N_FIRE6_S1_B        "squeezenet_weights/fire6_s1_b"
#define N_FIRE6_E1_B        "squeezenet_weights/fire6_e1_b"
#define N_FIRE6_E3_B        "squeezenet_weights/fire6_e3_b"

#define N_FIRE7_S1_B        "squeezenet_weights/fire7_s1_b"
#define N_FIRE7_E1_B        "squeezenet_weights/fire7_e1_b"
#define N_FIRE7_E3_B        "squeezenet_weights/fire7_e3_b"

#define N_FIRE8_S1_B        "squeezenet_weights/fire8_s1_b"
#define N_FIRE8_E1_B        "squeezenet_weights/fire8_e1_b"
#define N_FIRE8_E3_B        "squeezenet_weights/fire8_e3_b"

#define N_FIRE9_S1_B        "squeezenet_weights/fire9_s1_b"
#define N_FIRE9_E1_B        "squeezenet_weights/fire9_e1_b"
#define N_FIRE9_E3_B        "squeezenet_weights/fire9_e3_b"

#define N_CONV_B10          "squeezenet_weights/conv10_b"

// SqueezeNet weight size
#define L_CONV_W01          14112

#define L_FIRE2_S1_W        1536
#define L_FIRE2_E1_W        1024
#define L_FIRE2_E3_W        9216

#define L_FIRE3_S1_W        2048
#define L_FIRE3_E1_W        1024
#define L_FIRE3_E3_W        9216

#define L_FIRE4_S1_W        4096
#define L_FIRE4_E1_W        4096
#define L_FIRE4_E3_W        36864

#define L_FIRE5_S1_W        8192
#define L_FIRE5_E1_W        4096
#define L_FIRE5_E3_W        36864

#define L_FIRE6_S1_W        12288
#define L_FIRE6_E1_W        9216
#define L_FIRE6_E3_W        82944

#define L_FIRE7_S1_W        18432
#define L_FIRE7_E1_W        9216
#define L_FIRE7_E3_W        82944

#define L_FIRE8_S1_W        24576
#define L_FIRE8_E1_W        16384
#define L_FIRE8_E3_W        147456

#define L_FIRE9_S1_W        32768
#define L_FIRE9_E1_W        16384
#define L_FIRE9_E3_W        147456

#define L_CONV_W10          512000

// SqueezeNet bias size
#define L_CONV_B01          96

#define L_FIRE2_S1_B        16
#define L_FIRE2_E1_B        64
#define L_FIRE2_E3_B        64

#define L_FIRE3_S1_B        16
#define L_FIRE3_E1_B        64
#define L_FIRE3_E3_B        64

#define L_FIRE4_S1_B        32
#define L_FIRE4_E1_B        128
#define L_FIRE4_E3_B        128

#define L_FIRE5_S1_B        32
#define L_FIRE5_E1_B        128
#define L_FIRE5_E3_B        128

#define L_FIRE6_S1_B        48
#define L_FIRE6_E1_B        192
#define L_FIRE6_E3_B        192

#define L_FIRE7_S1_B        48
#define L_FIRE7_E1_B        192
#define L_FIRE7_E3_B        192

#define L_FIRE8_S1_B        64
#define L_FIRE8_E1_B        256
#define L_FIRE8_E3_B        256

#define L_FIRE9_S1_B        64
#define L_FIRE9_E1_B        256
#define L_FIRE9_E3_B        256

#define L_CONV_B10          1000

#define INPUT_FREE          1
#define INPUT_STAY          0

/* input.h */
struct layer {
	int width;
    int channel;
	int batch;
	float **data;
};
typedef struct layer layer;
void load_input(layer *input, const char *file_name[]);
void load_file(float **w_input, const char *file_name, int f_size);
void print_result(float prob, int index, const char *label_txt);
void softmax(layer input, int items, const char *file_name[], const char *label_txt);

/* conv_layer.h */
layer zero_pad(layer input, int pad);
layer zero_pad_asym(layer input, int up, int down, int left, int right);

// layer zero_pad_asym(layer input, int pad_up, int pad_down, int pad_left, int pad_right);
layer conv_layer(layer input, int filter, int f_size, int stride, int pad, int free_input, float *weight, float *bias);

// Normalization. Only LRN is supported now
layer norm_lrn(layer input, float k, int local_size, float alpha, float beta, int free_input);

// layer batch_norm_layer(layer input, int normalization);
layer concatenate_2layer(layer l1, layer l2, int free_input);
layer concatenate_4layer(layer l1, layer l2, layer l3, layer l4, int free_input);

/* pool_layer.h */
layer avg_pooling_layer(layer input, int kernel, int stride, int pad, int free_input);
layer max_pooling_layer(layer input, int kernel, int stride, int pad, int free_input);

/* fc_layer.h */
layer fc_layer(layer input, int output_channel, int free_input, float *weight, float *bias, int no_relu);

/* debug.h */
void debug_print_input(layer input, const char *log_name);
void debug_print_weight(float *weight, int filters, int i_channel, int f_size, const char *log_name);
void debug_print_bias(float *bias, int channel, const char *log_name);

int main() {
    
	layer input;
    
    layer conv1, pool1;
    layer fire2_s1, fire2_e1, fire2_e3, fire2_concat;
    layer fire3_s1, fire3_e1, fire3_e3, fire3_concat;
    layer fire4_s1, fire4_e1, fire4_e3, fire4_concat, pool4;
    layer fire5_s1, fire5_e1, fire5_e3, fire5_concat;
    layer fire6_s1, fire6_e1, fire6_e3, fire6_concat;
    layer fire7_s1, fire7_e1, fire7_e3, fire7_concat;
    layer fire8_s1, fire8_e1, fire8_e3, fire8_concat, pool8;
    layer fire9_s1, fire9_e1, fire9_e3, fire9_concat;
    layer conv10, pool10;
    
    float *weight = NULL, *bias = NULL;
	
	/* network size and input names */
	const char *input_names[BATCH_SIZE] = {N_INPUT_BICYCLE, N_INPUT_CAT, N_INPUT_CHEETAH, N_INPUT_FOX};
	input.width = IMG_WIDTH;
	input.channel = IMG_CHANNEL;
	input.batch = BATCH_SIZE;
	input.data = malloc(input.batch * sizeof(float *));
	
    /* conv1 */
    load_input(&input, input_names);
		//debug_print_input(input, "sqez_input");
    load_file(&weight, N_CONV_W01, L_CONV_W01);
		//debug_print_weight(weight, 96, 3, 11, "sim_w1");
	load_file(&bias, N_CONV_B01, L_CONV_B01);
		//debug_print_bias(bias, 96, "sim_b1");
    conv1 = conv_layer(input, 96, 7, 2, 0, INPUT_FREE, weight, bias);
		//debug_print_input(conv1, "sqez_conv1");
        
    /* pool1 */
	pool1 = max_pooling_layer(conv1, 3, 2, 0, INPUT_FREE);
		//debug_print_input(pool1, "sqez_pool1");

    /* fire2 */
    load_file(&weight, N_FIRE2_S1_W, L_FIRE2_S1_W);
	load_file(&bias,   N_FIRE2_S1_B, L_FIRE2_S1_B);
    fire2_s1 = conv_layer(pool1, 16, 1, 1, 0, INPUT_FREE, weight, bias);
        //debug_print_input(fire2_s1, "sqez_fire2_s1");
        
    load_file(&weight, N_FIRE2_E1_W, L_FIRE2_E1_W);
	load_file(&bias,   N_FIRE2_E1_B, L_FIRE2_E1_B);
    fire2_e1 = conv_layer(fire2_s1, 64, 1, 1, 0, INPUT_STAY, weight, bias);
        //debug_print_input(fire2_e1, "sqez_fire2_e1");
        
    load_file(&weight, N_FIRE2_E3_W, L_FIRE2_E3_W);
	load_file(&bias,   N_FIRE2_E3_B, L_FIRE2_E3_B);
    fire2_e3 = conv_layer(fire2_s1, 64, 3, 1, 1, INPUT_FREE, weight, bias);
        //debug_print_input(fire2_e3, "sqez_fire2_e3");
        
    fire2_concat = concatenate_2layer(fire2_e1, fire2_e3, INPUT_FREE);
        //debug_print_input(fire2_concat, "sqez_fire2_concat");
        
    /* fire3 */
    load_file(&weight, N_FIRE3_S1_W, L_FIRE3_S1_W);
	load_file(&bias,   N_FIRE3_S1_B, L_FIRE3_S1_B);
    fire3_s1 = conv_layer(fire2_concat, 16, 1, 1, 0, INPUT_FREE, weight, bias);
        //debug_print_input(fire3_s1, "sqez_fire3_s1");
        
    load_file(&weight, N_FIRE3_E1_W, L_FIRE3_E1_W);
	load_file(&bias,   N_FIRE3_E1_B, L_FIRE3_E1_B);
    fire3_e1 = conv_layer(fire3_s1, 64, 1, 1, 0, INPUT_STAY, weight, bias);
        //debug_print_input(fire3_e1, "sqez_fire3_e1");
        
    load_file(&weight, N_FIRE3_E3_W, L_FIRE3_E3_W);   
	load_file(&bias,   N_FIRE3_E3_B, L_FIRE3_E3_B);
    fire3_e3 = conv_layer(fire3_s1, 64, 3, 1, 1, INPUT_FREE, weight, bias);
        //debug_print_input(fire3_e3, "sqez_fire3_e3");
        
    fire3_concat = concatenate_2layer(fire3_e1, fire3_e3, INPUT_FREE);
        //debug_print_input(fire3_concat, "sqez_fire3_concat");
    
    /* fire4 */
    load_file(&weight, N_FIRE4_S1_W, L_FIRE4_S1_W);
	load_file(&bias,   N_FIRE4_S1_B, L_FIRE4_S1_B);
    fire4_s1 = conv_layer(fire3_concat, 32, 1, 1, 0, INPUT_FREE, weight, bias);
        //debug_print_input(fire4_s1, "sqez_fire4_s1");
        
    load_file(&weight, N_FIRE4_E1_W, L_FIRE4_E1_W);
	load_file(&bias,   N_FIRE4_E1_B, L_FIRE4_E1_B);
    fire4_e1 = conv_layer(fire4_s1, 128, 1, 1, 0, INPUT_STAY, weight, bias);
        //debug_print_input(fire4_e1, "sqez_fire4_e1");
        
    load_file(&weight, N_FIRE4_E3_W, L_FIRE4_E3_W);
        //debug_print_weight(weight, 128, 32, 3, "debugw");
	load_file(&bias,   N_FIRE4_E3_B, L_FIRE4_E3_B);
        //debug_print_bias(bias, 128, "debugb");
    fire4_e3 = conv_layer(fire4_s1, 128, 3, 1, 1, INPUT_FREE, weight, bias);
        //debug_print_input(fire4_e3, "sqez_fire4_e3");
        
    fire4_concat = concatenate_2layer(fire4_e1, fire4_e3, INPUT_FREE);
        //debug_print_input(fire4_concat, "sqez_fire4_concat");
    
    /* pool4 */
    pool4 = max_pooling_layer(fire4_concat, 3, 2, 0, INPUT_FREE);
        //debug_print_input(pool4, "sqez_pool4");

    /* fire5 */
    load_file(&weight, N_FIRE5_S1_W, L_FIRE5_S1_W);
	load_file(&bias,   N_FIRE5_S1_B, L_FIRE5_S1_B);
    fire5_s1 = conv_layer(pool4, 32, 1, 1, 0, INPUT_FREE, weight, bias);
        //debug_print_input(fire5_s1, "sqez_fire5_s1");
        
    load_file(&weight, N_FIRE5_E1_W, L_FIRE5_E1_W);
	load_file(&bias,   N_FIRE5_E1_B, L_FIRE5_E1_B);
    fire5_e1 = conv_layer(fire5_s1, 128, 1, 1, 0, INPUT_STAY, weight, bias);
        //debug_print_input(fire5_e1, "sqez_fire5_e1");
        
    load_file(&weight, N_FIRE5_E3_W, L_FIRE5_E3_W);
	load_file(&bias,   N_FIRE5_E3_B, L_FIRE5_E3_B);
    fire5_e3 = conv_layer(fire5_s1, 128, 3, 1, 1, INPUT_FREE, weight, bias);
        //debug_print_input(fire5_e3, "sqez_fire5_e3");
        
    fire5_concat = concatenate_2layer(fire5_e1, fire5_e3, INPUT_FREE);
        //debug_print_input(fire5_concat, "sqez_fire5_concat");
    
    /* fire6 */
    load_file(&weight, N_FIRE6_S1_W, L_FIRE6_S1_W);
	load_file(&bias,   N_FIRE6_S1_B, L_FIRE6_S1_B);
    fire6_s1 = conv_layer(fire5_concat, 48, 1, 1, 0, INPUT_FREE, weight, bias);
        //debug_print_input(fire6_s1, "sqez_fire6_s1");
        
    load_file(&weight, N_FIRE6_E1_W, L_FIRE6_E1_W);
	load_file(&bias,   N_FIRE6_E1_B, L_FIRE6_E1_B);
    fire6_e1 = conv_layer(fire6_s1, 192, 1, 1, 0, INPUT_STAY, weight, bias);
        //debug_print_input(fire6_e1, "sqez_fire6_e1");
        
    load_file(&weight, N_FIRE6_E3_W, L_FIRE6_E3_W);
	load_file(&bias,   N_FIRE6_E3_B, L_FIRE6_E3_B);
    fire6_e3 = conv_layer(fire6_s1, 192, 3, 1, 1, INPUT_FREE, weight, bias);
        //debug_print_input(fire6_e3, "sqez_fire6_e3");
        
    fire6_concat = concatenate_2layer(fire6_e1, fire6_e3, INPUT_FREE);
        //debug_print_input(fire6_concat, "sqez_fire6_concat");
    
    /* fire7 */
    load_file(&weight, N_FIRE7_S1_W, L_FIRE7_S1_W);
	load_file(&bias,   N_FIRE7_S1_B, L_FIRE7_S1_B);
    fire7_s1 = conv_layer(fire6_concat, 48, 1, 1, 0, INPUT_FREE, weight, bias);
        //debug_print_input(fire7_s1, "sqez_fire7_s1");
        
    load_file(&weight, N_FIRE7_E1_W, L_FIRE7_E1_W);
	load_file(&bias,   N_FIRE7_E1_B, L_FIRE7_E1_B);
    fire7_e1 = conv_layer(fire7_s1, 192, 1, 1, 0, INPUT_STAY, weight, bias);
        //debug_print_input(fire7_e1, "sqez_fire7_e1");
        
    load_file(&weight, N_FIRE7_E3_W, L_FIRE7_E3_W);
	load_file(&bias,   N_FIRE7_E3_B, L_FIRE7_E3_B);
    fire7_e3 = conv_layer(fire7_s1, 192, 3, 1, 1, INPUT_FREE, weight, bias);
        //debug_print_input(fire7_e3, "sqez_fire7_e3");
        
    fire7_concat = concatenate_2layer(fire7_e1, fire7_e3, INPUT_FREE);
        //debug_print_input(fire7_concat, "sqez_fire7_concat");
    
    /* fire8 */
    load_file(&weight, N_FIRE8_S1_W, L_FIRE8_S1_W);
	load_file(&bias,   N_FIRE8_S1_B, L_FIRE8_S1_B);
    fire8_s1 = conv_layer(fire7_concat, 64, 1, 1, 0, INPUT_FREE, weight, bias);
        //debug_print_input(fire8_s1, "sqez_fire8_s1");
        
    load_file(&weight, N_FIRE8_E1_W, L_FIRE8_E1_W);
	load_file(&bias,   N_FIRE8_E1_B, L_FIRE8_E1_B);
    fire8_e1 = conv_layer(fire8_s1, 256, 1, 1, 0, INPUT_STAY, weight, bias);
        //debug_print_input(fire8_e1, "sqez_fire8_e1");
        
    load_file(&weight, N_FIRE8_E3_W, L_FIRE8_E3_W);
	load_file(&bias,   N_FIRE8_E3_B, L_FIRE8_E3_B);
    fire8_e3 = conv_layer(fire8_s1, 256, 3, 1, 1, INPUT_FREE, weight, bias);
        //debug_print_input(fire8_e3, "sqez_fire8_e3");
        
    fire8_concat = concatenate_2layer(fire8_e1, fire8_e3, INPUT_FREE);
        //debug_print_input(fire8_concat, "sqez_fire8_concat");
    
    /* pool8 */
    pool8 = max_pooling_layer(fire8_concat, 3, 2, 0, INPUT_FREE);
        //debug_print_input(pool8, "sqez_pool8");
        
    /* fire9 */
    load_file(&weight, N_FIRE9_S1_W, L_FIRE9_S1_W);
	load_file(&bias,   N_FIRE9_S1_B, L_FIRE9_S1_B);
    fire9_s1 = conv_layer(pool8, 64, 1, 1, 0, INPUT_FREE, weight, bias);
        //debug_print_input(fire9_s1, "sqez_fire9_s1");
        
    load_file(&weight, N_FIRE9_E1_W, L_FIRE9_E1_W);
	load_file(&bias,   N_FIRE9_E1_B, L_FIRE9_E1_B);
    fire9_e1 = conv_layer(fire9_s1, 256, 1, 1, 0, INPUT_STAY, weight, bias);
        //debug_print_input(fire9_e1, "sqez_fire9_e1");
        
    load_file(&weight, N_FIRE9_E3_W, L_FIRE9_E3_W);
	load_file(&bias,   N_FIRE9_E3_B, L_FIRE9_E3_B);
    fire9_e3 = conv_layer(fire9_s1, 256, 3, 1, 1, INPUT_FREE, weight, bias);
        //debug_print_input(fire9_e3, "sqez_fire9_e3");
        
    fire9_concat = concatenate_2layer(fire9_e1, fire9_e3, INPUT_FREE);
        //debug_print_input(fire9_concat, "sqez_fire9_concat");
    
    /* conv10 */
    load_file(&weight, N_CONV_W10, L_CONV_W10);
	load_file(&bias,   N_CONV_B10, L_CONV_B10);
    conv10 = conv_layer(fire9_concat, 1000, 1, 1, 1, INPUT_FREE, weight, bias);
		//debug_print_input(conv10, "sqez_conv10");
    
    /* pool10 */
    pool10 = avg_pooling_layer(conv10, 15, 1, 0, INPUT_FREE);
        //debug_print_input(pool10, "sqez_pool10");

    /* softmax */
	printf("softmax : %d\n", pool10.channel);
    softmax(pool10, 1000, input_names, LABEL_FILE_NAME);

	return 0;
}

void load_input(layer *input, const char *file_name[]){
	int batch;
	int file_length; 
	FILE *fp;
	file_length = input->channel * input->width * input->width;
	
	for(batch=0; batch<input->batch; batch++) {
		//fp = fopen(file_name[batch], "r");
		input->data[batch] = (float *)malloc(file_length*sizeof(float));
		printf("batch %d, file_length %d file_name %s \n", batch, file_length, file_name[batch]);
		//fread(input->data[batch], sizeof(float), file_length, fp);
		//fclose(fp);
	}
}

void load_file(float **w_input, const char *file_name, int f_length) {
    int i;
    FILE *fp;
    if(*w_input == NULL)
        *w_input = (float *)malloc(f_length*sizeof(float));
    else
        *w_input = (float *)realloc(*w_input, f_length*sizeof(float));
	//fp = fopen(file_name, "r");
	//fread(*w_input, sizeof(float), f_length, fp);
	//fclose(fp);
}

void print_result(float prob, int index, const char *label_txt) {	
    int i;
	FILE *fp = fopen(label_txt,"r");
	char word[200];

	for(i=0; i<index+1; i++)
		fgets(word, sizeof(word), fp);
	word[strlen(word)-1] = '\0';

	printf("%f%% - \"%s\"\n", prob, word);
	fclose(fp);	
}

void softmax(layer input, int items, const char *file_name[], const char *label_txt) {
    
    int i;
	int batch;
    int top[5] = {0, 0, 0, 0, 0};
    float sum;
    float softmax[5];
	
	for(batch=0; batch<input.batch; batch++) {
		/* Initialization for next batch */
		sum = 0.0f; softmax[0] = 0.0f; softmax[1] = 0.0f;
		softmax[2] = 0.0f;  softmax[3] = 0.0f; softmax[4] = 0.0f;
		
		/* Softmax calculation */
		for(i=0; i<items; i++) {
			input.data[batch][i] = expf(input.data[batch][i]);
			sum += input.data[batch][i];
		}
		
		/* Showing percentage */
		for(i=0; i<items; i++)
			input.data[batch][i] = 100.0f * input.data[batch][i] / sum;
		
		/* choosing top-5 */
		for (i=0; i<items; i++) {
			if(input.data[batch][i] > softmax[0]) {
				softmax[4] = softmax[3]; top[4] = top[3];
				softmax[3] = softmax[2]; top[3] = top[2];
				softmax[2] = softmax[1]; top[2] = top[1];
				softmax[1] = softmax[0]; top[1] = top[0];
				softmax[0] = input.data[batch][i]; top[0] = i;
			}
			else if(input.data[batch][i] > softmax[1]) {
				softmax[4] = softmax[3]; top[4] = top[3];
				softmax[3] = softmax[2]; top[3] = top[2];
				softmax[2] = softmax[1]; top[2] = top[1];
				softmax[1] = input.data[batch][i]; top[1] = i;
			}
			else if (input.data[batch][i] > softmax[2]) {
				softmax[4] = softmax[3]; top[4] = top[3];
				softmax[3] = softmax[2]; top[3] = top[2];
				softmax[2] = input.data[batch][i]; top[2] = i;
			}
			else if (input.data[batch][i] > softmax[3]) {
				softmax[4] = softmax[3]; top[4] = top[3];
				softmax[3] = input.data[batch][i]; top[3] = i;
			}
			else if (input.data[batch][i] > softmax[4]) {
				softmax[4] = input.data[batch][i]; top[4] = i;
			}
		}
		printf("-------------Prediction by SqueezeNet for %s-------------\n", file_name[batch]);
		for(i=0; i<5; i++)
			print_result(softmax[i], top[i], label_txt);
	}
}

layer zero_pad_layer(layer input, int pad) {
	
	int batch, channel, row, col;
	int i_index, o_index;
	int i_channel_index, i_row_index;
	int o_channel_index, o_row_index;
	layer output;
    
    if(pad == 0)
        return input;

	output.width = input.width + pad*2;
	output.channel = input.channel;
	output.batch = input.batch;
	output.data = malloc(input.batch * sizeof(float *));
	
	for(batch=0; batch<output.batch; batch++) {
		output.data[batch] = (float *)malloc(output.width*output.width*output.channel*sizeof(float));
		for(channel=0; channel<output.channel; channel++) {
			i_channel_index = channel*input.width*input.width;
			o_channel_index = channel*output.width*output.width;
			for(row=0; row<output.width; row++) {
				i_row_index = (row-pad)*input.width;
				o_row_index = row*output.width;
				for(col=0; col<output.width; col++) {
					o_index = o_channel_index + o_row_index + col;
					//o_index = channel*output.width*output.width + row*output.width + col;
					if(col<pad || row<pad || (output.width-pad) <= col || (output.width-pad) <= row)
						output.data[batch][o_index] = 0.0f;
					else {
						i_index = i_channel_index + i_row_index + (col-pad);
						//i_index = channel*input.width*input.width + (row-pad)*input.width + (col-pad);
						output.data[batch][o_index] = input.data[batch][i_index]; 
					}
				}
			}
		}
		//free(input.data[batch]);
	}
	//free(input.data);
	return output;
}

layer zero_pad_asym(layer input, int up, int down, int left, int right) {
    
	int batch, channel, row, col;
	int i_index, o_index;
	int i_channel_index, i_row_index;
	int o_channel_index, o_row_index;
	layer output;
    //printf("up %d down %d left %d right %d\n", up, down, left, right);
    if((up||down||left||right) == 0) {
		return input; 
	}
	
	output.width = input.width + up + down;
	output.channel = input.channel;
	output.batch = input.batch;
	output.data = malloc(input.batch * sizeof(float *));
	printf("input %d up %d down %d output %d\n", input.width, up, down, output.width);
	for(batch=0; batch<output.batch; batch++) {
		output.data[batch] = (float *)malloc(output.width*output.width*output.channel*sizeof(float));
		for(channel=0; channel<output.channel; channel++) {
			i_channel_index = channel*input.width*input.width;
			o_channel_index = channel*output.width*output.width;
			for(row=0; row<output.width; row++) {
				i_row_index = (row-left)*input.width;
				o_row_index = row*output.width;
				for(col=0; col<output.width; col++) {
					o_index = o_channel_index + o_row_index + col;
					//o_index = channel*output.width*output.width + row*output.width + col;
					if(col<left || row<up || (output.width-right) <= col || (output.width-down) <= row)
						output.data[batch][o_index] = 0.0f;
					else {
						i_index = i_channel_index + i_row_index + (col-up);
						//i_index = channel*input.width*input.width + (row-pad)*input.width + (col-pad);
						output.data[batch][o_index] = input.data[batch][i_index]; 
					}
				}
			}
		}
		//free(input.data[batch]);
	}
	//free(input.data);
	return output;
}

layer conv_layer(layer input, int filter, int f_size, int stride, int pad, int free_input, float *weight, float *bias) {
	
    int i, j;
	int batch, o_channel, row, col, i_channel;
	
	int i_channel_index, i_row_offset, i_col_offset;
	int w_o_channel_index, w_i_channel_index;
	int o_index, o_channel_index, o_row_index, output_size_jh;

	float sum;
    layer output;
	printf("conv : %d * %d * %d\n", input.width, input.width, input.channel);

	layer z_pad = zero_pad_layer(input, pad);
		//printf("\tpad : %d * %d * %d\n", z_pad.width, z_pad.width, z_pad.channel);

	output.width = (z_pad.width + stride - f_size) / stride;
	output.channel = filter;
	output.batch = z_pad.batch;
	output.data = malloc(z_pad.batch * sizeof(float *));
	output_size_jh = output.width*output.width*output.channel;
	
	for(batch=0; batch<output.batch; batch++) {
		output.data[batch] = (float *)malloc(output.width*output.width*output.channel*sizeof(float));
		for(o_channel=0; o_channel<filter; o_channel++) {
			w_o_channel_index = o_channel*z_pad.channel*f_size*f_size;
			o_channel_index = o_channel*output.width*output.width;
			for(row=0; row<output.width; row++) {
				i_row_offset = row*stride;
				o_row_index = row*output.width;
				for(col=0; col<output.width; col++) {
					sum = 0.0f;
					i_col_offset = col*stride;
					o_index = o_channel_index + o_row_index + col;
					for(i_channel=0; i_channel<z_pad.channel; i_channel++) {
						i_channel_index = i_channel*z_pad.width*z_pad.width;
						w_i_channel_index = i_channel*f_size*f_size;
						for(i=0; i<f_size; i++) {
							for(j=0; j<f_size; j++) {
								sum += z_pad.data[batch][i_channel_index + (i_row_offset+i)*z_pad.width + (i_col_offset+j)] * weight[w_o_channel_index + w_i_channel_index + i*f_size + j];
								//sum += z_pad.data[i_channel*z_pad.width*z_pad.width + (row*stride+i)*z_pad.width + col*stride+j] * weight[o_channel*z_pad.channel*f_size*f_size + i_channel*f_size*f_size + i*f_size + j];
							}
						}
					}
					sum += bias[o_channel];
					if(sum < 0) {
						output.data[batch][o_index] = 0.0f;
						//output.data[o_channel*output.width*output.width + row*output.width + col] = 0.0f;
					}
					else {
						output.data[batch][o_index] = sum;
						//output.data[o_channel*output.width*output.width + row*output.width + col] = sum;
					}
					printf("computing conv!! : output [%d][%d] / total[%d][%d] \n", batch, o_index , output.batch,output_size_jh);
				}
			}
		}	
	}
	
	// Free if z_pad is used.
	if(pad != 0) {
		for(batch=0; batch<output.batch; batch++) {
			free(z_pad.data[batch]);
		}
		free(z_pad.data);
	}
    // Freeing with free_input arg
    if(free_input) {
       for(batch=0; batch<output.batch; batch++) { 
            free(input.data[batch]);
        }
        free(input.data);
    }
    printf("conv layer done!!  \n");
	return output;
}

layer norm_lrn(layer input, float k, int local_size, float alpha, float beta, int free_input) {
    
    int batch, channel, row, col, i_channel;
    int channel_index, row_offset, col_offset;
    int data_index, lrn_index;
	int channel_count;
    float lrn_sum;
    
    layer output;
	output.width = input.width;
	output.channel = input.channel;
	output.batch = input.batch;
	output.data = malloc(output.batch * sizeof(float *));
    
	for(batch=0; batch<output.batch; batch++) {
		output.data[batch] = (float *)malloc(output.width*output.width*output.channel*sizeof(float));
		for(channel=0; channel<output.channel; channel++) {
            channel_index = channel*output.width*output.width;
			for(row=0; row<output.width; row++) {
                row_offset = row*output.width;
				for(col=0; col<output.width; col++) {
                    lrn_sum = 0.0f;
					channel_count = 0;
                    for(i_channel=MAX(0, channel-local_size/2); i_channel<=MIN(output.channel-1, channel+local_size/2); i_channel++) {
                        lrn_index = i_channel*output.width*output.width + row_offset + col;
                        lrn_sum += input.data[batch][lrn_index]*input.data[batch][lrn_index];
						channel_count++;
                    }
                    data_index = channel_index+row_offset+col;
                    output.data[batch][data_index] = input.data[batch][data_index] / powf(k + ((alpha*lrn_sum) / (float)local_size), beta);
                    //printf("count = %f\n", (float)channel_count);
                }
            }
        }
    }

    // Freeing with free_input arg
    if(free_input) {
       for(batch=0; batch<output.batch; batch++) { 
            free(input.data[batch]);
        }
        free(input.data);
    }
    return output;
}

// layer width is identical
layer concatenate_2layer(layer l1, layer l2, int free_input) {
    
    int batch;
    int l1_length, l2_length, output_length;
    layer output;
    
    output.width = l1.width;
    output.channel = l1.channel + l2.channel;
    output.batch = l1.batch;
    output.data = malloc(output.batch * sizeof(float *));
    
    l1_length = l1.channel * l1.width * l1.width;
    l2_length = l2.channel * l2.width * l2.width;
    output_length = output.channel * output.width * output.width;
    
    printf("layer concatenate %d * %d * (%d + %d)\n", output.width, output.width, l1.channel, l2.channel);
    for(batch=0; batch<output.batch; batch++) {
        output.data[batch] = (float *)malloc(output_length * sizeof(float));
        memcpy(output.data[batch], l1.data[batch], l1_length * sizeof(float));
        memcpy(output.data[batch] + l1_length, l2.data[batch], l2_length * sizeof(float));
    }
    
    // Freeing with free_input arg
    if(free_input) {
        for(batch=0; batch<output.batch; batch++) { 
            free(l1.data[batch]);
            free(l2.data[batch]);
        }
        free(l1.data);
        free(l2.data);
    }
    return output;
}

// layer width is identical
layer concatenate_4layer(layer l1, layer l2, layer l3, layer l4, int free_input) {
    
    int batch;
    int l1_length, l2_length, l3_length, l4_length, output_length;
    layer output;
    
    output.width = l1.width;
    output.channel = l1.channel + l2.channel + l3.channel + l4.channel;
    output.batch = l1.batch;
    output.data = malloc(output.batch * sizeof(float *));
    
    l1_length = l1.channel * l1.width * l1.width;
    l2_length = l2.channel * l2.width * l2.width;
    l3_length = l3.channel * l3.width * l3.width;
    l4_length = l4.channel * l4.width * l4.width;
    output_length = output.channel * output.width * output.width;
    
    printf("layer concatenate %d * %d * (%d + %d + %d + %d)\n", output.width, output.width, l1.channel, l2.channel, l3.channel, l4.channel);
    for(batch=0; batch<output.batch; batch++) {
        output.data[batch] = (float *)malloc(output_length * sizeof(float));
        memcpy(output.data[batch], l1.data[batch], l1_length * sizeof(float));
        memcpy(output.data[batch] + l1_length, l2.data[batch], l2_length * sizeof(float));
        memcpy(output.data[batch] + l1_length + l2_length, l3.data[batch], l3_length * sizeof(float));
        memcpy(output.data[batch] + l1_length + l2_length + l3_length, l4.data[batch], l4_length * sizeof(float));
    }
    
    // Freeing with free_input arg
    if(free_input) {
        for(batch=0; batch<output.batch; batch++) { 
            free(l1.data[batch]);
            free(l2.data[batch]);
            free(l3.data[batch]);
            free(l4.data[batch]);
        }
        free(l1.data);
        free(l2.data);
        free(l3.data);
        free(l4.data);
    }
    return output;
}

layer avg_pooling_layer(layer input, int kernel, int stride, int pad, int free_input) {
    
    int batch, channel, row, col, i, j;
	int i_channel_index, i_row_stride, i_col_stride;
	int o_channel_index, o_row_index;
    int filter_elements;
	float sum;
	layer output;
    
    filter_elements = kernel*kernel;
    layer z_pad = zero_pad_layer(input, pad);
	
	output.width = (z_pad.width + stride - kernel) / stride;
	output.channel = z_pad.channel;
	output.batch = z_pad.batch;
	output.data = malloc(z_pad.batch * sizeof(float *));

	printf("avg pool : %d * %d * %d\n", input.width, input.width, input.channel);
	
	for(batch=0; batch<output.batch; batch++) {
		output.data[batch] = (float *)malloc(output.width*output.width*output.channel*sizeof(float));
		for(channel=0; channel<output.channel; channel++) {
			i_channel_index = channel*z_pad.width*z_pad.width;
			o_channel_index = channel*output.width*output.width;
			for(row=0; row<output.width; row++) {
				i_row_stride = row*stride;
				o_row_index = row*output.width;
				for(col=0; col<output.width; col++) {
					sum = 0.0;
					i_col_stride = col*stride;
					for(i=0; i<kernel; i++) {
						for(j=0; j<kernel; j++) {
							//i_index = channel*input.width*input.width + (row*stride+i)*input.width + (col*stride+j);
                            sum += z_pad.data[batch][i_channel_index + (i_row_stride+i)*z_pad.width + (i_col_stride+j)];
						}
					}
                    sum = sum / filter_elements;
					output.data[batch][o_channel_index+o_row_index+col] = sum;
					//output.data[channel*output.width*output.width + row*output.width + col] = sum;
				}
			}
		}
	}

	// Free if z_pad is used.
	if(pad != 0) {
		for(batch=0; batch<output.batch; batch++) {
			free(z_pad.data[batch]);
		}
		free(z_pad.data);
	}
    // Freeing with free_input arg
    if(free_input) {
       for(batch=0; batch<output.batch; batch++) { 
            free(input.data[batch]);
        }
        free(input.data);
    }
	return output;
}

layer max_pooling_layer(layer input, int kernel, int stride, int pad, int free_input) {
	
	int batch, channel, row, col, i, j;
	int i_channel_index, i_row_stride, i_col_stride;
	int o_channel_index, o_row_index;
    int asym_pad_incr;
	float max;
	layer z_pad;
	layer asym_pad;
	layer output;
    
    z_pad = zero_pad_layer(input, pad);
	asym_pad_incr = (z_pad.width + stride - kernel) % stride;
	asym_pad = zero_pad_asym(z_pad, 0, asym_pad_incr, 0, asym_pad_incr);
	//printf("asym_pad : %d %d %d\n", asym_pad.batch, asym_pad.width, asym_pad.channel);
        
	output.width = (asym_pad.width + stride - kernel) / stride;
	output.channel = asym_pad.channel;
	output.batch = asym_pad.batch;
	output.data = malloc(asym_pad.batch * sizeof(float *));

	printf("max pool : %d * %d * %d\n", input.width, input.width, input.channel);
	
	for(batch=0; batch<output.batch; batch++) {
		output.data[batch] = (float *)malloc(output.width*output.width*output.channel*sizeof(float));
		for(channel=0; channel<output.channel; channel++) {
			i_channel_index = channel*asym_pad.width*asym_pad.width;
			o_channel_index = channel*output.width*output.width;
			for(row=0; row<output.width; row++) {
				i_row_stride = row*stride;
				o_row_index = row*output.width;
				for(col=0; col<output.width; col++) {
					max = 0.0;
					i_col_stride = col*stride;
					for(i=0; i<kernel; i++) {
						for(j=0; j<kernel; j++) {
							//i_index = channel*input.width*input.width + (row*stride+i)*input.width + (col*stride+j);
							if(max < asym_pad.data[batch][i_channel_index + (i_row_stride+i)*asym_pad.width + (i_col_stride+j)])
								max = asym_pad.data[batch][i_channel_index + (i_row_stride+i)*asym_pad.width + (i_col_stride+j)];
						}
					}
					output.data[batch][o_channel_index+o_row_index+col] = max;
					//output.data[channel*output.width*output.width + row*output.width + col] = max;
				}
			}
		}
	}
	
	// Free if asym_pad is used.
	if(asym_pad_incr != 0) {
		for(batch=0; batch<output.batch; batch++) {
			free(asym_pad.data[batch]);
		}
		free(asym_pad.data);
	}
	// Free if z_pad is used.
	if(pad != 0) {
		for(batch=0; batch<output.batch; batch++) {
			free(z_pad.data[batch]);
		}
		free(z_pad.data);
	}
    // Freeing with free_input arg
    if(free_input && !asym_pad_incr && !pad) {
        for(batch=0; batch<output.batch; batch++) {
            //printf("hell\n");
            //printf("batch %d output.batch %d\n", batch, output.batch);
            //error at batch = 1
            free(input.data[batch]);
        }
        free(input.data);
    }
	return output;
}

layer fc_layer(layer input, int filter, int free_input, float *weight, float *bias, int no_relu) {
	
	int i;
	int batch, o_channel;
	int i_length = input.width * input.width * input.channel;
	int o_channel_index;
	float sum;
	
	layer output;
	output.width = 1;
	output.channel = filter;
	output.batch = input.batch;
	output.data = malloc(input.batch * sizeof(float *));
	printf("fc : %d\n", i_length);
	
	if(no_relu) {
		for(batch=0; batch<output.batch; batch++) {
			output.data[batch] = (float *)malloc(output.channel * sizeof(float));
			for(o_channel=0; o_channel<output.channel; o_channel++) {
				o_channel_index = o_channel*i_length;
				sum = 0.0f;
				for(i=0; i<i_length; i++) {
					sum += input.data[batch][i] * weight[o_channel_index+i];
					//bias[o_channel] += input.data[i] * weight[o_channel*i_length+i];
				}
				sum += bias[o_channel];
				output.data[batch][o_channel] = sum;
			}
		}
	}
	else {
		for(batch=0; batch<output.batch; batch++) {
			output.data[batch] = (float *)malloc(output.channel * sizeof(float));
			for(o_channel=0; o_channel<output.channel; o_channel++) {
				o_channel_index = o_channel*i_length;
				sum = 0.0f;
				for(i=0; i<i_length; i++) {
					sum += input.data[batch][i] * weight[o_channel_index+i];
					//bias[o_channel] += input.data[i] * weight[o_channel*i_length+i];
				}
				sum += bias[o_channel];
				if(sum < 0)
					output.data[batch][o_channel] = 0.0f;
				else
					output.data[batch][o_channel] = sum;
			}
		}
	}
    
    // Freeing with free_input arg
    if(free_input) {
       for(batch=0; batch<output.batch; batch++) { 
            free(input.data[batch]);
        }
        free(input.data);
    }
	return output;
}

void debug_print_input(layer input, const char *log_name) {
    
    int batch, channel, row, col;
    FILE *fp_w;
		printf("\tdebug: %d %d %d\n", input.batch, input.channel, input.width);
    fp_w = fopen(log_name, "w");
    fprintf(fp_w, "layer file : batch = %d channel = %d width = %d\n", input.batch, input.channel, input.width);
	for(batch=0; batch<input.batch; batch++) {
		fprintf(fp_w, "batch #%4d\n", batch+1);
		for(channel=0; channel<input.channel; channel++) {
			fprintf(fp_w, "channel #%4d\n", channel+1);
			for(row=0; row<input.width; row++) {
				for(col=0; col<input.width; col++) {
					fprintf(fp_w, "%f  ", input.data[batch][channel*input.width*input.width + row*input.width + col]);
				}
				fprintf(fp_w, "\n");
			}
		}
	}
    fclose(fp_w);
}

void debug_print_weight(float *weight, int filters, int i_channel, int f_size, const char *log_name) {
    
    int channel, ic, row, col;
    FILE *fp_w;
    fp_w = fopen(log_name, "w");
    
    fprintf(fp_w, "weight file : f_chan = %d i_chan = %d width = %d\n", filters, i_channel, f_size);
    for(channel=0; channel<filters; channel++) {
        fprintf(fp_w, "channel #%4d\n", channel+1);
        for(ic=0; ic<i_channel; ic++) {
            fprintf(fp_w, "input channel #%4d\n", ic+1);
            for(row=0; row<f_size; row++) {
                for(col=0; col<f_size; col++) {
                    fprintf(fp_w, "%f  ", weight[channel*i_channel*f_size*f_size + ic*f_size*f_size+ row*f_size + col]);
                }
                fprintf(fp_w, "\n");
            }
        }
    }
    fclose(fp_w);
}

void debug_print_bias(float *bias, int channel, const char *log_name) {
    
    int i;
    FILE *fp_w;
    fp_w = fopen(log_name, "w");
    
    fprintf(fp_w, "bias file with channel = %d\n", channel);
    for(i=0; i<channel; i++)
        fprintf(fp_w, "channel #%4d bias = %f\n", i+1, bias[i]);
    fclose(fp_w);
}
