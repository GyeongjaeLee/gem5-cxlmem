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

//Input and weight definition
#define N_INPUT_BICYCLE     "googlenet_weights/bicycle.bin"
#define N_INPUT_CAT         "googlenet_weights/cat.bin"
#define N_INPUT_CHEETAH     "googlenet_weights/cheetah.bin"
#define N_INPUT_FOX         "googlenet_weights/fox.bin"
#define IMG_WIDTH           224
#define IMG_CHANNEL         3
#define BATCH_SIZE          4
#define LABEL_FILE_NAME     "synset_words.txt"

// Googlenet weights
#define N_CONV_W01          "googlenet_weights/conv1_w"

#define N_CONV_RD_W02       "googlenet_weights/conv2_reduce_w"
#define N_CONV_W02          "googlenet_weights/conv2_w"

#define N_INCT_3A_W1        "googlenet_weights/inception_3a_1x1_w"
#define N_INCT_3A_RD_W3     "googlenet_weights/inception_3a_3x3_reduce_w"
#define N_INCT_3A_W3        "googlenet_weights/inception_3a_3x3_w"
#define N_INCT_3A_RD_W5     "googlenet_weights/inception_3a_5x5_reduce_w"
#define N_INCT_3A_W5        "googlenet_weights/inception_3a_5x5_w"
#define N_INCT_3A_PJ_W      "googlenet_weights/inception_3a_pool_proj_w"

#define N_INCT_3B_W1        "googlenet_weights/inception_3b_1x1_w"
#define N_INCT_3B_RD_W3     "googlenet_weights/inception_3b_3x3_reduce_w"
#define N_INCT_3B_W3        "googlenet_weights/inception_3b_3x3_w"
#define N_INCT_3B_RD_W5     "googlenet_weights/inception_3b_5x5_reduce_w"
#define N_INCT_3B_W5        "googlenet_weights/inception_3b_5x5_w"
#define N_INCT_3B_PJ_W      "googlenet_weights/inception_3b_pool_proj_w"

#define N_INCT_4A_W1        "googlenet_weights/inception_4a_1x1_w"
#define N_INCT_4A_RD_W3     "googlenet_weights/inception_4a_3x3_reduce_w"
#define N_INCT_4A_W3        "googlenet_weights/inception_4a_3x3_w"
#define N_INCT_4A_RD_W5     "googlenet_weights/inception_4a_5x5_reduce_w"
#define N_INCT_4A_W5        "googlenet_weights/inception_4a_5x5_w"
#define N_INCT_4A_PJ_W      "googlenet_weights/inception_4a_pool_proj_w"

#define N_INCT_4B_W1        "googlenet_weights/inception_4b_1x1_w"
#define N_INCT_4B_RD_W3     "googlenet_weights/inception_4b_3x3_reduce_w"
#define N_INCT_4B_W3        "googlenet_weights/inception_4b_3x3_w"
#define N_INCT_4B_RD_W5     "googlenet_weights/inception_4b_5x5_reduce_w"
#define N_INCT_4B_W5        "googlenet_weights/inception_4b_5x5_w"
#define N_INCT_4B_PJ_W      "googlenet_weights/inception_4b_pool_proj_w"

#define N_INCT_4C_W1        "googlenet_weights/inception_4c_1x1_w"
#define N_INCT_4C_RD_W3     "googlenet_weights/inception_4c_3x3_reduce_w"
#define N_INCT_4C_W3        "googlenet_weights/inception_4c_3x3_w"
#define N_INCT_4C_RD_W5     "googlenet_weights/inception_4c_5x5_reduce_w"
#define N_INCT_4C_W5        "googlenet_weights/inception_4c_5x5_w"
#define N_INCT_4C_PJ_W      "googlenet_weights/inception_4c_pool_proj_w"

#define N_INCT_4D_W1        "googlenet_weights/inception_4d_1x1_w"
#define N_INCT_4D_RD_W3     "googlenet_weights/inception_4d_3x3_reduce_w"
#define N_INCT_4D_W3        "googlenet_weights/inception_4d_3x3_w"
#define N_INCT_4D_RD_W5     "googlenet_weights/inception_4d_5x5_reduce_w"
#define N_INCT_4D_W5        "googlenet_weights/inception_4d_5x5_w"
#define N_INCT_4D_PJ_W      "googlenet_weights/inception_4d_pool_proj_w"

#define N_INCT_4E_W1        "googlenet_weights/inception_4e_1x1_w"
#define N_INCT_4E_RD_W3     "googlenet_weights/inception_4e_3x3_reduce_w"
#define N_INCT_4E_W3        "googlenet_weights/inception_4e_3x3_w"
#define N_INCT_4E_RD_W5     "googlenet_weights/inception_4e_5x5_reduce_w"
#define N_INCT_4E_W5        "googlenet_weights/inception_4e_5x5_w"
#define N_INCT_4E_PJ_W      "googlenet_weights/inception_4e_pool_proj_w"

#define N_INCT_5A_W1        "googlenet_weights/inception_5a_1x1_w"
#define N_INCT_5A_RD_W3     "googlenet_weights/inception_5a_3x3_reduce_w"
#define N_INCT_5A_W3        "googlenet_weights/inception_5a_3x3_w"
#define N_INCT_5A_RD_W5     "googlenet_weights/inception_5a_5x5_reduce_w"
#define N_INCT_5A_W5        "googlenet_weights/inception_5a_5x5_w"
#define N_INCT_5A_PJ_W      "googlenet_weights/inception_5a_pool_proj_w"

#define N_INCT_5B_W1        "googlenet_weights/inception_5b_1x1_w"
#define N_INCT_5B_RD_W3     "googlenet_weights/inception_5b_3x3_reduce_w"
#define N_INCT_5B_W3        "googlenet_weights/inception_5b_3x3_w"
#define N_INCT_5B_RD_W5     "googlenet_weights/inception_5b_5x5_reduce_w"
#define N_INCT_5B_W5        "googlenet_weights/inception_5b_5x5_w"
#define N_INCT_5B_PJ_W      "googlenet_weights/inception_5b_pool_proj_w"

#define N_FC_W06            "googlenet_weights/fc6_w"

// Googlenet bias
#define N_CONV_B01          "googlenet_weights/conv1_b"

#define N_CONV_RD_B02       "googlenet_weights/conv2_reduce_b"
#define N_CONV_B02          "googlenet_weights/conv2_b"

#define N_INCT_3A_B1        "googlenet_weights/inception_3a_1x1_b"
#define N_INCT_3A_RD_B3     "googlenet_weights/inception_3a_3x3_reduce_b"
#define N_INCT_3A_B3        "googlenet_weights/inception_3a_3x3_b"
#define N_INCT_3A_RD_B5     "googlenet_weights/inception_3a_5x5_reduce_b"
#define N_INCT_3A_B5        "googlenet_weights/inception_3a_5x5_b"
#define N_INCT_3A_PJ_B      "googlenet_weights/inception_3a_pool_proj_b"

#define N_INCT_3B_B1        "googlenet_weights/inception_3b_1x1_b"
#define N_INCT_3B_RD_B3     "googlenet_weights/inception_3b_3x3_reduce_b"
#define N_INCT_3B_B3        "googlenet_weights/inception_3b_3x3_b"
#define N_INCT_3B_RD_B5     "googlenet_weights/inception_3b_5x5_reduce_b"
#define N_INCT_3B_B5        "googlenet_weights/inception_3b_5x5_b"
#define N_INCT_3B_PJ_B      "googlenet_weights/inception_3b_pool_proj_b"

#define N_INCT_4A_B1        "googlenet_weights/inception_4a_1x1_b"
#define N_INCT_4A_RD_B3     "googlenet_weights/inception_4a_3x3_reduce_b"
#define N_INCT_4A_B3        "googlenet_weights/inception_4a_3x3_b"
#define N_INCT_4A_RD_B5     "googlenet_weights/inception_4a_5x5_reduce_b"
#define N_INCT_4A_B5        "googlenet_weights/inception_4a_5x5_b"
#define N_INCT_4A_PJ_B      "googlenet_weights/inception_4a_pool_proj_b"

#define N_INCT_4B_B1        "googlenet_weights/inception_4b_1x1_b"
#define N_INCT_4B_RD_B3     "googlenet_weights/inception_4b_3x3_reduce_b"
#define N_INCT_4B_B3        "googlenet_weights/inception_4b_3x3_b"
#define N_INCT_4B_RD_B5     "googlenet_weights/inception_4b_5x5_reduce_b"
#define N_INCT_4B_B5        "googlenet_weights/inception_4b_5x5_b"
#define N_INCT_4B_PJ_B      "googlenet_weights/inception_4b_pool_proj_b"

#define N_INCT_4C_B1        "googlenet_weights/inception_4c_1x1_b"
#define N_INCT_4C_RD_B3     "googlenet_weights/inception_4c_3x3_reduce_b"
#define N_INCT_4C_B3        "googlenet_weights/inception_4c_3x3_b"
#define N_INCT_4C_RD_B5     "googlenet_weights/inception_4c_5x5_reduce_b"
#define N_INCT_4C_B5        "googlenet_weights/inception_4c_5x5_b"
#define N_INCT_4C_PJ_B      "googlenet_weights/inception_4c_pool_proj_b"

#define N_INCT_4D_B1        "googlenet_weights/inception_4d_1x1_b"
#define N_INCT_4D_RD_B3     "googlenet_weights/inception_4d_3x3_reduce_b"
#define N_INCT_4D_B3        "googlenet_weights/inception_4d_3x3_b"
#define N_INCT_4D_RD_B5     "googlenet_weights/inception_4d_5x5_reduce_b"
#define N_INCT_4D_B5        "googlenet_weights/inception_4d_5x5_b"
#define N_INCT_4D_PJ_B      "googlenet_weights/inception_4d_pool_proj_b"

#define N_INCT_4E_B1        "googlenet_weights/inception_4e_1x1_b"
#define N_INCT_4E_RD_B3     "googlenet_weights/inception_4e_3x3_reduce_b"
#define N_INCT_4E_B3        "googlenet_weights/inception_4e_3x3_b"
#define N_INCT_4E_RD_B5     "googlenet_weights/inception_4e_5x5_reduce_b"
#define N_INCT_4E_B5        "googlenet_weights/inception_4e_5x5_b"
#define N_INCT_4E_PJ_B      "googlenet_weights/inception_4e_pool_proj_b"

#define N_INCT_5A_B1        "googlenet_weights/inception_5a_1x1_b"
#define N_INCT_5A_RD_B3     "googlenet_weights/inception_5a_3x3_reduce_b"
#define N_INCT_5A_B3        "googlenet_weights/inception_5a_3x3_b"
#define N_INCT_5A_RD_B5     "googlenet_weights/inception_5a_5x5_reduce_b"
#define N_INCT_5A_B5        "googlenet_weights/inception_5a_5x5_b"
#define N_INCT_5A_PJ_B      "googlenet_weights/inception_5a_pool_proj_b"

#define N_INCT_5B_B1        "googlenet_weights/inception_5b_1x1_b"
#define N_INCT_5B_RD_B3     "googlenet_weights/inception_5b_3x3_reduce_b"
#define N_INCT_5B_B3        "googlenet_weights/inception_5b_3x3_b"
#define N_INCT_5B_RD_B5     "googlenet_weights/inception_5b_5x5_reduce_b"
#define N_INCT_5B_B5        "googlenet_weights/inception_5b_5x5_b"
#define N_INCT_5B_PJ_B      "googlenet_weights/inception_5b_pool_proj_b"

#define N_FC_B06            "googlenet_weights/fc6_b"

// Weight length
#define L_CONV_W01          9408

#define L_CONV_RD_W02       4096
#define L_CONV_W02          110592

#define L_INCT_3A_W1        12288
#define L_INCT_3A_RD_W3     18432
#define L_INCT_3A_W3        110592
#define L_INCT_3A_RD_W5     3072
#define L_INCT_3A_W5        12800
#define L_INCT_3A_PJ_W      6144

#define L_INCT_3B_W1        32768
#define L_INCT_3B_RD_W3     32768
#define L_INCT_3B_W3        221184
#define L_INCT_3B_RD_W5     8192
#define L_INCT_3B_W5        76800
#define L_INCT_3B_PJ_W      16384

#define L_INCT_4A_W1        92160
#define L_INCT_4A_RD_W3     46080
#define L_INCT_4A_W3        179712
#define L_INCT_4A_RD_W5     7680
#define L_INCT_4A_W5        19200
#define L_INCT_4A_PJ_W      30720

#define L_INCT_4B_W1        81920
#define L_INCT_4B_RD_W3     57344
#define L_INCT_4B_W3        225792
#define L_INCT_4B_RD_W5     12288
#define L_INCT_4B_W5        38400
#define L_INCT_4B_PJ_W      32768

#define L_INCT_4C_W1        65536
#define L_INCT_4C_RD_W3     65536
#define L_INCT_4C_W3        294912
#define L_INCT_4C_RD_W5     12288
#define L_INCT_4C_W5        38400
#define L_INCT_4C_PJ_W      32768

#define L_INCT_4D_W1        57344
#define L_INCT_4D_RD_W3     73728
#define L_INCT_4D_W3        373248
#define L_INCT_4D_RD_W5     16384
#define L_INCT_4D_W5        51200
#define L_INCT_4D_PJ_W      32768

#define L_INCT_4E_W1        135168
#define L_INCT_4E_RD_W3     84480
#define L_INCT_4E_W3        460800
#define L_INCT_4E_RD_W5     16896
#define L_INCT_4E_W5        102400
#define L_INCT_4E_PJ_W      67584

#define L_INCT_5A_W1        212992
#define L_INCT_5A_RD_W3     133120
#define L_INCT_5A_W3        460800
#define L_INCT_5A_RD_W5     26624
#define L_INCT_5A_W5        102400
#define L_INCT_5A_PJ_W      106496

#define L_INCT_5B_W1        319488
#define L_INCT_5B_RD_W3     159744
#define L_INCT_5B_W3        663552
#define L_INCT_5B_RD_W5     39936
#define L_INCT_5B_W5        153600
#define L_INCT_5B_PJ_W      106496

#define L_FC_W06            1024000

// Bias length
#define L_CONV_B01          64

#define L_CONV_RD_B02       64
#define L_CONV_B02          192

#define L_INCT_3A_B1        64
#define L_INCT_3A_RD_B3     96
#define L_INCT_3A_B3        128
#define L_INCT_3A_RD_B5     16
#define L_INCT_3A_B5        32
#define L_INCT_3A_PJ_B      32

#define L_INCT_3B_B1        128
#define L_INCT_3B_RD_B3     128
#define L_INCT_3B_B3        192
#define L_INCT_3B_RD_B5     32
#define L_INCT_3B_B5        96
#define L_INCT_3B_PJ_B      64

#define L_INCT_4A_B1        192
#define L_INCT_4A_RD_B3     96
#define L_INCT_4A_B3        208
#define L_INCT_4A_RD_B5     16
#define L_INCT_4A_B5        48
#define L_INCT_4A_PJ_B      64

#define L_INCT_4B_B1        160
#define L_INCT_4B_RD_B3     112
#define L_INCT_4B_B3        224
#define L_INCT_4B_RD_B5     24
#define L_INCT_4B_B5        64
#define L_INCT_4B_PJ_B      64

#define L_INCT_4C_B1        128
#define L_INCT_4C_RD_B3     128
#define L_INCT_4C_B3        256
#define L_INCT_4C_RD_B5     24
#define L_INCT_4C_B5        64
#define L_INCT_4C_PJ_B      64

#define L_INCT_4D_B1        112
#define L_INCT_4D_RD_B3     144
#define L_INCT_4D_B3        288
#define L_INCT_4D_RD_B5     32
#define L_INCT_4D_B5        64
#define L_INCT_4D_PJ_B      64

#define L_INCT_4E_B1        256
#define L_INCT_4E_RD_B3     160
#define L_INCT_4E_B3        320
#define L_INCT_4E_RD_B5     32
#define L_INCT_4E_B5        128
#define L_INCT_4E_PJ_B      128

#define L_INCT_5A_B1        256
#define L_INCT_5A_RD_B3     160
#define L_INCT_5A_B3        320
#define L_INCT_5A_RD_B5     32
#define L_INCT_5A_B5        128
#define L_INCT_5A_PJ_B      128

#define L_INCT_5B_B1        384
#define L_INCT_5B_RD_B3     192
#define L_INCT_5B_B3        384
#define L_INCT_5B_RD_B5     48
#define L_INCT_5B_B5        128
#define L_INCT_5B_PJ_B      128

#define L_FC_B06            1000

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
    
    layer conv1;
    layer conv_rd2, conv2;
    
    layer inc3a_1x1, inc3a_3x3_rd, inc3a_3x3, inc3a_5x5_rd, inc3a_5x5, inc3a_pool, inc3a_proj, inc3a_concat;
    layer inc3b_1x1, inc3b_3x3_rd, inc3b_3x3, inc3b_5x5_rd, inc3b_5x5, inc3b_pool, inc3b_proj, inc3b_concat;
    
    layer inc4a_1x1, inc4a_3x3_rd, inc4a_3x3, inc4a_5x5_rd, inc4a_5x5, inc4a_pool, inc4a_proj, inc4a_concat;
    layer inc4b_1x1, inc4b_3x3_rd, inc4b_3x3, inc4b_5x5_rd, inc4b_5x5, inc4b_pool, inc4b_proj, inc4b_concat;
    layer inc4c_1x1, inc4c_3x3_rd, inc4c_3x3, inc4c_5x5_rd, inc4c_5x5, inc4c_pool, inc4c_proj, inc4c_concat;
    layer inc4d_1x1, inc4d_3x3_rd, inc4d_3x3, inc4d_5x5_rd, inc4d_5x5, inc4d_pool, inc4d_proj, inc4d_concat;
    layer inc4e_1x1, inc4e_3x3_rd, inc4e_3x3, inc4e_5x5_rd, inc4e_5x5, inc4e_pool, inc4e_proj, inc4e_concat;
    
    layer inc5a_1x1, inc5a_3x3_rd, inc5a_3x3, inc5a_5x5_rd, inc5a_5x5, inc5a_pool, inc5a_proj, inc5a_concat;
    layer inc5b_1x1, inc5b_3x3_rd, inc5b_3x3, inc5b_5x5_rd, inc5b_5x5, inc5b_pool, inc5b_proj, inc5b_concat;
    
    layer fc6;
    layer pool1, pool2, pool3, pool4, pool5;
	layer norm1, norm2;
    
    float *weight = NULL, *bias = NULL;
	
	/* network size and input names */
	const char *input_names[BATCH_SIZE] = {N_INPUT_BICYCLE, N_INPUT_CAT, N_INPUT_CHEETAH, N_INPUT_FOX};
	input.width = IMG_WIDTH;
	input.channel = IMG_CHANNEL;
	input.batch = BATCH_SIZE;
	input.data = malloc(input.batch * sizeof(float *));
	
    /* conv1 */
    load_input(&input, input_names);
		//debug_print_input(input, "goog_input");
    load_file(&weight, N_CONV_W01, L_CONV_W01);
		//debug_print_weight(weight, 96, 3, 11, "sim_w1");
	load_file(&bias,   N_CONV_B01, L_CONV_B01);
		//debug_print_bias(bias, 96, "sim_b1");
    conv1 = conv_layer(input, 64, 7, 2, 3, INPUT_FREE, weight, bias);
		//debug_print_input(conv1, "goog_conv1");
    
    /* pool1 */
	pool1 = max_pooling_layer(conv1, 3, 2, 0, INPUT_FREE);
		//debug_print_input(pool1, "goog_pool1");
        
	/* norm1 */
	norm1 = norm_lrn(pool1, 1.0f, 5, 0.0001, 0.75, INPUT_FREE);
		//debug_print_input(norm1, "goog_norm1");
        
    /* conv2 */
    load_file(&weight, N_CONV_RD_W02, L_CONV_RD_W02);
	load_file(&bias,   N_CONV_RD_B02, L_CONV_RD_B02);
    conv_rd2 = conv_layer(norm1, 64, 1, 1, 0, INPUT_FREE, weight, bias);
        //debug_print_input(conv_rd2, "goog_conv2_rd");
    
    load_file(&weight, N_CONV_W02, L_CONV_W02);
	load_file(&bias,   N_CONV_B02, L_CONV_B02);
    conv2 = conv_layer(conv_rd2, 192, 3, 1, 1, INPUT_FREE, weight, bias);
		//debug_print_input(conv2, "goog_conv2");
        
    /* norm2 */
	norm2 = norm_lrn(conv2, 1.0f, 5, 0.0001, 0.75, INPUT_FREE);
		//debug_print_input(norm2, "goog_norm2");
    
    /* pool2 */
	pool2 = max_pooling_layer(norm2, 3, 2, 0, INPUT_FREE);
		debug_print_input(pool2, "goog_pool2");
        
    /* inception 3a */
    printf("Inception module 3a\n");
    load_file(&weight, N_INCT_3A_W1, L_INCT_3A_W1);
	load_file(&bias,   N_INCT_3A_B1, L_INCT_3A_B1);
    inc3a_1x1 = conv_layer(pool2, 64, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc3a_1x1, "goog_inc3a_1x1");
    
    load_file(&weight, N_INCT_3A_RD_W3, L_INCT_3A_RD_W3);
	load_file(&bias,   N_INCT_3A_RD_B3, L_INCT_3A_RD_B3);
    inc3a_3x3_rd = conv_layer(pool2, 96, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc3a_3x3_rd, "goog_inc3a_3x3_rd");
        
    load_file(&weight, N_INCT_3A_W3, L_INCT_3A_W3);
	load_file(&bias,   N_INCT_3A_B3, L_INCT_3A_B3);
    inc3a_3x3 = conv_layer(inc3a_3x3_rd, 128, 3, 1, 1, INPUT_FREE, weight, bias);
		//debug_print_input(inc3a_3x3, "goog_inc3a_3x3");
        
    load_file(&weight, N_INCT_3A_RD_W5, L_INCT_3A_RD_W5);
	load_file(&bias,   N_INCT_3A_RD_B5, L_INCT_3A_RD_B5);
    inc3a_5x5_rd = conv_layer(pool2, 16, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc3a_5x5_rd, "goog_inc3a_5x5_rd");
        
    load_file(&weight, N_INCT_3A_W5, L_INCT_3A_W5);
	load_file(&bias,   N_INCT_3A_B5, L_INCT_3A_B5);
    inc3a_5x5 = conv_layer(inc3a_5x5_rd, 32, 5, 1, 2, INPUT_FREE, weight, bias);
		//debug_print_input(inc3a_5x5, "goog_inc3a_5x5");
        
    inc3a_pool = max_pooling_layer(pool2, 3, 1, 1, INPUT_FREE); 
        //debug_print_input(inc3a_pool, "goog_inc3a_pool");

    load_file(&weight, N_INCT_3A_PJ_W, L_INCT_3A_PJ_W);
	load_file(&bias,   N_INCT_3A_PJ_B, L_INCT_3A_PJ_B);
    inc3a_proj = conv_layer(inc3a_pool, 32, 1, 1, 0, INPUT_FREE, weight, bias);
		//debug_print_input(inc3a_proj, "goog_inc3a_proj");    

    inc3a_concat = concatenate_4layer(inc3a_1x1, inc3a_3x3, inc3a_5x5, inc3a_proj, INPUT_FREE);
        //debug_print_input(inc3a_concat, "goog_inc3a_concat");    
    
    /* inception 3b */
    printf("Inception module 3b\n");
    load_file(&weight, N_INCT_3B_W1, L_INCT_3B_W1);
	load_file(&bias,   N_INCT_3B_B1, L_INCT_3B_B1);
    inc3b_1x1 = conv_layer(inc3a_concat, 128, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc3b_1x1, "goog_inc3b_1x1");
    
    load_file(&weight, N_INCT_3B_RD_W3, L_INCT_3B_RD_W3);
	load_file(&bias,   N_INCT_3B_RD_B3, L_INCT_3B_RD_B3);
    inc3b_3x3_rd = conv_layer(inc3a_concat, 128, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc3b_3x3_rd, "goog_inc3b_3x3_rd");
        
    load_file(&weight, N_INCT_3B_W3, L_INCT_3B_W3);
	load_file(&bias,   N_INCT_3B_B3, L_INCT_3B_B3);
    inc3b_3x3 = conv_layer(inc3b_3x3_rd, 192, 3, 1, 1, INPUT_FREE, weight, bias);
		//debug_print_input(inc3b_3x3, "goog_inc3b_3x3");
        
    load_file(&weight, N_INCT_3B_RD_W5, L_INCT_3B_RD_W5);
	load_file(&bias,   N_INCT_3B_RD_B5, L_INCT_3B_RD_B5);
    inc3b_5x5_rd = conv_layer(inc3a_concat, 32, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc3b_5x5_rd, "goog_inc3b_5x5_rd");
        
    load_file(&weight, N_INCT_3B_W5, L_INCT_3B_W5);
	load_file(&bias,   N_INCT_3B_B5, L_INCT_3B_B5);
    inc3b_5x5 = conv_layer(inc3b_5x5_rd, 96, 5, 1, 2, INPUT_FREE, weight, bias);
		//debug_print_input(inc3b_5x5, "goog_inc3b_5x5");
        
    inc3b_pool = max_pooling_layer(inc3a_concat, 3, 1, 1, INPUT_FREE); 
        //debug_print_input(inc3b_pool, "goog_inc3b_pool");

    load_file(&weight, N_INCT_3B_PJ_W, L_INCT_3B_PJ_W);
	load_file(&bias,   N_INCT_3B_PJ_B, L_INCT_3B_PJ_B);
    inc3b_proj = conv_layer(inc3b_pool, 64, 1, 1, 0, INPUT_FREE, weight, bias);
		//debug_print_input(inc3b_proj, "goog_inc3b_proj");    

    inc3b_concat = concatenate_4layer(inc3b_1x1, inc3b_3x3, inc3b_5x5, inc3b_proj, INPUT_FREE);
        //debug_print_input(inc3b_concat, "goog_inc3b_concat");    
    
    /* pool3 */
	pool3 = max_pooling_layer(inc3b_concat, 3, 2, 0, INPUT_FREE);
		//debug_print_input(pool3, "goog_pool3");
    
    /* inception 4a */
    printf("Inception module 4a\n");
    load_file(&weight, N_INCT_4A_W1, L_INCT_4A_W1);
	load_file(&bias,   N_INCT_4A_B1, L_INCT_4A_B1);
    inc4a_1x1 = conv_layer(pool3, 192, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc4a_1x1, "goog_inc4a_1x1");
    
    load_file(&weight, N_INCT_4A_RD_W3, L_INCT_4A_RD_W3);
	load_file(&bias,   N_INCT_4A_RD_B3, L_INCT_4A_RD_B3);
    inc4a_3x3_rd = conv_layer(pool3, 96, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc4a_3x3_rd, "goog_inc4a_3x3_rd");
        
    load_file(&weight, N_INCT_4A_W3, L_INCT_4A_W3);
	load_file(&bias,   N_INCT_4A_B3, L_INCT_4A_B3);
    inc4a_3x3 = conv_layer(inc4a_3x3_rd, 208, 3, 1, 1, INPUT_FREE, weight, bias);
		//debug_print_input(inc4a_3x3, "goog_inc4a_3x3");
        
    load_file(&weight, N_INCT_4A_RD_W5, L_INCT_4A_RD_W5);
	load_file(&bias,   N_INCT_4A_RD_B5, L_INCT_4A_RD_B5);
    inc4a_5x5_rd = conv_layer(pool3, 16, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc4a_5x5_rd, "goog_inc4a_5x5_rd");
        
    load_file(&weight, N_INCT_4A_W5, L_INCT_4A_W5);
	load_file(&bias,   N_INCT_4A_B5, L_INCT_4A_B5);
    inc4a_5x5 = conv_layer(inc4a_5x5_rd, 48, 5, 1, 2, INPUT_FREE, weight, bias);
		//debug_print_input(inc4a_5x5, "goog_inc4a_5x5");
        
    inc4a_pool = max_pooling_layer(pool3, 3, 1, 1, INPUT_FREE); 
        //debug_print_input(inc4a_pool, "goog_inc4a_pool");

    load_file(&weight, N_INCT_4A_PJ_W, L_INCT_4A_PJ_W);
	load_file(&bias,   N_INCT_4A_PJ_B, L_INCT_4A_PJ_B);
    inc4a_proj = conv_layer(inc4a_pool, 64, 1, 1, 0, INPUT_FREE, weight, bias);
		//debug_print_input(inc4a_proj, "goog_inc4a_proj");    

    inc4a_concat = concatenate_4layer(inc4a_1x1, inc4a_3x3, inc4a_5x5, inc4a_proj, INPUT_FREE);
        //debug_print_input(inc4a_concat, "goog_inc4a_concat");
    
    /* inception 4b */
    printf("Inception module 4b\n");
    load_file(&weight, N_INCT_4B_W1, L_INCT_4B_W1);
	load_file(&bias,   N_INCT_4B_B1, L_INCT_4B_B1);
    inc4b_1x1 = conv_layer(inc4a_concat, 160, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc4b_1x1, "goog_inc4b_1x1");
    
    load_file(&weight, N_INCT_4B_RD_W3, L_INCT_4B_RD_W3);
        //debug_print_weight(weight, 112, 512, 1, "debugw");
	load_file(&bias,   N_INCT_4B_RD_B3, L_INCT_4B_RD_B3);
        //debug_print_bias(bias, 112, "debugb");
    inc4b_3x3_rd = conv_layer(inc4a_concat, 112, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc4b_3x3_rd, "goog_inc4b_3x3_rd");
        
    load_file(&weight, N_INCT_4B_W3, L_INCT_4B_W3);
	load_file(&bias,   N_INCT_4B_B3, L_INCT_4B_B3);
    inc4b_3x3 = conv_layer(inc4b_3x3_rd, 224, 3, 1, 1, INPUT_FREE, weight, bias);
		//debug_print_input(inc4b_3x3, "goog_inc4b_3x3");
        
    load_file(&weight, N_INCT_4B_RD_W5, L_INCT_4B_RD_W5);
	load_file(&bias,   N_INCT_4B_RD_B5, L_INCT_4B_RD_B5);
    inc4b_5x5_rd = conv_layer(inc4a_concat, 24, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc4b_5x5_rd, "goog_inc4b_5x5_rd");
        
    load_file(&weight, N_INCT_4B_W5, L_INCT_4B_W5);
	load_file(&bias,   N_INCT_4B_B5, L_INCT_4B_B5);
    inc4b_5x5 = conv_layer(inc4b_5x5_rd, 64, 5, 1, 2, INPUT_FREE, weight, bias);
		//debug_print_input(inc4b_5x5, "goog_inc4b_5x5");
        
    inc4b_pool = max_pooling_layer(inc4a_concat, 3, 1, 1, INPUT_FREE); 
        //debug_print_input(inc4b_pool, "goog_inc4b_pool");

    load_file(&weight, N_INCT_4B_PJ_W, L_INCT_4B_PJ_W);
	load_file(&bias,   N_INCT_4B_PJ_B, L_INCT_4B_PJ_B);
    inc4b_proj = conv_layer(inc4b_pool, 64, 1, 1, 0, INPUT_FREE, weight, bias);
		//debug_print_input(inc4b_proj, "goog_inc4b_proj");    

    inc4b_concat = concatenate_4layer(inc4b_1x1, inc4b_3x3, inc4b_5x5, inc4b_proj, INPUT_FREE);
        //debug_print_input(inc4b_concat, "goog_inc4b_concat");
    
    /* inception 4c */
    printf("Inception module 4c\n");
    load_file(&weight, N_INCT_4C_W1, L_INCT_4C_W1);
	load_file(&bias,   N_INCT_4C_B1, L_INCT_4C_B1);
    inc4c_1x1 = conv_layer(inc4b_concat, 128, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc4c_1x1, "goog_inc4c_1x1");
    
    load_file(&weight, N_INCT_4C_RD_W3, L_INCT_4C_RD_W3);
	load_file(&bias,   N_INCT_4C_RD_B3, L_INCT_4C_RD_B3);
    inc4c_3x3_rd = conv_layer(inc4b_concat, 128, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc4c_3x3_rd, "goog_inc4c_3x3_rd");
        
    load_file(&weight, N_INCT_4C_W3, L_INCT_4C_W3);
	load_file(&bias,   N_INCT_4C_B3, L_INCT_4C_B3);
    inc4c_3x3 = conv_layer(inc4c_3x3_rd, 256, 3, 1, 1, INPUT_FREE, weight, bias);
		//debug_print_input(inc4c_3x3, "goog_inc4c_3x3");
        
    load_file(&weight, N_INCT_4C_RD_W5, L_INCT_4C_RD_W5);
	load_file(&bias,   N_INCT_4C_RD_B5, L_INCT_4C_RD_B5);
    inc4c_5x5_rd = conv_layer(inc4b_concat, 24, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc4c_5x5_rd, "goog_inc4c_5x5_rd");
        
    load_file(&weight, N_INCT_4C_W5, L_INCT_4C_W5);
	load_file(&bias,   N_INCT_4C_B5, L_INCT_4C_B5);
    inc4c_5x5 = conv_layer(inc4c_5x5_rd, 64, 5, 1, 2, INPUT_FREE, weight, bias);
		//debug_print_input(inc4c_5x5, "goog_inc4c_5x5");
        
    inc4c_pool = max_pooling_layer(inc4b_concat, 3, 1, 1, INPUT_FREE); 
        //debug_print_input(inc4c_pool, "goog_inc4c_pool");

    load_file(&weight, N_INCT_4C_PJ_W, L_INCT_4C_PJ_W);
	load_file(&bias,   N_INCT_4C_PJ_B, L_INCT_4C_PJ_B);
    inc4c_proj = conv_layer(inc4c_pool, 64, 1, 1, 0, INPUT_FREE, weight, bias);
		//debug_print_input(inc4c_proj, "goog_inc4c_proj");    

    inc4c_concat = concatenate_4layer(inc4c_1x1, inc4c_3x3, inc4c_5x5, inc4c_proj, INPUT_FREE);
        //debug_print_input(inc4c_concat, "goog_inc4c_concat");
    
    /* inception 4d */
    printf("Inception module 4d\n");
    load_file(&weight, N_INCT_4D_W1, L_INCT_4D_W1);
	load_file(&bias,   N_INCT_4D_B1, L_INCT_4D_B1);
    inc4d_1x1 = conv_layer(inc4c_concat, 112, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc4d_1x1, "goog_inc4d_1x1");
    
    load_file(&weight, N_INCT_4D_RD_W3, L_INCT_4D_RD_W3);
	load_file(&bias,   N_INCT_4D_RD_B3, L_INCT_4D_RD_B3);
    inc4d_3x3_rd = conv_layer(inc4c_concat, 144, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc4d_3x3_rd, "goog_inc4d_3x3_rd");
        
    load_file(&weight, N_INCT_4D_W3, L_INCT_4D_W3);
	load_file(&bias,   N_INCT_4D_B3, L_INCT_4D_B3);
    inc4d_3x3 = conv_layer(inc4d_3x3_rd, 288, 3, 1, 1, INPUT_FREE, weight, bias);
		//debug_print_input(inc4d_3x3, "goog_inc4d_3x3");
        
    load_file(&weight, N_INCT_4D_RD_W5, L_INCT_4D_RD_W5);
	load_file(&bias,   N_INCT_4D_RD_B5, L_INCT_4D_RD_B5);
    inc4d_5x5_rd = conv_layer(inc4c_concat, 32, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc4d_5x5_rd, "goog_inc4d_5x5_rd");
        
    load_file(&weight, N_INCT_4D_W5, L_INCT_4D_W5);
	load_file(&bias,   N_INCT_4D_B5, L_INCT_4D_B5);
    inc4d_5x5 = conv_layer(inc4d_5x5_rd, 64, 5, 1, 2, INPUT_FREE, weight, bias);
		//debug_print_input(inc4d_5x5, "goog_inc4d_5x5");
        
    inc4d_pool = max_pooling_layer(inc4c_concat, 3, 1, 1, INPUT_FREE); 
        //debug_print_input(inc4d_pool, "goog_inc4d_pool");

    load_file(&weight, N_INCT_4D_PJ_W, L_INCT_4D_PJ_W);
	load_file(&bias,   N_INCT_4D_PJ_B, L_INCT_4D_PJ_B);
    inc4d_proj = conv_layer(inc4d_pool, 64, 1, 1, 0, INPUT_FREE, weight, bias);
		//debug_print_input(inc4d_proj, "goog_inc4d_proj");    

    inc4d_concat = concatenate_4layer(inc4d_1x1, inc4d_3x3, inc4d_5x5, inc4d_proj, INPUT_FREE);
        //debug_print_input(inc4d_concat, "goog_inc4d_concat");
    
    /* inception 4e */
    printf("Inception module 4e\n");
    load_file(&weight, N_INCT_4E_W1, L_INCT_4E_W1);
	load_file(&bias,   N_INCT_4E_B1, L_INCT_4E_B1);
    inc4e_1x1 = conv_layer(inc4d_concat, 256, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc4e_1x1, "goog_inc4e_1x1");
    
    load_file(&weight, N_INCT_4E_RD_W3, L_INCT_4E_RD_W3);
	load_file(&bias,   N_INCT_4E_RD_B3, L_INCT_4E_RD_B3);
    inc4e_3x3_rd = conv_layer(inc4d_concat, 160, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc4e_3x3_rd, "goog_inc4e_3x3_rd");
        
    load_file(&weight, N_INCT_4E_W3, L_INCT_4E_W3);
	load_file(&bias,   N_INCT_4E_B3, L_INCT_4E_B3);
    inc4e_3x3 = conv_layer(inc4e_3x3_rd, 320, 3, 1, 1, INPUT_FREE, weight, bias);
		//debug_print_input(inc4e_3x3, "goog_inc4e_3x3");
        
    load_file(&weight, N_INCT_4E_RD_W5, L_INCT_4E_RD_W5);
	load_file(&bias,   N_INCT_4E_RD_B5, L_INCT_4E_RD_B5);
    inc4e_5x5_rd = conv_layer(inc4d_concat, 32, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc4e_5x5_rd, "goog_inc4e_5x5_rd");
        
    load_file(&weight, N_INCT_4E_W5, L_INCT_4E_W5);
	load_file(&bias,   N_INCT_4E_B5, L_INCT_4E_B5);
    inc4e_5x5 = conv_layer(inc4e_5x5_rd, 128, 5, 1, 2, INPUT_FREE, weight, bias);
		//debug_print_input(inc4e_5x5, "goog_inc4e_5x5");
        
    inc4e_pool = max_pooling_layer(inc4d_concat, 3, 1, 1, INPUT_FREE); 
        //debug_print_input(inc4e_pool, "goog_inc4e_pool");

    load_file(&weight, N_INCT_4E_PJ_W, L_INCT_4E_PJ_W);
	load_file(&bias,   N_INCT_4E_PJ_B, L_INCT_4E_PJ_B);
    inc4e_proj = conv_layer(inc4e_pool, 128, 1, 1, 0, INPUT_FREE, weight, bias);
		//debug_print_input(inc4e_proj, "goog_inc4e_proj");    

    inc4e_concat = concatenate_4layer(inc4e_1x1, inc4e_3x3, inc4e_5x5, inc4e_proj, INPUT_FREE);
        //debug_print_input(inc4e_concat, "goog_inc4e_concat");
    
    /* pool4 */
	pool4 = max_pooling_layer(inc4e_concat, 3, 2, 0, INPUT_FREE);
		//debug_print_input(pool4, "goog_pool4");
    
    /* inception 5a */
    printf("Inception module 5a\n");
    load_file(&weight, N_INCT_5A_W1, L_INCT_5A_W1);
	load_file(&bias,   N_INCT_5A_B1, L_INCT_5A_B1);
    inc5a_1x1 = conv_layer(pool4, 256, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc5a_1x1, "goog_inc5a_1x1");
    
    load_file(&weight, N_INCT_5A_RD_W3, L_INCT_5A_RD_W3);
	load_file(&bias,   N_INCT_5A_RD_B3, L_INCT_5A_RD_B3);
    inc5a_3x3_rd = conv_layer(pool4, 160, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc5a_3x3_rd, "goog_inc5a_3x3_rd");
        
    load_file(&weight, N_INCT_5A_W3, L_INCT_5A_W3);
	load_file(&bias,   N_INCT_5A_B3, L_INCT_5A_B3);
    inc5a_3x3 = conv_layer(inc5a_3x3_rd, 320, 3, 1, 1, INPUT_FREE, weight, bias);
		//debug_print_input(inc5a_3x3, "goog_inc5a_3x3");
        
    load_file(&weight, N_INCT_5A_RD_W5, L_INCT_5A_RD_W5);
	load_file(&bias,   N_INCT_5A_RD_B5, L_INCT_5A_RD_B5);
    inc5a_5x5_rd = conv_layer(pool4, 32, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc5a_5x5_rd, "goog_inc5a_5x5_rd");
        
    load_file(&weight, N_INCT_5A_W5, L_INCT_5A_W5);
	load_file(&bias,   N_INCT_5A_B5, L_INCT_5A_B5);
    inc5a_5x5 = conv_layer(inc5a_5x5_rd, 128, 5, 1, 2, INPUT_FREE, weight, bias);
		//debug_print_input(inc5a_5x5, "goog_inc5a_5x5");
        
    inc5a_pool = max_pooling_layer(pool4, 3, 1, 1, INPUT_FREE); 
        //debug_print_input(inc4a_pool, "goog_inc5a_pool");

    load_file(&weight, N_INCT_5A_PJ_W, L_INCT_5A_PJ_W);
	load_file(&bias,   N_INCT_5A_PJ_B, L_INCT_5A_PJ_B);
    inc5a_proj = conv_layer(inc5a_pool, 128, 1, 1, 0, INPUT_FREE, weight, bias);
		//debug_print_input(inc5a_proj, "goog_inc5a_proj");    

    inc5a_concat = concatenate_4layer(inc5a_1x1, inc5a_3x3, inc5a_5x5, inc5a_proj, INPUT_FREE);
        //debug_print_input(inc5a_concat, "goog_inc5a_concat");
    
    /* inception 5b */
    printf("Inception module 5b\n");
    load_file(&weight, N_INCT_5B_W1, L_INCT_5B_W1);
	load_file(&bias,   N_INCT_5B_B1, L_INCT_5B_B1);
    inc5b_1x1 = conv_layer(inc5a_concat, 384, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc5b_1x1, "goog_inc5b_1x1");
    
    load_file(&weight, N_INCT_5B_RD_W3, L_INCT_5B_RD_W3);
	load_file(&bias,   N_INCT_5B_RD_B3, L_INCT_5B_RD_B3);
    inc5b_3x3_rd = conv_layer(inc5a_concat, 192, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc5b_3x3_rd, "goog_inc5b_3x3_rd");
        
    load_file(&weight, N_INCT_5B_W3, L_INCT_5B_W3);
	load_file(&bias,   N_INCT_5B_B3, L_INCT_5B_B3);
    inc5b_3x3 = conv_layer(inc5b_3x3_rd, 384, 3, 1, 1, INPUT_FREE, weight, bias);
		//debug_print_input(inc5b_3x3, "goog_inc5b_3x3");
        
    load_file(&weight, N_INCT_5B_RD_W5, L_INCT_5B_RD_W5);
	load_file(&bias,   N_INCT_5B_RD_B5, L_INCT_5B_RD_B5);
    inc5b_5x5_rd = conv_layer(inc5a_concat, 48, 1, 1, 0, INPUT_STAY, weight, bias);
		//debug_print_input(inc5b_5x5_rd, "goog_inc5b_5x5_rd");
        
    load_file(&weight, N_INCT_5B_W5, L_INCT_5B_W5);
	load_file(&bias,   N_INCT_5B_B5, L_INCT_5B_B5);
    inc5b_5x5 = conv_layer(inc5b_5x5_rd, 128, 5, 1, 2, INPUT_FREE, weight, bias);
		//debug_print_input(inc5b_5x5, "goog_inc5b_5x5");
        
    inc5b_pool = max_pooling_layer(inc5a_concat, 3, 1, 1, INPUT_FREE); 
        //debug_print_input(inc5b_pool, "goog_inc5b_pool");

    load_file(&weight, N_INCT_5B_PJ_W, L_INCT_5B_PJ_W);
	load_file(&bias,   N_INCT_5B_PJ_B, L_INCT_5B_PJ_B);
    inc5b_proj = conv_layer(inc5b_pool, 128, 1, 1, 0, INPUT_FREE, weight, bias);
		//debug_print_input(inc5b_proj, "goog_inc5b_proj");    

    inc5b_concat = concatenate_4layer(inc5b_1x1, inc5b_3x3, inc5b_5x5, inc5b_proj, INPUT_FREE);
        //debug_print_input(inc5b_concat, "goog_inc5b_concat");
    
    /* pool5 */
    pool5 = avg_pooling_layer(inc5b_concat, 7, 1, 0, INPUT_FREE);
        //debug_print_input(pool5, "goog_pool5");
    
    /* fc6 */
    load_file(&weight, N_FC_W06, L_FC_W06);
	load_file(&bias, N_FC_B06, L_FC_B06);
    fc6 = fc_layer(pool5, 1000, INPUT_FREE, weight, bias, 1);
        //debug_print_input(fc6, "goog_fc6");
    
    /* softmax */
    printf("softmax : %d\n", fc6.channel);
    softmax(fc6, 1000, input_names, LABEL_FILE_NAME);
    
	return 0;
}

void load_input(layer *input, const char *file_name[]){
	int batch;
	int file_length; 
	FILE *fp;
	file_length = input->channel * input->width * input->width;
	
	for(batch=0; batch<input->batch; batch++) {
		fp = fopen(file_name[batch], "r");
		input->data[batch] = (float *)malloc(file_length*sizeof(float));
		printf("batch %d, file_length %d file_name %s \n", batch, file_length, file_name[batch]);
		fread(input->data[batch], sizeof(float), file_length, fp);
		fclose(fp);
	}
}

void load_file(float **w_input, const char *file_name, int f_length) {
    int i;
    FILE *fp;
    if(*w_input == NULL)
        *w_input = (float *)malloc(f_length*sizeof(float));
    else
        *w_input = (float *)realloc(*w_input, f_length*sizeof(float));
	fp = fopen(file_name, "r");
	fread(*w_input, sizeof(float), f_length, fp);
	fclose(fp);
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
		printf("-------------Prediction by GoogLeNet for %s-------------\n", file_name[batch]);
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
	int o_index, o_channel_index, o_row_index;

	float sum;
    layer output;
	printf("conv : %d * %d * %d\n", input.width, input.width, input.channel);

	layer z_pad = zero_pad_layer(input, pad);
		//printf("\tpad : %d * %d * %d\n", z_pad.width, z_pad.width, z_pad.channel);

	output.width = (z_pad.width + stride - f_size) / stride;
	output.channel = filter;
	output.batch = z_pad.batch;
	output.data = malloc(z_pad.batch * sizeof(float *));
	
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
		printf("aaaa\n");
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
