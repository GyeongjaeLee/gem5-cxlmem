#include <stdio.h>
#include <stdlib.h>

#include "input.h"
#include "conv_layer.h"
#include "pool_layer.h"
#include "fc_layer.h"
//#include "debug.h"

//Input and weight definition
#define NAME_INPUT "b_weights/input.bin"
#define NAME_CONV_W01 "b_weights/b_conv_w1"
#define NAME_CONV_W02 "b_weights/b_conv_w2"
#define NAME_CONV_W03 "b_weights/b_conv_w3"
#define NAME_CONV_W04 "b_weights/b_conv_w4"
#define NAME_CONV_W05 "b_weights/b_conv_w5"

#define NAME_FC_W01 "b_weights/b_fc_w1"
#define NAME_FC_W02 "b_weights/b_fc_w2"
#define NAME_FC_W03 "b_weights/b_fc_w3"

#define NAME_CONV_B01 "b_weights/b_conv_b1"
#define NAME_CONV_B02 "b_weights/b_conv_b2"
#define NAME_CONV_B03 "b_weights/b_conv_b3"
#define NAME_CONV_B04 "b_weights/b_conv_b4"
#define NAME_CONV_B05 "b_weights/b_conv_b5"

#define NAME_FC_B01 "b_weights/b_fc_b1"
#define NAME_FC_B02 "b_weights/b_fc_b2"
#define NAME_FC_B03 "b_weights/b_fc_b3"

//Length definition
#define LENGTH_INPUT 154587
#define LENGTH_CONV_W01 34848
#define LENGTH_CONV_W02 614400
#define LENGTH_CONV_W03 884736
#define LENGTH_CONV_W04 1327104
#define LENGTH_CONV_W05 884736

#define LENGTH_FC_W01 150994944
#define LENGTH_FC_W02 67108864
#define LENGTH_FC_W03 16384000

#define LENGTH_CONV_B01 96
#define LENGTH_CONV_B02 256
#define LENGTH_CONV_B03 384
#define LENGTH_CONV_B04 384
#define LENGTH_CONV_B05 256

#define LENGTH_FC_B01 4096
#define LENGTH_FC_B02 4096
#define LENGTH_FC_B03 1000

int main() {
	
	layer input;
    layer conv1, conv2, conv3, conv4, conv5;
    layer pool1, pool2, pool5;
    layer fc6, fc7, fc8;
    float *weight = NULL, *bias = NULL;
	input.width = 227;
	input.channel = 3;
    input.data = NULL;
	
    /* conv1 */
    load_file(&input.data, NAME_INPUT, LENGTH_INPUT);
		//debug_print_input(input, "layer_00");
    load_file(&weight, NAME_CONV_W01, LENGTH_CONV_W01);
		//debug_print_weight(weight, 96, 3, 11, "sim_w1");
	load_file(&bias, NAME_CONV_B01, LENGTH_CONV_B01);
		//debug_print_bias(bias, 96, "sim_b1");
    conv1 = conv_layer(input, 96, 11, 4, 0, weight, bias);
		//debug_print_input(conv1, "layer_01");
    
    /* pool1 */
	pool1 = max_pooling_layer(conv1, 3, 2);
		//debug_print_input(pool1, "layer_02");

    /* conv2 */
    load_file(&weight, NAME_CONV_W02, LENGTH_CONV_W02);
	load_file(&bias, NAME_CONV_B02, LENGTH_CONV_B02);
    conv2 = conv_layer(pool1, 256, 5, 1, 2, weight, bias);
		//debug_print_input(conv2, "layer_03");
    
    /* pool2 */
	pool2 = max_pooling_layer(conv2, 3, 2);
		//debug_print_input(pool2, "layer_04");
    
    /* conv3 */
    load_file(&weight, NAME_CONV_W03, LENGTH_CONV_W03);
		//debug_print_weight(weight, 384, 256, 3, "conv_w3");
	load_file(&bias, NAME_CONV_B03, LENGTH_CONV_B03);
		//debug_print_bias(bias, 384, "conv_b3");
    conv3 = conv_layer(pool2, 384, 3, 1, 1, weight, bias);
		//debug_print_input(conv3, "layer_05");
    
    /* conv4 */
    load_file(&weight, NAME_CONV_W04, LENGTH_CONV_W04);
		//debug_print_weight(weight, 384, 384, 3, "conv_w4");
	load_file(&bias, NAME_CONV_B04, LENGTH_CONV_B04);
		//debug_print_bias(bias, 384, "conv_b4");
    conv4 = conv_layer(conv3, 384, 3, 1, 1, weight, bias);
		//debug_print_input(conv4, "layer_06");
    
    /* conv5 */
    load_file(&weight, NAME_CONV_W05, LENGTH_CONV_W05);
	load_file(&bias, NAME_CONV_B05, LENGTH_CONV_B05);
    conv5 = conv_layer(conv4, 256, 3, 1, 1, weight, bias);
		//debug_print_input(conv5, "layer_07"); // 256 * 13 * 13
    
    /* pool5 */
	pool5 = max_pooling_layer(conv5, 3, 2);
		//debug_print_input(pool5, "layer_08"); //9216
    
    /* fc6 */
    load_file(&weight, NAME_FC_W01, LENGTH_FC_W01);
	load_file(&bias, NAME_FC_B01, LENGTH_FC_B01);
    fc6 = fc_layer(pool5, 4096, weight, bias, 0);
		//debug_print_input(fc6, "layer_10"); // 4096
    
    /* fc7 */
    load_file(&weight, NAME_FC_W02, LENGTH_FC_W02);
	load_file(&bias, NAME_FC_B02, LENGTH_FC_B02);
    fc7 = fc_layer(fc6, 4096, weight, bias, 0);
		//debug_print_input(fc7, "layer_12");  //4096
		
    /* fc8 */
    load_file(&weight, NAME_FC_W03, LENGTH_FC_W03);
	load_file(&bias, NAME_FC_B03, LENGTH_FC_B03);
    fc8 = fc_layer(fc7, 1000, weight, bias, 1);
		//debug_print_input(fc8, "layer_13"); //1000
	printf("softmax : %d\n", fc8.channel);
    /* softmax */
    softmax(fc8.data, 1000);
	
	return 0;
}
