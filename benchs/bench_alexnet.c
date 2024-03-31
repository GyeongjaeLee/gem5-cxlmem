#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

//Input and weight definition
#define NAME_INPUT_BICYCLE  "alexnet_weights/bicycle227.bin"
#define NAME_INPUT_CAT      "alexnet_weights/cat227.bin"
#define NAME_INPUT_CHEETAH  "alexnet_weights/cheetah227.bin"
#define NAME_INPUT_FOX      "alexnet_weights/fox227.bin"
#define IMG_WIDTH           227
#define IMG_CHANNEL         3
#define BATCH_SIZE          4
#define BATCH_SIZE          4
#define LABEL_FILE_NAME     "label_name.txt"

//Alexnet weights
#define NAME_CONV_W01       "alexnet_weights/b_conv_w1"
#define NAME_CONV_W02       "alexnet_weights/b_conv_w2"
#define NAME_CONV_W03       "alexnet_weights/b_conv_w3"
#define NAME_CONV_W04       "alexnet_weights/b_conv_w4"
#define NAME_CONV_W05       "alexnet_weights/b_conv_w5"

#define NAME_FC_W01         "alexnet_weights/b_fc_w1"
#define NAME_FC_W02         "alexnet_weights/b_fc_w2"
#define NAME_FC_W03         "alexnet_weights/b_fc_w3"

#define NAME_CONV_B01       "alexnet_weights/b_conv_b1"
#define NAME_CONV_B02       "alexnet_weights/b_conv_b2"
#define NAME_CONV_B03       "alexnet_weights/b_conv_b3"
#define NAME_CONV_B04       "alexnet_weights/b_conv_b4"
#define NAME_CONV_B05       "alexnet_weights/b_conv_b5"

#define NAME_FC_B01         "alexnet_weights/b_fc_b1"
#define NAME_FC_B02         "alexnet_weights/b_fc_b2"
#define NAME_FC_B03         "alexnet_weights/b_fc_b3"

//Length definition
#define LENGTH_INPUT        154587
#define LENGTH_CONV_W01     34848
#define LENGTH_CONV_W02     614400
#define LENGTH_CONV_W03     884736
#define LENGTH_CONV_W04     1327104
#define LENGTH_CONV_W05     884736

#define LENGTH_FC_W01       150994944
#define LENGTH_FC_W02       67108864
#define LENGTH_FC_W03       16384000

#define LENGTH_CONV_B01     96
#define LENGTH_CONV_B02     256
#define LENGTH_CONV_B03     384
#define LENGTH_CONV_B04     384
#define LENGTH_CONV_B05     256

#define LENGTH_FC_B01       4096
#define LENGTH_FC_B02       4096
#define LENGTH_FC_B03       1000

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
layer conv_layer(layer input, int filter, int f_size, int stride, int pad, float *weight, float *bias);
/* pool_layer.h */
layer avg_pooling_layer(layer input, int kernel, int stride, int pad);
layer max_pooling_layer(layer input, int kernel, int stride, int pad);
/* fc_layer.h */
layer fc_layer(layer input, int output_channel, float *weight, float *bias, int no_relu);

/* debug.h */
void debug_print_input(layer input, const char *log_name);
void debug_print_weight(float *weight, int filters, int i_channel, int f_size, const char *log_name);
void debug_print_bias(float *bias, int channel, const char *log_name);

int main() {
	layer input;
    layer conv1, conv2, conv3, conv4, conv5;
    layer pool1, pool2, pool5;
    layer fc6, fc7, fc8;
    float *weight = NULL, *bias = NULL;
	
	/* network size and input names */
	const char *input_names[BATCH_SIZE] = {NAME_INPUT_BICYCLE, NAME_INPUT_CAT, NAME_INPUT_CHEETAH, NAME_INPUT_FOX};
	input.width = IMG_WIDTH;
	input.channel = IMG_CHANNEL;
	input.batch = BATCH_SIZE;
	input.data = malloc(input.batch * sizeof(float *));
	
    /* conv1 */
    load_input(&input, input_names);
		//debug_print_input(input, "layer_00");
    load_file(&weight, NAME_CONV_W01, LENGTH_CONV_W01);
		//debug_print_weight(weight, 96, 3, 11, "sim_w1");
	load_file(&bias, NAME_CONV_B01, LENGTH_CONV_B01);
		//debug_print_bias(bias, 96, "sim_b1");
    conv1 = conv_layer(input, 96, 11, 4, 0, weight, bias);
		//debug_print_input(conv1, "layer_01");
    
    /* pool1 */
	pool1 = max_pooling_layer(conv1, 3, 2, 0);
		//debug_print_input(pool1, "layer_02");

    /* conv2 */
    load_file(&weight, NAME_CONV_W02, LENGTH_CONV_W02);
	load_file(&bias, NAME_CONV_B02, LENGTH_CONV_B02);
    conv2 = conv_layer(pool1, 256, 5, 1, 2, weight, bias);
		//debug_print_input(conv2, "layer_03");
    
    /* pool2 */
	pool2 = max_pooling_layer(conv2, 3, 2, 0);
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
	pool5 = max_pooling_layer(conv5, 3, 2, 0);
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
    softmax(fc8, 1000, input_names, LABEL_FILE_NAME);
	
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
		printf("-------------Prediction by AlexNet for %s-------------\n", file_name[batch]);
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

layer conv_layer(layer input, int filter, int f_size, int stride, int pad, float *weight, float *bias) {
	
    int i, j;
	int batch, o_channel, row, col, i_channel;
	
	int i_channel_index, i_row_offset, i_col_offset;
	int w_o_channel_index, w_i_channel_index;
	int o_index, o_channel_index, o_row_index ,output_size_jh;

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
		//free(z_pad.data[batch]);
	}
	printf("conv layer done!!  \n");
	//free(z_pad.data);
	return output;
}

layer avg_pooling_layer(layer input, int kernel, int stride, int pad) {
    
    int batch, channel, row, col, i, j;
	int i_index;
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
							i_index = i_channel_index + (i_row_stride+i)*z_pad.width + (i_col_stride+j);
							//i_index = channel*input.width*input.width + (row*stride+i)*input.width + (col*stride+j);
                            sum += z_pad.data[batch][i_index];
						}
					}
                    sum = sum / filter_elements;
					output.data[batch][o_channel_index+o_row_index+col] = sum;
					//output.data[channel*output.width*output.width + row*output.width + col] = sum;
				}
			}
		}
		// what about input parameters
		//free(input.data[batch]);
	}
	//free(input.data);
	return output;
}

layer max_pooling_layer(layer input, int kernel, int stride, int pad) {
	
	int batch, channel, row, col, i, j;
	int i_index;
	int i_channel_index, i_row_stride, i_col_stride;
	int o_channel_index, o_row_index;
	float max;
	layer output;
    
    layer z_pad = zero_pad_layer(input, pad);
	
	output.width = (z_pad.width + stride - kernel) / stride;
	output.channel = z_pad.channel;
	output.batch = z_pad.batch;
	output.data = malloc(z_pad.batch * sizeof(float *));

	printf("max pool : %d * %d * %d\n", input.width, input.width, input.channel);
	
	for(batch=0; batch<output.batch; batch++) {
		output.data[batch] = (float *)malloc(output.width*output.width*output.channel*sizeof(float));
		for(channel=0; channel<output.channel; channel++) {
			i_channel_index = channel*z_pad.width*z_pad.width;
			o_channel_index = channel*output.width*output.width;
			for(row=0; row<output.width; row++) {
				i_row_stride = row*stride;
				o_row_index = row*output.width;
				for(col=0; col<output.width; col++) {
					max = 0.0;
					i_col_stride = col*stride;
					for(i=0; i<kernel; i++) {
						for(j=0; j<kernel; j++) {
							i_index = i_channel_index + (i_row_stride+i)*z_pad.width + (i_col_stride+j);
							//i_index = channel*input.width*input.width + (row*stride+i)*input.width + (col*stride+j);
							if(max < z_pad.data[batch][i_index])
								max = z_pad.data[batch][i_index];
						}
					}
					output.data[batch][o_channel_index+o_row_index+col] = max;
					//output.data[channel*output.width*output.width + row*output.width + col] = max;
				}
			}
		}
		// what about input parameters
		//free(input.data[batch]);
	}
	//free(input.data);
	return output;
}

layer fc_layer(layer input, int filter, float *weight, float *bias, int no_relu) {
	
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
			//free(input.data[batch]);
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
			//free(input.data[batch]);
		}
	}
	//free(input.data);
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
