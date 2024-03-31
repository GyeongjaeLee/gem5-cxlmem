#include <stdlib.h>
#include "input.h"
#include <stdio.h>
layer zero_pad_layer(layer input, int pad) {
	
	int channel, row, col;
	int i_index, o_index;
    int output_element;
	int i_channel_index, i_row_index;
	int o_channel_index, o_row_index;
	layer output;
    
    if(pad == 0)
        return input;

	output.width = input.width + pad*2;
	output.channel = input.channel;
    output_element = output.width*output.width*output.channel;
	output.data = (float *)malloc(output_element*sizeof(float));
	
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
					output.data[o_index] = 0.0f;
				else {
					i_index = i_channel_index + i_row_index + (col-pad);
					//i_index = channel*input.width*input.width + (row-pad)*input.width + (col-pad);
					output.data[o_index] = input.data[i_index]; 
				}
			}
		}
	}
	free(input.data);
	return output;
}

layer conv_layer(layer input, int filter, int f_size, int stride, int pad, float *weight, float *bias) {
	
    int i, j;
	int o_channel, row, col, i_channel;
	
	int i_channel_index, i_row_offset, i_col_offset;
	int w_o_channel_index, w_i_channel_index;
	int o_index, o_channel_index, o_row_index;
	
    int output_element;
	float sum;
    layer output;
	printf("conv : %d * %d * %d\n", input.width, input.width, input.channel);

	layer z_pad = zero_pad_layer(input, pad);
		//printf("\tpad : %d * %d * %d\n", z_pad.width, z_pad.width, z_pad.channel);

	output.width = (z_pad.width + stride - f_size) / stride;
	output.channel = filter;
    output_element = output.width*output.width*output.channel;
	output.data = (float *)malloc(output_element*sizeof(float));
	
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
							sum += z_pad.data[i_channel_index + (i_row_offset+i)*z_pad.width + (i_col_offset+j)] * weight[w_o_channel_index + w_i_channel_index + i*f_size + j];
                            //sum += z_pad.data[i_channel*z_pad.width*z_pad.width + (row*stride+i)*z_pad.width + col*stride+j] * weight[o_channel*z_pad.channel*f_size*f_size + i_channel*f_size*f_size + i*f_size + j];
                        }
					}
				}
                sum += bias[o_channel];
                if(sum < 0) {
					output.data[o_index] = 0.0f;
                    //output.data[o_channel*output.width*output.width + row*output.width + col] = 0.0f;
				}
                else {
					output.data[o_index] = sum;
                    //output.data[o_channel*output.width*output.width + row*output.width + col] = sum;
				}
			}
		}
	}	

	free(z_pad.data);
	return output;
}