#include <stdio.h>
#include <stdlib.h>
#include "input.h"

layer max_pooling_layer(layer input, int f_size, int stride) {
	
	int channel, row, col, i, j;
	int i_index;
	int i_channel_index, i_row_stride, i_col_stride;
	int o_channel_index, o_row_index;
    int output_size;
	float max;
	layer output;
	
	output.width = (input.width + stride - f_size) / stride;
	output.channel = input.channel;
    output_size = output.width*output.width*output.channel*sizeof(float);
	output.data = (float *)malloc(output_size);
	printf("pool : %d * %d * %d\n", input.width, input.width, input.channel);

	
	for(channel=0; channel<output.channel; channel++) {
		i_channel_index = channel*input.width*input.width;
		o_channel_index = channel*output.width*output.width;
		for(row=0; row<output.width; row++) {
			i_row_stride = row*stride;
			o_row_index = row*output.width;
			for(col=0; col<output.width; col++) {
				max = 0.0;
				i_col_stride = col*stride;
				for(i=0; i<f_size; i++) {
					for(j=0; j<f_size; j++) {
						i_index = i_channel_index + (i_row_stride+i)*input.width + (i_col_stride+j);
						//i_index = channel*input.width*input.width + (row*stride+i)*input.width + (col*stride+j);
						if(max < input.data[i_index])
							max = input.data[i_index];
					}
				}
				output.data[o_channel_index+o_row_index+col] = max;
				//output.data[channel*output.width*output.width + row*output.width + col] = max;
			}
		}
	}
	// what about input parameters
	free(input.data);
	return output;
}