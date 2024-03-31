#include <stdio.h>
#include <stdlib.h>
#include "input.h"

layer fc_layer(layer input, int filter, float *weight, float *bias, int no_relu) {
	
	int i, o_channel;
	int i_length = input.width * input.width * input.channel;
	int o_channel_index;
	
	layer output;
	output.width = 1;
	output.channel = filter;
	output.data = (float *)malloc(output.channel * sizeof(float));
	printf("fc : %d\n", i_length);
	
	if(no_relu) {
		for(o_channel=0; o_channel<output.channel; o_channel++) {
			o_channel_index = o_channel*i_length;
			for(i=0; i<i_length; i++) {
				bias[o_channel] += input.data[i] * weight[o_channel_index+i];
				//bias[o_channel] += input.data[i] * weight[o_channel*i_length+i];
			}
			output.data[o_channel] = bias[o_channel];
		}
	}
	else {
		for(o_channel=0; o_channel<output.channel; o_channel++) {
			for(i=0; i<i_length; i++) {
				o_channel_index = o_channel*i_length;
				bias[o_channel] += input.data[i] * weight[o_channel_index+i];
				//bias[o_channel] += input.data[i] * weight[o_channel*i_length+i];
			}
			if(bias[o_channel] < 0)
				output.data[o_channel] = 0.0f;
			else
				output.data[o_channel] = bias[o_channel];
		}
	}
	
	free(input.data);
	return output;
}
