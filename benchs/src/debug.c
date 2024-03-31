#include <stdio.h>
#include "input.h"

//*********
void debug_print_input(layer input, const char *log_name) {
    
    int channel, row, col;
    FILE *fp_w;
    fp_w = fopen(log_name, "w");
    fprintf(fp_w, "layer file : channel = %d width = %d\n", input.channel, input.width);
    for(channel=0; channel<input.channel; channel++) {
        fprintf(fp_w, "channel #%4d\n", channel+1);
        for(row=0; row<input.width; row++) {
            for(col=0; col<input.width; col++) {
                fprintf(fp_w, "%f  ", input.data[channel*input.width*input.width + row*input.width + col]);
            }
            fprintf(fp_w, "\n");
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