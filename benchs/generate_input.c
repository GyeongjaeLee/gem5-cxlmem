#include <stdio.h>
#include <stdlib.h>

#define INPUT_WIDTH 7
#define INPUT_CHANNEL 2

#define FILTER_WIDTH 3
#define FILTER_CHANNEL 2

int main() {
    
    int channel, ic, row, col;
    int parameters = INPUT_WIDTH*INPUT_WIDTH*INPUT_CHANNEL;
    float *binary;
    FILE *fp_w;
    
    binary = (float *)malloc(parameters * sizeof(float));
    
    // input    
    for(channel=0; channel<INPUT_CHANNEL; channel++)
        for(row=0; row<INPUT_WIDTH; row++)
            for(col=0; col<INPUT_WIDTH; col++) {
				//printf("%d\n", channel * 100 + row * 7 + col);
                binary[channel*INPUT_WIDTH*INPUT_WIDTH+row*INPUT_WIDTH+col] = channel * 100 + row * 7 + col+1;
            }
    
    fp_w = fopen("b_weights/input3.bin", "w");
    fwrite(binary, sizeof(float), parameters, fp_w);
    fclose(fp_w);
    free(binary);
    
    // weight
    parameters = FILTER_CHANNEL * INPUT_CHANNEL * FILTER_WIDTH * FILTER_WIDTH;
    binary = (float *)malloc(parameters * sizeof(float));
    for(channel=0; channel<FILTER_CHANNEL ; channel++)
        for(ic=0; ic<INPUT_CHANNEL ; ic++)
            for(row=0; row<FILTER_WIDTH; row++)
                for(col=0; col<FILTER_WIDTH; col++) {
                    binary[channel*INPUT_CHANNEL*FILTER_WIDTH*FILTER_WIDTH + ic*FILTER_WIDTH*FILTER_WIDTH + row*FILTER_WIDTH + col] = channel*100+ ic*10 + 1;
                }
    
    fp_w = fopen("b_weights/weight3", "w");
    fwrite(binary, sizeof(float), parameters, fp_w);
    fclose(fp_w);
    free(binary);
    
    // bias
    parameters = FILTER_CHANNEL;
    binary = (float *)malloc(parameters * sizeof(float));
    for(channel=0; channel<INPUT_CHANNEL; channel++)
        binary[channel] = 7000000*channel + 2000000;
    
    fp_w = fopen("b_weights/bias3", "w");
    fwrite(binary, sizeof(float), parameters, fp_w);
    fclose(fp_w);
    free(binary);
    
	// fc_Weight
    parameters = 32;
    binary = (float *)malloc(parameters * sizeof(float));
    for(channel=0; channel<parameters; channel++)
        binary[channel] = (channel/8)+1;
    
    fp_w = fopen("b_weights/fc_weight3", "w");
    fwrite(binary, sizeof(float), parameters, fp_w);
    fclose(fp_w);
    free(binary);
	
	// fc_bias
    parameters = 4;
    binary = (float *)malloc(parameters * sizeof(float));
    for(channel=0; channel<parameters; channel++)
        binary[channel] = channel*(11111);
    
    fp_w = fopen("b_weights/fc_bias3", "w");
    fwrite(binary, sizeof(float), parameters, fp_w);
    fclose(fp_w);
    free(binary);
	
    return 0;
}