#include <stdio.h>
#include <stdlib.h>

#define SIZE_CONV_WEIGHT_01 37632
#define SIZE_CONV_BIAS_01 256


void print_conv_weight(const char *read_name, const char *write_name, int file_size, int filter_size) {
    
    int i,j,c,channel;
    float *buf;
    FILE *fp_r, *fp_w;
    
    channel = file_size / (filter_size * filter_size * sizeof(float));
    // A routine
    fp_r = fopen(read_name, "r");
    buf = (float *)malloc(file_size);
    fread(buf, sizeof(float), file_size/sizeof(float), fp_r);
    fp_w = fopen(write_name, "w");
    
    fprintf(fp_w, "Filter file size : %d\n", file_size);
    fprintf(fp_w, "Number of elements : %d\n", file_size/sizeof(float));
    fprintf(fp_w, "Channel : %d\n", channel);
    fprintf(fp_w, "Filter length : %d\n", filter_size);
    
    for(c=0; c<channel; c++) {
        fprintf(fp_w, "channel #%3d : \n", c);
        for(i=0; i<filter_size; i++) {
            for(j=0; j<filter_size; j++) {
                fprintf(fp_w, "%f ", buf[c*filter_size*filter_size+j*filter_size+i]);
            }
            fprintf(fp_w, "\n");
        }
    }
}

print_fc_weight(const char *read_name, const char *write_name, int file_size, int channel) {
    
    int i,j,c, element, line;
    float *buf;
    FILE *fp_r, *fp_w;
    
    element = file_size/sizeof(float);
    line = element / channel;
    fp_r = fopen(read_name, "r");
    buf = (float *)malloc(file_size);
    fread(buf, sizeof(float), file_size/sizeof(float), fp_r);
    fp_w = fopen(write_name, "w");
    
    fprintf(fp_w, "Filter file size : %d\n", file_size);
    fprintf(fp_w, "Number of elements : %d\n", element);
    fprintf(fp_w, "Channel : %d\n", channel);
    
    for(c=0; c<channel; c++) {
        fprintf(fp_w, "channel #%3d : \n", c);
        for(i=0; i<line; i++) {
            fprintf(fp_w, "%f ", buf[c*line+i]);
        }
        fprintf(fp_w, "\n");
    }
} 

void print_bias(const char *read_name, const char *write_name, int file_size) {
    
    int i,j, bias_length;
    float *buf;
    FILE *fp_r, *fp_w;
    
    bias_length = file_size/sizeof(float);
    
    fp_r = fopen(read_name, "r");
    buf = (float *)malloc(file_size);
    fread(buf, sizeof(float), file_size/sizeof(float), fp_r);
    fp_w = fopen(write_name, "w");
    
    fprintf(fp_w, "Bias file size : %d\n", file_size);
    fprintf(fp_w, "Number of bias : %d\n", bias_length);
    
    for(i=0; i<bias_length; i++) {
        fprintf(fp_w, "channel #%3d : %f\n", i, buf[i]);
    } 
}

int main() {
    
    const char *conv_w1 = "conv1_w";   
    const char *conv_b1 = "conv1_b";
  
    print_conv_weight(conv_w1, "n_conv_w1", SIZE_CONV_WEIGHT_01, 7);
    print_bias(conv_b1, "n_conv_b1", SIZE_CONV_BIAS_01);

    return 0;
}