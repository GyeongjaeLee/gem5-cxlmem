#include <stdio.h>
#include <stdlib.h>

#define LENGTH 52

int main() {
    
    int i;
    float *buf;
    FILE *fp_r;
    fp_r = fopen("b_conv_b2", "r");
    buf = malloc(sizeof(float) * LENGTH);
    fread(buf, sizeof(float), LENGTH, fp_r);
    
    for(i=0; i<LENGTH; i++)
        printf("weight #%3d : %f\n", i+1, buf[i]);
    
    return 0;
}