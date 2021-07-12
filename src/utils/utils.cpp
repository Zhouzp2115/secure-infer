#include "utils.h"


float *read(char *fileName, int length)
{
    FILE *file = fopen(fileName, "r");
    
    float *data = new float[length];
    int readed = fread(data, sizeof(float), length, file);

    if(readed != length)
    {
        cout << "io error: readed != length" << endl;
        exit(1);
    }
     
    fclose(file);
    return data;
}