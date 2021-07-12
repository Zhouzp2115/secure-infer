#include "dataloader.h"

/******** MNIST *********/
MNIST::MNIST(Device device)
{
    length = 10000;
    load(device);
}

void MNIST::load(Device device)
{
    int *test = new int[10000 * 28 * 28];
    int *label = new int[10000];
    unsigned char num;

    //read test data
    FILE *file = fopen("./DATA/MNIST/t10k-images-idx3-ubyte", "r");
    fseek(file, 16, SEEK_SET);
    for (int i = 0; i < 10000 * 28 * 28; i++)
    {
        num = fgetc(file);
        test[i] = (unsigned int)num;
    }
    fclose(file);

    //read test label
    file = fopen("./DATA/MNIST/t10k-labels-idx1-ubyte", "r");
    fseek(file, 8, SEEK_SET);
    for (int i = 0; i < 10000; i++)
    {
        num = fgetc(file);
        label[i] = (unsigned int)num;
    }
    fclose(file);

    testData = torch::from_blob(test, {10000, 1, 28, 28}, kInt32).to(device);
    testLabel = torch::from_blob(label, {10000, 1}, kInt32).to(device);

    testData = testData.to(kFloat) / testData.max().to(kFloat);

    delete[] test;
    delete[] label;
}

/******** FASHIONMNIST *********/
FASHIONMNIST::FASHIONMNIST(Device device)
{
    length = 10000;
    load(device);
}

void FASHIONMNIST::load(Device device)
{
    int *test = new int[10000 * 28 * 28];
    int *label = new int[10000];
    unsigned char num;

    //read test data
    FILE *file = fopen("./DATA/FASHIONMNIST/t10k-images-idx3-ubyte", "r");
    fseek(file, 16, SEEK_SET);
    for (int i = 0; i < 10000 * 28 * 28; i++)
    {
        num = fgetc(file);
        test[i] = (unsigned int)num;
    }
    fclose(file);

    //read test label
    file = fopen("./DATA/FASHIONMNIST/t10k-labels-idx1-ubyte", "r");
    fseek(file, 8, SEEK_SET);
    for (int i = 0; i < 10000; i++)
    {
        num = fgetc(file);
        label[i] = (unsigned int)num;
    }
    fclose(file);

    testData = torch::from_blob(test, {10000, 1, 28, 28}, kInt32).to(device);
    testLabel = torch::from_blob(label, {10000, 1}, kInt32).to(device);

    testData = testData.to(kFloat) / testData.max().to(kFloat);

    delete[] test;
    delete[] label;
}

/******** CIFAR10 *********/
CIFAR10::CIFAR10(Device device)
{
    length = 10000;
    load(device);
}

void CIFAR10::load(Device device)
{
    int *test = new int[10000 * 3 * 32 * 32];
    int *label = new int[10000];
    unsigned char num;
    int count = 0;

    //read test data
    int dataIndex = 0, labelIndex = 0;
    FILE *file = fopen("./DATA/CIFAR10/test_batch.bin", "r");
    while (true)
    {
        //read label
        num = fgetc(file);
        label[labelIndex] = (unsigned int)num;
        labelIndex++;

        //read data
        while (true)
        {
            num = fgetc(file);
            test[dataIndex] = (unsigned int)num;
            dataIndex++;
            if (dataIndex % 3072 == 0)
                break;
        }

        if (labelIndex % 10000 == 0)
            break;
    }
    fclose(file);

    testData = torch::from_blob(test, {10000, 3, 32, 32}, kInt32).to(device);
    testLabel = torch::from_blob(label, {10000, 1}, kInt32).to(device);

    testData = testData.to(kFloat) / testData.max().to(kFloat);

    delete[] test;
    delete[] label;
}