#ifndef __DATALOADER__H__
#define __DATALOADER__H__

#include <sys/time.h>
#include <unistd.h>
#include <torch/torch.h>

using namespace std;
using namespace torch;

class MNIST
{
public:
    MNIST(Device device);

    Tensor testData, testLabel;
    int length;

private:
    void load(Device device);
};

class FASHIONMNIST
{
public:
    FASHIONMNIST(Device device);

    Tensor testData, testLabel;
    int length;

private:
    void load(Device device);
};

class CIFAR10
{
public:
    CIFAR10(Device device);

    Tensor testData, testLabel;
    int length;

private:
    void load(Device device);
};

#endif