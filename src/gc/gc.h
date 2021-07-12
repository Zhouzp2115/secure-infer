#ifndef __GC_H__
#define __GC_H__

#include <emp-ot/emp-ot.h>
#include <emp-tool/emp-tool.h>
#include <emp-sh2pc/emp-sh2pc.h>
#include <iostream>
#include <torch/torch.h>
#include "../base/smcbase.h"

using namespace emp;
using namespace std;
using namespace torch;

class SmcContext;

class GC
{
public:
    GC(SmcContext *context, int ldlen);

    ~GC();

    void setLdlen(int ldlen);

    Tensor truncate(Tensor tensor);

    Tensor reluWithTruncate(Tensor tensor);

    Tensor relu(Tensor tensor);

    Tensor gcSoftmax(Tensor numerator, Tensor denominator);

    int *argmax(Tensor tensor);

    bool *gcCompare(int *input, long length);
private:
    Tensor truncateByOT(Tensor, bool *msb0, bool *msb1);

    Tensor reluByOT(Tensor tensor, bool *msb);

    int *yshareReveal(Integer *input, long len, long bitlen, long party);
    
    SmcContext *context;
    int party, ldlen;
};

#endif