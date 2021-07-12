#include <sys/time.h>
#include <cstdlib>
#include <cmath>
#include "gc.h"

GC::GC(SmcContext *context, int ldlen)
{
    this->context = context;
    this->ldlen = ldlen;
    party = context->party;
    
    auto gcContext = setup_semi_honest(context->netServer->getNetIO(), party);
    context->netServer->getNetIO()->flush();
}

GC::~GC()
{

}

void GC::setLdlen(int ldlen)
{
    this->ldlen = ldlen;
}

Tensor GC::truncateByOT(Tensor tensor, bool *msb0, bool *msb1)
{
    int length = tensor.numel();

    if (party == ALICE)
    {
        Tensor r = torch::randint(0, 1 << 30, tensor.sizes(), tensor.device()).to(kInt32);
        Tensor m0 = r, m1 = r - (1 << (32 - ldlen));
        Tensor m0Cpu = m0.cpu(), m1Cpu = m1.cpu();

        unsigned int *m0Data = (unsigned int *)m0Cpu.data<int>();
        unsigned int *m1Data = (unsigned int *)m1Cpu.data<int>();
        block *m0Block = new block[length * 2], *m1Block = new block[length * 2];
        for (int i = 0; i < length; i++)
        {
            m0Block[i] = makeBlock(0, msb0[i] * m1Data[i] + (1 - msb0[i]) * m0Data[i]);
            m1Block[i] = makeBlock(0, (1 - msb0[i]) * m1Data[i] + msb0[i] * m0Data[i]);
        }
        
        int mask = 1 << (32 - ldlen);
        tensor = tensor.__rshift__(ldlen).__and__(mask - 1) - r;

        r = torch::randint(0, 1 << 30, tensor.sizes(), tensor.device()).to(kInt32);
        m0 = r;
        m1 = m0 - (1 << (32 - ldlen));
        m0Cpu = m0.cpu();
        m1Cpu = m1.cpu();
        m0Data = (unsigned int *)m0Cpu.data<int>();
        m1Data = (unsigned int *)m1Cpu.data<int>();
        for (int i = 0; i < length; i++)
        {
            m0Block[i + length] = makeBlock(0, msb1[i] * m1Data[i] + (1 - msb1[i]) * m0Data[i]);
            m1Block[i + length] = makeBlock(0, (1 - msb1[i]) * m1Data[i] + msb1[i] * m0Data[i]);
        }
        
        IKNP<NetIO> *iknp = new IKNP<NetIO>(context->netServer->getNetIO());
        iknp->send(m0Block, m1Block, length * 2);
        context->netServer->getNetIO()->flush();

        delete[] m0Block;
        delete[] m1Block;
        delete iknp;

        return tensor - r;
    }
    else
    {
        bool *msb = new bool[length * 2];
        memcpy(msb, msb0, length);
        memcpy(msb + length, msb1, length);

        block *recvBlock = new block[length * 2];
        IKNP<NetIO> *iknp = new IKNP<NetIO>(context->netServer->getNetIO());
        iknp->recv(recvBlock, msb, length * 2);
        context->netServer->getNetIO()->flush();

        unsigned int *recvInt = new unsigned int[length * 2];
        for (int i = 0; i < length * 2; i++)
        {
            uint64_t *v64val = (uint64_t *)&(recvBlock[i]);
            recvInt[i] = v64val[0];
        }

        Tensor recvTensor = torch::from_blob(recvInt, {2, tensor.numel()}, kInt32).to(tensor.device());
        delete[] recvBlock;
        delete[] recvInt;
        delete[] msb;
        delete iknp;
        
        int mask = 1 << (32 - ldlen);
        tensor = tensor.__rshift__(ldlen).__and__(mask - 1);
        tensor += recvTensor[0].reshape(tensor.sizes());
        tensor += recvTensor[1].reshape(tensor.sizes());
        return tensor;
    }
}

Tensor GC::truncate(Tensor tensor)
{
    int length = tensor.numel();
    Tensor tensorCpu = tensor.cpu();
    unsigned int *data = (unsigned int *)tensorCpu.data<int>();

    Integer *data0 = new Integer[length];
    Integer *data1 = new Integer[length];

    for (int i = 0; i < length; i++)
	{
		data0[i] = Integer(33, data[i], ALICE);
		data1[i] = Integer(33, data[i], BOB);
	}
    
    for (int i = 0; i < length; i++)
        data0[i] = data0[i] + data1[i];

    bool *msb0 = new bool[length];
    bool *msb1 = new bool[length];
    for (int i = 0; i < length; i++)
    {
        msb0[i] = getLSB(data0[i].bits[32].bit);
        msb1[i] = getLSB(data0[i].bits[31].bit);
    }
    
    delete[] data0;
    delete[] data1;

    //truncate by ot
    Tensor res = truncateByOT(tensor, msb0, msb1);
    delete[] msb0;
    delete[] msb1;
    return res;
}

Tensor GC::relu(Tensor tensor)
{
    int length = tensor.numel();
    Tensor tensorCpu = tensor.cpu();
    unsigned int *data = (unsigned int *)tensorCpu.data<int>();

    Integer *data0 = new Integer[length];
    Integer *data1 = new Integer[length];

    for (int i = 0; i < length; i++)
	{
		data0[i] = Integer(32, data[i], ALICE);
		data1[i] = Integer(32, data[i], BOB);
	}
    
    for (int i = 0; i < length; i++)
        data0[i] = data0[i] + data1[i];

    bool *msbRelu = new bool[length];
    for (int i = 0; i < length; i++)
        msbRelu[i] = getLSB(data0[i].bits[31].bit);
    
    delete[] data0;
    delete[] data1;
    
    //relu by ot
    Tensor relued = reluByOT(tensor, msbRelu);
    delete[] msbRelu;
    return relued;
}

Tensor GC::reluWithTruncate(Tensor tensor)
{
    int length = tensor.numel();
    Tensor tensorCpu = tensor.cpu();
    unsigned int *data = (unsigned int *)tensorCpu.data<int>();

    Integer *data0 = new Integer[length];
    Integer *data1 = new Integer[length];

    for (int i = 0; i < length; i++)
	{
		data0[i] = Integer(33, data[i], ALICE);
		data1[i] = Integer(33, data[i], BOB);
	}
    
    for (int i = 0; i < length; i++)
        data0[i] = data0[i] + data1[i];

    bool *msbTruncate = new bool[length];
    bool *msbRelu = new bool[length];
    for (int i = 0; i < length; i++)
    {
        msbTruncate[i] = getLSB(data0[i].bits[32].bit);
        msbRelu[i] = getLSB(data0[i].bits[31].bit);
    }
    
    delete[] data0;
    delete[] data1;

    //truncate by ot
    Tensor truncated = truncateByOT(tensor, msbTruncate, msbRelu);
    delete[] msbTruncate;
    
    //relu by ot
    Tensor relued = reluByOT(truncated, msbRelu);
    delete[] msbRelu;
    return relued;
}

Tensor GC::reluByOT(Tensor tensor, bool *msb)
{
    Tensor m0 = torch::randint(0, 1l << 30, tensor.sizes(), tensor.device()).to(kInt32);
    Tensor m1 = m0 + tensor;
    Tensor m1Cpu = m1.cpu();
    Tensor m0Cpu = m0.cpu();
    unsigned int *m0Data = (unsigned int *)m0Cpu.data<int>();
    unsigned int *m1Data = (unsigned int *)m1Cpu.data<int>();

    int length = tensor.numel();
    block *m0Block = new block[length];
    block *m1Block = new block[length];
    for (int i = 0; i < length; i++)
    {
        m1Block[i] = makeBlock(0, msb[i] * m1Data[i] + (1 - msb[i]) * m0Data[i]);
        m0Block[i] = makeBlock(0, (1 - msb[i]) * m1Data[i] + msb[i] * m0Data[i]);
    }

    block *recvBlock = new block[length];
    
    if(party == ALICE)
    {
        IKNP<NetIO> *iknp = new IKNP<NetIO>(context->netServer->getNetIO());
        iknp->send(m0Block, m1Block, length);
        delete iknp;
        iknp = new IKNP<NetIO>(context->netServer->getNetIO());
        iknp->recv(recvBlock, msb, length);
        delete iknp;
    }
    else
    {
        IKNP<NetIO> *iknp = new IKNP<NetIO>(context->netServer->getNetIO());
        iknp->recv(recvBlock, msb, length);
        delete iknp;
        iknp = new IKNP<NetIO>(context->netServer->getNetIO());
        iknp->send(m0Block, m1Block, length);
        delete iknp;
    }
    
    //get int from block
    unsigned int *recvInt = new unsigned int[length];
    for (int i = 0; i < length; i++)
    {
        uint64_t *v64val = (uint64_t *)&(recvBlock[i]);
        recvInt[i] = v64val[0];
    }
    Tensor recvTensor = torch::from_blob(recvInt, tensor.sizes(), kInt32).to(tensor.device());

    delete[] m0Block;
    delete[] m1Block;
    delete[] recvBlock;
    delete[] recvInt;
    return recvTensor - m0;
}

Tensor GC::gcSoftmax(Tensor numerator, Tensor denominator)
{
    int kind = numerator.sizes()[0];
    numerator = numerator.cpu();
    denominator = denominator.cpu();

    int *data0 = numerator.data<int>();
    int *data1 = denominator.data<int>();
    Integer *numerator0 = new Integer[kind];
    Integer *numerator1 = new Integer[kind];
    Integer *denominator0 = new Integer[kind];
    Integer *denominator1 = new Integer[kind];

    for (int i = 0; i < kind; i++)
    {
        numerator0[i] = Integer(32, data0[i], ALICE);
        numerator1[i] = Integer(32, data0[i], BOB);
        denominator0[i] = Integer(32, data1[i], ALICE);
        denominator1[i] = Integer(32, data1[i], BOB);
    }

    for (int i = 0; i < kind; i++)
    {
        numerator0[i] = numerator0[i] + numerator1[i];
        denominator0[i] = (denominator0[i] + denominator1[i]);
        Bit sig = denominator0[i].bits[31];
        denominator0[i] = denominator0[i] >> ldlen;
        for (int j = 0; j < ldlen; j++)
            denominator0[i].bits[31 - j] = sig;

        numerator0[i] = numerator0[i] / denominator0[i];
    }

    int *res = yshareReveal(numerator0, kind, 32, BOB);
    Tensor resTensor = torch::from_blob(res, numerator.sizes(), kInt32).to(*context->device);

    delete[] numerator0;
    delete[] numerator1;
    delete[] denominator0;
    delete[] denominator1;
    context->netServer->getNetIO()->flush();

    return resTensor;
}

int *GC::yshareReveal(Integer *input, long len, long bitlen, long party)
{
	block *toreveal = new block[len * bitlen];
	block *ptr = toreveal;
	for (int i = 0; i < len; i++)
	{
		memcpy(ptr, (block *)input[i].bits.data(), sizeof(block) * bitlen);
		ptr += bitlen;
	}

	bool *resreveal = new bool[len * bitlen];
	ProtocolExecution::prot_exec->reveal(resreveal, party, toreveal, len * bitlen);

	//recover from bool to long
	int *resLong = new int[len];
	memset(resLong, 0x00, len * sizeof(int));

	for (int i = 0; i < len; i++)
	{
		long tmp = 0;
		tmp = tmp | resreveal[i * bitlen + bitlen - 1];
		for (int j = bitlen - 2; j >= 0; j--)
		{
			tmp = tmp << 1;
			tmp = tmp | resreveal[i * bitlen + j];
		}
		resLong[i] = tmp;
	}

	delete[] toreveal;
	delete[] resreveal;

	return resLong;
}

int *GC::argmax(Tensor tensor)
{
    int kind = tensor.sizes()[0];
    tensor = tensor.cpu();

    int *data = tensor.data<int>();
	int indexBitLen = (int)(log(kind) / log(2) + 1);
    Integer *A0 = new Integer[kind];
    Integer *A1 = new Integer[kind];
    Integer *maxIndex = new Integer;
	Integer *index = new Integer[kind];

    for (int i = 0; i < kind; i++)
    {
		A0[i] = Integer(32, data[i], ALICE);
		A1[i] = Integer(32, data[i], BOB);
		index[i] = Integer(indexBitLen, i, PUBLIC);
	}

    for (int i = 0; i < kind; i++)
        A0[i] = A0[i] + A1[i];

    //find max
    *maxIndex = Integer(indexBitLen, 0, PUBLIC);
    Integer max = A0[0];
    for (int j = 1; j < kind; j++)
    {
        Integer tmp = A0[j];
        Bit geq = true;
        Bit to_swap = ((max < tmp) == geq);
        swap(to_swap, max, tmp);
        swap(to_swap, *maxIndex, index[j]);
    }

    int *reveal = yshareReveal(maxIndex, 1, indexBitLen, PUBLIC);
	context->netServer->getNetIO()->flush();

	delete[] A0;
	delete[] A1;
    delete maxIndex;
    delete[] index;

	return reveal;
}

bool *GC::gcCompare(int *input, long length)
{
	Integer *A0 = new Integer[length];
	Integer *A1 = new Integer[length];
	bool *signal = new bool[length];

	for (long i = 0; i < length; i++)
	{
		A0[i] = Integer(32, input[i], ALICE);
		A1[i] = Integer(32, input[i], BOB);
	}

	for (long i = 0; i < length; i++)
		A0[i] = A0[i] + A1[i];
	
	for (int i = 0; i < length; i++)
		signal[i] = getLSB(A0[i].bits[31].bit);
    
	bool *recved_signal = new bool[length];
	if(party == ALICE)
	{
		context->netServer->getNetIO()->send_data(signal, length);
		context->netServer->getNetIO()->recv_data(recved_signal ,length);
	}
	else
	{
		context->netServer->getNetIO()->recv_data(recved_signal, length);
		context->netServer->getNetIO()->send_data(signal, length);
	}
    context->netServer->getNetIO()->flush();

	for (long i = 0; i < length; i++)
		signal[i] = !(signal[i] ^ recved_signal[i]);
	
	delete[] A0;
	delete[] A1;
	delete[] recved_signal;

	return signal;
}
