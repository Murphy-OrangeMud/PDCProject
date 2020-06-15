#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdio>
#include <stack>
using namespace std;

# define N 1024

struct state {
    double *s;
    int low, high;
};

int cellSize;

__global__ void 
Forward(double *u, double *l, double *d, double *rhs, int *p, int cellSize, int *s, int low, int high) {
    int b = threadIdx.x;
    __synchronize();
    for (int i = b + low; i <= high; i += N) {
        if (s[i]) continue;
        rhs[i] -= l[i] * rhs[p[i]];
        rhs[i] /= d[i];
    }
    __synchronize();
}

__global__ state
Backward(double *u, double *l, double *d, double *rhs, int *p, int cellSize, int *s, int low, int high) {
    int b = threadIdx.x;
    int *ns;
    int nhigh = 0, nlow = cellSize - 1;
    cudaMalloc(&ns, cellSize * sizeof(int));
    cudaMemcpy(ns, s, cudaMemcpyDefault);
    __synchronize();
    for (int i = high - b; i >= low; i -= N) {
        if (s[i]) continue;
        factor = u[i] / d[i];
        d[p[i]] -= factor * l[i];
        rhs[p[i]] -= factor * rhs[i];
        ns[p[i]]--;
        ns[i]--;
        nhigh = max(nhigh, p[i]);
        nlow = min(nlow, p[i]);
    }
    __synchronize();
    state nstate;
    nstate.low = nlow;
    nstate.high = nhigh;
    nstate.ns = ns;
    return nstate;
}

int main(int argc, char *argv[]) {
    int devID = 0;
    cudaSetDevice(devID);

    cudaError_t error;
    cudaDevice Prop deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess) {
        cout << "cudaGetDevice returned error " << cudaGetErrorString(error) << " (code " << error << "), line(" << __LINE__ << endl;
        exit(0); 
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited) {
        cerr << "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n";
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess) {
        cout << "cudaGetDeviceProperties returned error " << cudaGetErrorString(error) << " (code " << error << "), line(" << __LINE__ << endl;
    } else {
        cout << "GPU Device " << devID << ": " << deviceProp.name << " with compute capability " << deviceProp.major << "." << deviceProp.minor << endl << endl;
    }

    char *path = argv[1];
    fstream file(path, ios::in);
    int dsize = (cellSize + 5) * sizeof(double);
    int isize = (cellSize + 5) * sizeof(int);
    file >> cellSize;
    double *u, *l, *d, *rhs;
    int *p, *s;
    cudaMalloc(&u, dsize);
    cudaMalloc(&l, dsize);
    cudaMalloc(&d, dsize);
    cudaMalloc(&rhs, dsize);
    cudaMalloc(&p, isize);
    cudaMalloc(&s, isize);
    cudaMemset(s, 0, isize);
    int idx;
    int low = cellSize - 1, high = 0;
    for (int i = 0; i < cellSize; i++) {
        file >> idx >> u[i] >> l[i] >> rhs[i] >> d[i] >> p[i];
        if (!i) continue;
        s[p[i]]++;
    }
    for (int i = 0; i < cellSize; i++) {
        if (s[i] == 0) {
            low = min(low, i);
            high = max(high, i);
        }
    }
    unsigned start = clock();
    stack <state> stk;
    state cur;
    state nstate;
    cur.low = low;
    cur.high = high;
    cur.ns = s;
    stk.push(cur);
    while (true) {
        cur = stk.top();
        nstate = Backward<<<1, N>>>(u, l, d, rhs, p, cellSize, cur.ns, cur.low, cur.high);
        if (nstate.high > 0) {
            stk.push(nstate);
        } else break;
    }
    while (!stk.empty()) {
        cur = stk.top();
        stk.pop();
        Forward<<<1, N>>>(u, l, d, rhs, p, cellSize, cur.ns, cur.low, cur.high);
    }
    unsigned end = clock();
    cout << "time cost: " << (float)(end-start)*1000.0/CLOCKS_PER_SEC << "ms" << endl;
    fstream outfile(argv[2], ios::out);
    for (int i = 0; i < cellSize; i++) {
        outfile << i << " " << u[i] << " " << l[i] << " " << rhs[i] << " " << d[i] << endl;
    }
    /*
    delete [] u;
    cout << "u deleted\n";
    delete [] l;
    cout << "l deleted\n";
    delete [] d;
    cout << "d deleted\n";
    delete [] rhs;
    cout << "rhs deleted\n";
    delete [] p;
    cout << "p deleted\n";
    */
    return 0;
}