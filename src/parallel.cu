#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <ctime>
#include <fstream>
using namespace std;

#define dsize  ((cellSize) * sizeof(double))
#define isize  ((cellSize) * sizeof(int))

__global__ void
Forward(double *u, double *l, double *d, double *rhs, int *p, int *s, int *son, int cellSize, int start) {
    int cur = start;
    while (s[cur] == 1) {
        rhs[cur] -= l[cur] * rhs[p[cur]];
        rhs[cur] /= d[cur];
        cur = son[cur][0];
    }

    if (s[cur] == 0) return;

    cudaStream_t stream[s[cur]];
    for (int i = 0; i < s[cur]; i++) {
        cudaStreamCreate(&s[cur]);
        Forward(u, l, d, rhs, p, s, son, cellSize, son[i]);
        cudaStreamDestroy(s[cur]);
    }
}

__global__ void
Backward(double *u, double *l, double *d, double *rhs, int *p, int *s, int *son, int cellSize, int start, int id) {
    double factor;
    int cur = start;
    while (p[cur] >= 0 && s[p[cur]] <= 1) {
        factor = u[cur] / d[cur];
        d[p[cur]] -= factor * l[cur];
        rhs[p[cur]] -= factor * rhs[cur];
        cur = p[cur];
    }
    if (cur > 0) {
        factor = u[cur] / d[cur];
        atomicSub(&d[p[i]], factor * l[cur]);
        atomicSub(&rhs[p[cur]], factor * rhs[cur]);
    }
    cur = p[cur];
    endpoint[id] = cur;
}

__global__ void
hines(double *u, double *l, double *d, double *rhs, int *p, int *s, int *son, int cellSize, int *start) {
    // backward
    bool flag = true;
    while (flag) {
        __shared__ int *endpoint;
        int streamSize = start[0] + 1;
        cudaMalloc(&endpoint, (streamSize * sizeof(int)));
        cudaStream_t stream[streamSize];
        for (int i = 1; i < streamSize; i++) {
            cudaStreamCreate(&stream[i-1]);
            Backward<<<1, 1, 0, stream[i-1]>>>(u, l, d, rhs, p, s, son, cellSize, start[i], i-1);
            cudaStreamDestroy(stream[i-1]);
        }
        cudaDeviceSynchronize();
        start[0] = 0;
        int *visited;
        cudaMalloc(&visited, isize);
        cudaMemset(visited, 0, isize);
        bool flag = false;
        for (int i = 0; i < streamSize; i++) {
            if (!visited[endpoint[i]]) {
                start[++start[0]] = endpoint[i];
                if (endpoint[i] == -1) {
                    flag = false;
                }
            }
        }
        cudaFree(visited);
        cudaFree(endpoint);
    }

    rhs[0] /= d[0];

    // forward
    cudaStream_t stream[s[0]];
    for (int i = 0; i < s[cur]; i++) {
        cudaStreamCreate(&s[cur]);
        Forward(u, l, d, rhs, p, s, son, cellSize, son[i]);
        cudaStreamDestroy(s[cur]);
    }
    cudaDeviceSynchronize();
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

    double *hu = new double[cellSize];
    double *hl = new double[cellSize];
    double *hd = new double[cellSize];
    double *hrhs = new double[cellSize];
    int *hp = new int[cellSize];
    int *hs = new int[cellSize];
    int *hson = new int[cellSize * 10];
    int *hstart = new int[cellSize];
    // link *hson = new link*[cellSize];
    memset(hson, -1, 10 * isize);
    memset(hs, 0, isize);
    memset(hstart, -1, sizeof(hstart));
    
    char *path = argv[1];
    fstream file(path, ios::in);
    file >> cellSize;

    int idx;
    for (int i = 0; i < cellSize; i++) {
        file >> idx >> hu[i] >> hl[i] >> hrhs[i] >> hd[i] >> hp[i];
        if (!i) continue;
        hson[hp[i]][hs[hp[i]]++] = i;
    }
    
    hstart[0] = 0;
    for (int i = 0; i < cellSize; i++) {
        if (hs[i] == 0) {
            hstart[++hstart[0]] = hs[i];
        }
    }

    int tmpisize = isize * 10;
    double *u, *l, *d, *rhs;
    int *p, *s, *son, *start;
    cudaMalloc(&u, dsize);
    cudaMalloc(&l, dsize);
    cudaMalloc(&d, dsize);
    cudaMalloc(&rhs, dsize);
    cudaMalloc(&p, isize);
    cudaMalloc(&s, isize);
    cudaMalloc(&son, tmpisize);
    cudaMalloc(&start, isize);

    cudaMemset(s, 0, isize);

    cudaMemcpy(start, hstart, isize, cudaMemcpyHostToDevice);
    cudaMemcpy(u, hu, dsize, cudaMemcpyHostToDevice);
    cudaMemcpy(l, hl, dsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d, hd, dsize, cudaMemcpyHostToDevice);
    cudaMemcpy(rhs, hrhs, dsize, cudaMemcpyHostToDevice);
    cudaMemcpy(p, hp, isize, cudaMemcpyHostToDevice);
    cudaMemcpy(son, hson, isize * 10, cudaMemcpyHostToDevice);
    cudaMemcpy(s, hs, isize, cudaMemcpyHostToDevice);

    cudaStream_t initstream;
    cudaStreamCreate(&initstream);
    unsigned tBegin = clock();
    Hines<<<1, 1, 0, initstream>>>(u, l, d, rhs, p, s, son, cellSize, start);
    unsigned tEnd = clock();
    cudaStreamDestroy(initstream);

    cout << "Time cost: " << (double) (tEnd - tBegin) * 1000.0 / CLOCKS_PER_SEC << "ms\n";

    cudaMemcpy(hu, u, dsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hl, l, dsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hd, d, dsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hrhs, rhs, dsize, cudaMemcpyDeviceToHost);
    
    fstream outfile(argv[2], ios::out);
    for (int i = 0; i < cellSize; i++) {
        outfile << i << " " << hu[i] << " " << hl[i] << " " << hrhs[i] << " " << hd[i] << endl;
    }

    cudaFree(u);
    cudaFree(l);
    cudaFree(d);
    cudaFree(rhs);
    cudaFree(s);
    cudaFree(son);
    cudaFree(p);
    cudaFree(start);

    delete [] hu;
    delete [] hl;
    delete [] hd;
    delete [] hrhs;
    delete [] hp;
    delete [] hs;
    delete [] hson;
    delete [] hstart;

    return 0;
}