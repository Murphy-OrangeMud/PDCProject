// 半年之后，你要和半年前的学长，至少不差太多吧。
// 虽然认识的人的多少肯定是比不上他了，虽然social的本事肯定和他也是不在一个段位上的。
// 但是你至少要，不那么垃圾吧。
// 我会的。

#include <iostream>
#include <fstream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <cstring>
using namespace std;

#define dsize  ((cellSize) * sizeof(double))
#define isize  ((cellSize) * sizeof(int))
#define MAXSON 5
int cellSize;

void Forward(double *u, double *l, double *d, double *rhs, int *p, int *sonset, int *sonnum, int cellSize, int start) {
    int cur = start;
    while (sonnum[cur] == 1) {
        rhs[cur] -= l[cur] * rhs[p[cur]];
        rhs[cur] /= d[cur];
        cur = sonset[cur * MAXSON];
    }
    rhs[cur] -= l[cur] * rhs[p[cur]];
    rhs[cur] /= d[cur];

    //递归子树
    # pragma omp parallel
    for (int i = 0; i < sonnum[cur]; i++) {
        Forward(u, l, d, rhs, p, sonset, sonnum, cellSize, sonset[cur * MAXSON + i]);
    }
    //# pragma omp barrier
}

void Backward(double *u, double *l, double *d, double *rhs, int *p, int cellSize, int nodeid, int *sonset, int *sonnum, int tid, int *endpoint, int *record) {
    double factor;
    int cur = nodeid;
    //树不分叉的时候，不用分出其他线程。到了树的分叉点停止
    while (p[cur] >= 0 && sonnum[p[cur]] <= 1) {
        record[cur] = 1;
        factor = u[cur] / d[cur];
        d[p[cur]] -= factor * l[cur];
        rhs[p[cur]] -= factor * rhs[cur];
        cur = p[cur];
    }

    //对分叉点进行操作
    if (cur > 0) {
        record[cur] = 1;
        factor = u[cur] / d[cur];
        #pragma omp critical
        {
            d[p[cur]] -= factor * l[cur];
            rhs[p[cur]] -= factor * rhs[cur];
        }
    }

    //返回分叉点
    cur = p[cur];
    endpoint[tid] = cur;
}

void Hines(double *u, double *l, double *d, double *rhs, int *p, int cellSize, int *start, int *sonset, int *sonnum) {
    // backward
    double factor;
    bool flag = true;
    int *record = new int[cellSize];
    memset(record, 0, sizeof(record));
    while (flag) {
        // 多线程倒序直到分叉点返回
        int threadnum = start[0];
        int *endpoint = new int[threadnum];
        memset(endpoint, 0, sizeof(int) * threadnum);
        # pragma omp parallel shared(endpoint, record)
        for (int i = 0; i < threadnum; i++) {
            bool temp = true;
            for (int j = 0; j < sonnum[start[i + 1]]; j++) {
                if (!record[sonset[start[i + 1] * MAXSON + j]]) {
                    temp = false;
                    break;
                }
            }
            if (temp && !record[start[i + 1]]) {
                Backward(u, l, d, rhs, p, cellSize, start[i + 1], sonset, sonnum, i, endpoint, record);
            }
        }
        # pragma omp barrier

        // 下一次迭代倒序的入口
        // 一定有多个线程返回同一个分叉点，用visited记录
        int *visited = new int[cellSize];
        memset(visited, 0, sizeof(int) * cellSize);
        memset(start, 0, sizeof(int) * cellSize);

        for (int i = 0; i < threadnum; i++) {
            if (endpoint[i] == -1) flag = false;
            else if (!visited[endpoint[i]]) {
                start[++start[0]] = endpoint[i];
                visited[endpoint[i]] = 1;
            }
        }

        delete [] visited;
        delete [] endpoint;
    }

    rhs[0] /= d[0];

    // forward
    # pragma omp parallel
    for (int i = 0; i < sonnum[0]; i++) {
        Forward(u, l, d, rhs, p, sonset, sonnum, cellSize, sonset[i]);
    }
    //# pragma omp barrier
}

int main(int argc, char *argv[]) {
    char *path = argv[1];
    fstream file(path, ios::in);
    file >> cellSize;
    
    double *u = new double[cellSize];
    double *l = new double[cellSize];
    double *d = new double[cellSize];
    double *rhs = new double[cellSize];
    int *p = new int[cellSize];
    int *sonnum = new int[cellSize];
    int *sonset = new int[cellSize * MAXSON];
    
    memset(u, 0, dsize);
    memset(l, 0, dsize);
    memset(d, 0, dsize);
    memset(rhs, 0, dsize);
    memset(p, 0, isize);
    memset(sonnum, 0, isize);
    memset(sonset, 0, sizeof(int)*MAXSON);

    int idx;
    for (int i = 0; i < cellSize; i++) {
        int idx;
        file >> idx >> u[i] >> l[i] >> rhs[i] >> d[i] >> p[i];
        if (p[i] == -1) continue;
        sonset[p[i] * MAXSON + sonnum[p[i]]] = i;
        sonnum[p[i]]++;
    }

    int *start = new int[isize];
    memset(start, 0, sizeof(start));
    // start是没有完成backward计算的节点构成的树中的所有叶节点。从1开始标号。其中start[0]中装的是叶节点数目。
    start[0] = 0;
    for (int i = 0; i < cellSize; i++) {
        if (sonnum[i] == 0) {
            start[++start[0]] = i;
        }
    }

    unsigned tBegin = clock();
    Hines(u, l, d, rhs, p, cellSize, start, sonset, sonnum);
    unsigned tEnd = clock();
    cout << "Time cost: " << (double) (tEnd - tBegin) * 1000 / CLOCKS_PER_SEC << "ms\n";

    fstream outfile(argv[2], ios::out);
    for (int i = 0; i < cellSize; i++) {
        outfile << i << " " << u[i] << " " << l[i] << " " << rhs[i] << " " << d[i] << endl;
    }

    delete [] u;
    delete [] l;
    delete [] d;
    delete [] rhs;
    delete [] p;
    delete [] sonset;
    delete [] sonnum;
}