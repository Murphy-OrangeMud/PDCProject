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
int cellSize;

struct linkedson {
    linkedson *nextson;
    int sonid;
    linkedson() {}
    linkedson(int _sonid, linkedson *_nextson): sonid(_sonid), nextson(_nextson) {}
};

void Forward(double *u, double *l, double *d, double *rhs, int *p, linkedson **s, int *sonnum, int cellSize, int start) {
    int cur = start;
    while (sonnum[cur] == 1) {
        rhs[cur] -= l[cur] * rhs[p[cur]];
        rhs[cur] /= d[cur];
        cur = s[cur]->nextson->sonid;
    }
    rhs[cur] -= l[cur] * rhs[p[cur]];
    rhs[cur] /= d[cur];

    // 碰到叶节点
    if (sonnum[cur] == 0) return;

    //递归子树
    linkedson *tmp = s[cur]->nextson;
    # pragma omp parallel
    {
        # pragma omp single
        while (tmp) {
            # pragma omp task
            Forward(u, l, d, rhs, p, s, sonnum, cellSize, tmp->sonid);
            tmp = tmp->nextson;
        }
    }
}

void Backward(double *u, double *l, double *d, double *rhs, int *p, int cellSize, int nodeid, linkedson **s, int *sonnum, int tid, int *endpoint, int *record) {
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

void Hines(double *u, double *l, double *d, double *rhs, int *p, int cellSize, int *start, linkedson **s, int *sonnum) {
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
            linkedson *tmp = s[start[i + 1]]->nextson;
            while (tmp) {
                if (!record[tmp->sonid]) {
                    temp = false;
                    break;
                }
                tmp = tmp->nextson;
            }
            if (temp && !record[start[i + 1]]) {
                Backward(u, l, d, rhs, p, cellSize, start[i + 1], s, sonnum, i, endpoint, record);
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
    linkedson *tmp = s[0]->nextson;
    # pragma omp parallel
    {
        # pragma omp single
        while (tmp) {
            # pragma omp task
            Forward(u, l, d, rhs, p, s, sonnum, cellSize, tmp->sonid);
            tmp = tmp->nextson;
        }
    }
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
    linkedson **s = new linkedson*[cellSize];
    
    memset(u, 0, dsize);
    memset(l, 0, dsize);
    memset(d, 0, dsize);
    memset(rhs, 0, dsize);
    memset(p, 0, isize);
    memset(sonnum, 0, isize);
    memset(s, 0, sizeof(linkedson *)*cellSize);

    for (int i = 0; i < cellSize; i++) {
        s[i] = new linkedson(-1, NULL);
    }

    int idx;
    for (int i = 0; i < cellSize; i++) {
        int idx;
        file >> idx >> u[i] >> l[i] >> rhs[i] >> d[i] >> p[i];
        if (p[i] == -1) continue;
        if (s[p[i]]->nextson == NULL) {
            s[p[i]]->nextson = new linkedson(i, NULL);
        }
        else {
            linkedson *tmp = new linkedson(i, s[p[i]]->nextson);
            s[p[i]]->nextson = tmp;
        }
        sonnum[p[i]]++;
    }

    int *start = new int[isize];
    memset(start, 0, sizeof(start));
    // start是没有完成backward计算的节点构成的树中的所有叶节点。从1开始标号。其中start[0]中装的是叶节点数目。
    start[0] = 0;
    for (int i = 0; i < cellSize; i++) {
        if (s[i]->nextson == NULL) {
            start[++start[0]] = i;
        }
    }

    unsigned tBegin = clock();
    Hines(u, l, d, rhs, p, cellSize, start, s, sonnum);
    unsigned tEnd = clock();
    cout << "Time cost: " << (double) (tEnd - tBegin) * 1000 / CLOCKS_PER_SEC << "ms\n";

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
}