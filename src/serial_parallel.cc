#include <iostream>
#include <cstring>
#include <fstream>
#include <ctime>
using namespace std;

void Hines(double *u, double *l, double *d, double *rhs, int *p, int cellSize, int *s, int low, int high) {
    if (high == 0) {
        rhs[0] /= d[0];
        return;
    }
    int nlow = cellSize - 1, nhigh = 0;
    double factor;
    int *ns = new int[cellSize + 5];
    memcpy(ns, s, sizeof(int) * cellSize);
    for (int i = high; i >= low; i--) {
        if (s[i]) continue;
        factor = u[i] / d[i];
        d[p[i]] -= factor * l[i];
        rhs[p[i]] -= factor * rhs[i];
        ns[p[i]]--;
        ns[i]--;
        nhigh = max(nhigh, p[i]);
        nlow = min(nlow, p[i]);
    }
    Hines(u, l, d, rhs, p, cellSize, ns, nlow, nhigh);
    for (int i = low; i <= high; i++) {
        if (s[i]) continue;
        rhs[i] -= l[i] * rhs[p[i]];
        rhs[i] /= d[i];
    }
}
int cellSize;
int main(int argc, char *argv[]) {
    char *path = argv[1];
    fstream file(path, ios::in);
    file >> cellSize;
    double *u = new double[cellSize + 5];
    double *l = new double[cellSize + 5];
    double *d = new double[cellSize + 5];
    double *rhs = new double[cellSize + 5];
    int *p = new int[cellSize + 5];
    int *s = new int[cellSize + 5];
    memset(s, 0, sizeof(int) * (cellSize + 5));
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
    Hines(u, l, d, rhs, p, cellSize, s, low, high);
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