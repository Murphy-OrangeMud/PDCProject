#include <iostream>
#include <fstream>
#include <ctime>
#include <cstring>
using namespace std;
void Hines(double *u, double *l, double *d, double *rhs, int *p, int cellSize) {
    int i;
    double factor;
    for (i = cellSize - 1; i >= 0; i--) {
        factor = u[i] / d[i];
        d[p[i]] -= factor * l[i];
        rhs[p[i]] -= factor * rhs[i];
    }
    rhs[0] /= d[0];
    for (i = 1; i <= cellSize - 1; i++) {
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
    int idx;
    for (int i = 0; i < cellSize; i++) {
        file >> idx >> u[i] >> l[i] >> rhs[i] >> d[i] >> p[i];
    }
    unsigned start = clock();
    Hines(u, l, d, rhs, p, cellSize);
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