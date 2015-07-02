#include <stdio.h>
#include <omp.h>
void run(double u[], double v[],int idx_i[], int idx_j[], double result[], int n,double w) {
  #pragma omp parallel for
  for(long i = 0; i < n; i++) {
    result[i] = u[idx_i[i]-1]*v[idx_j[i]-1]*w;
  }
}

void phi(double** u, double** v,long* idx_i, long* idx_j, double* result, double* w, int n, int r) {
  int j = 0;
  #pragma omp parallel for
  for(long i = 0; i < n; i++) {
    double* u_i = u[idx_i[i]-1];
    double* v_i = v[idx_j[i]-1];
    float r = 0.0;
    for (j = 0; j < r; j++){
      r += w[j]*u_i[j]*v_i[j];
    }
    result[i] = (double)r;
  }
}

double f_only(double** u, double** v,long* idx_i, long* idx_j, double* w, double*y, int n, int r) {
  double s = 0.0;
  #pragma omp parallel for reduction (+ : s)
  for(long i = 0; i < n; i++) {
    double* u_i = u[idx_i[i]-1];
    double* v_i = v[idx_j[i]-1];

    double residual = 0.0;
    for (int j = 0; j < r; j++){
      residual += w[j]*u_i[j]*v_i[j];
    }

    residual = residual - y[i];
    s = s + residual*residual;
  }
  return 0.5*s;
}

double fg(double** up, double** vp, double** u, double** v,long* idx_i, long* idx_j, double* w,double*y, int n, int r) {
  int j = 0;
  double s = 0.0;
  #pragma omp parallel for reduction (+ : s)
  for(long i = 0; i < n; i++) {
    double* u_i = u[idx_i[i]-1];
    double* v_i = v[idx_j[i]-1];

    double result = 0.0;
    for (j = 0; j < r; j++){
      result += w[j]*u_i[j]*v_i[j];
    }

    result = result - y[i];
    s += result*result;

    for (j = 0; j < r; j++){
      up[idx_i[i]-1][j] += result*v_i[j]*w[j];
      vp[idx_j[i]-1][j] += result*u_i[j]*w[j];
    }

  }
  return 0.5*s;
}
