//this is for the purpose of measuring performance in C
#include <stdio.h>
#include <stdlib.h> // for RAND, and rand
#include <math.h>

double gauss() {

static char has_saved;
static double saved;
	if(has_saved){ 
		has_saved = 0;
		return saved;
	}
    double u = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1) return gauss();
    double c = sqrt(-2 * log(r) / r);
	saved = v*c; has_saved = 1;
    return u * c;
}

//given scenario (k,d), and policy x, find online revenue.
double OnlineR(int k, int d, int m, int* x, 
				int* L, int* U, double* f)
{
	int T=0, Di=0, bi=0, ax, i; 
	double R=0;
	for(i=m; i>0; i--){
	   if(i<k) Di = L[i];
	   if(i==k) Di = d;
	   if(i>k) Di = U[i];
	   if(x[i]>0) bi += x[i];
	   ax = T + Di;
	   if (ax > bi) ax = bi;
	   R += (ax - T) * f[i];
	   T = ax;
	}
	return R;
}
//runs under 3s with:
//time cperf
int main(void){
int k = 2, d=20,  m = 3; 
int x[4] = {0, 20, 30, 40}; 
int L[4] = {0, 12, 15, 20}; 
int U[4] = {0, 30, 40, 70};
double f[4] ={0, 100, 60, 40};
printf("Revenue = %f\n", OnlineR(k,d,m,x,L,U,f));

int scens = 100, i,j,l;
for(i=m; i>0; i--) scens += U[i]-L[i];
for(i=0; i< 2000*50; i++){ 
		for(l=0;l<m;l++) gauss();
		for(j=0;j<scens;j++){ 
			k = rand()%m+1;
			OnlineR(k,d,m,x,L,U,f);
		}
}
}
