#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cstring>
#include <limits>		// radi definiranja beskonačnosti
#include <ctime>		// radi mjerenja vremena izvršavanja
#include <cmath>		// radi "strop" funkcije
using namespace std;

/* Definiramo beskonačnost kao najveći mogući integer broj. */
#define infty 	std::numeric_limits<int>::max()

void printMatrix (int* G, unsigned int dim) {
	cout << "\r\n";
	for (int i = 0; i < dim*dim; i++) {
		if (G[i] < infty) {
			cout << G[i] << "\t";
		}
		else {
			cout << "∞" << "\t";
		}
		/* Ako je ispisao sve za jedan vrh, prijeđi na sljedeći u novi redak. */
		if ((i+1)%dim == 0) {
			cout << "\r\n";
		}
	}
}

/* Kernel za device koji paralelizira unutarnje dvije for petlje Floyd-Warshall algoritma. */
__global__ void FW_Cuda(int k, int* D, int* PI, unsigned int dim) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (i < dim && j < dim && D[i*dim+k] < INT_MAX && D[k*dim+j] < INT_MAX) {
		if (D[i*dim+j] > D[i*dim+k] + D[k*dim+j]) {
			D[i*dim+j] = D[i*dim+k] + D[k*dim+j];
			PI[i*dim+j] = PI[k*dim+j];
		}
	}
}


void Floyd_Warshall_Cuda (int* W, int* D, int* PI, unsigned int dim) {	
	unsigned int n = dim*dim;
	
    /* Error varijabla za handleanje CUDA errora. */
    cudaError_t err = cudaSuccess;
	
	/* Alociranje device varijabli matrica D i PI. */
    int* d_D = NULL;
    err = cudaMalloc((void**) &d_D, n*sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Neuspješno alociranje matrice D (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
    int* d_PI = NULL;
    err = cudaMalloc((void**) &d_PI, n*sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Neuspješno alociranje matrice PI (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    
	
	/* Kopiranje podataka iz host matrica u device. */
    err = cudaMemcpy(d_D, D, n*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Neuspješno kopiranje matrice D iz hosta u device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
    err = cudaMemcpy(d_PI, PI, n*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Neuspješno kopiranje matrice PI iz hosta u device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	/* Pozivanje CUDA kernela. */
	int blockDim = 32;									/* Ukoliko je dimenzija bloka 32, imamo 1024 threadova po bloku. */
	int gridDim = ceil((float)dim/(float)blockDim);		/* Računa se kolika mora biti dimenzija grida ovisno o dimenziji blokova. */
	
	cout << "CUDA kernel se pokreće sa " << gridDim*gridDim << " blokova i " << blockDim*blockDim << " threadova po bloku.\r\n";
	
	/* Vanjsku petlju Floyd-Warshall algoritma vrtimo na CPU, unutarnje dvije paraleliziramo. */
	for (int k = 0; k < dim; k++) {
		FW_Cuda<<<dim3(gridDim, gridDim, 1), dim3(blockDim, blockDim, 1)>>> (k, d_D, d_PI, dim);
		
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			fprintf(stderr, "Neuspješno pokrenuta kernel metoda (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		
		/* Sinkronizacija threadova kako bi se završila k-ta iteracija, te kako bi se prešlo na (k+1). iteraciju. */
		cudaThreadSynchronize();
	}
	
	/* Kopiranje podataka iz device matrica u host. */	
    err = cudaMemcpy(D, d_D, n*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Neuspješno kopiranje matrice D iz devicea u host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
    err = cudaMemcpy(PI, d_PI, n*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Neuspješno kopiranje matrice PI iz devicea u host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	/* Dealociranje device varijabli matrica D i PI. */
	err = cudaFree(d_D);
    if (err != cudaSuccess) {
        fprintf(stderr, "Neuspješno dealociranje matrice D (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
    err = cudaFree(d_PI);
    if (err != cudaSuccess) {
        fprintf(stderr, "Neuspješno dealociranje matrice PI (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	/* Reset CUDA devicea i završavanje CUDA Floyd-Warshalla. */
    err = cudaDeviceReset();
    if (err != cudaSuccess) {
        fprintf(stderr, "Neuspješno resetiranje devicea (završavanje sa CUDA FW, priprema za sljedeće pokretanje)! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
}


/* Metoda koja rekonstruira težinu najkraćeg puta za dani par vrhova koristeći matricu
   prethodnika PI i matricu inicijalnih težina W. */
int getPath (int* W, int* PI, int i, int j, unsigned int dim) {
	if (i == j) {
		return 0;
	}
    else if (PI[i*dim+j] == -1) {
		return infty;
	}
	else {
		int recursivePath = getPath(W, PI, i, PI[i*dim+j], dim);
		if (recursivePath < infty) {
			return recursivePath + W[PI[i*dim+j]*dim+j];
		}
		else {
			return infty;
		}
	}
}


/* Za svaki par vrhova pokreće getPath metodu koja rekonstruira težinu najkraćeg puta
   između njih koristeći matricu prethodnika PI. Tu težinu onda uspoređuje sa dobivenom težinom
   za isti par vrhova u matrici najkraćih putova D. */
bool checkSolutionCorrectness (int* W, int* D, int* PI, unsigned int dim) {
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			if (getPath(W, PI, i, j, dim) != D[i*dim+j]) {
				return false;
			}
		}
	}
	
	return true;
}


int main() {
	
	/* 
	   V - broj vrhova
	   E - broj bridova
	   u - prvi vrh pri učitavanju grafa iz datoteke
	   v - drugi vrh pri učitavanju grafa iz datoteke
	   w - težina između v1 i v2 pri učitavanju grafa iz datoteke
	*/
	unsigned int V, E;
	int u, v, w;

	ifstream inputGraphFile; inputGraphFile.open("graphFile.txt");
	ofstream outputFile; outputFile.open("output_cuda.txt");
	
	inputGraphFile >> V >> E;
	cout << "V = " << V << ", E = " << E << "\r\n";
	
	unsigned int n = V*V;
	
	/* Inicijalizacija grafova u memoriji. */
	int* W = (int*)malloc(n*sizeof(int));
	int* D = (int*)malloc(n*sizeof(int));
	int* PI = (int*)malloc(n*sizeof(int));
	
	
	/* Postavljanje inicijalnih vrijednosti za matricu prethodnika PI(0),
	   matricu težina W i matricu najkraćih putova D(0). */
	fill_n(W, n, infty); 
	fill_n(PI, n, -1);
	
	for (int i = 0; i < E; i++) {
		inputGraphFile >> u >> v >> w;
		//cout << u << " <-- " << w << " --> " << v << "\r\n";
		
		W[u*V+v] = w;
		if (u != v) {
			PI[u*V+v] = u;
		}
	}
	
	for (int i = 0; i < V; i++) {
		W[i*V+i] = 0;
	}
	
	/* D(0) = W na početku. */
	memcpy (D, W, n*sizeof(int));
	
	// printMatrix(W, V); printMatrix(D, V); printMatrix(PI, V);
	
	/* Početak mjerenja izvršavanja Floyd-Warshall algoritma. */
	clock_t begin = clock();
	
	/* Pozivamo Floyd-Warshall CPU algoritam nad učitanim grafom. */
	Floyd_Warshall_Cuda(W, D, PI, V);
	
	/* Kraj mjerenja izvršavanja Floyd-Warshall algoritma. */
	clock_t end = clock();
	double elapsedTime = double(end - begin) / CLOCKS_PER_SEC;
	
	//printMatrix(W, V); printMatrix(D, V); printMatrix(PI, V);
	
	/* Ispis rezultata u datoteku. */
	outputFile << "|V| = " << V << ", |E| = " << E << "\r\n\r\n";
	for (int i = 0; i < n; i++) {
		if (i%V==0) outputFile << "\r\n";
		if (D[i] < infty)
			outputFile << D[i] << "\t";
		else
			outputFile << "∞" << "\t";
	}
	outputFile << "\r\n\r\n";
	for (int i = 0; i < n; i++) {
		if (i%V==0) outputFile << "\r\n";
		outputFile << PI[i] << "\t";
	}
	
	cout << "Vrijeme izvršavanja Floyd-Warshall algoritma: " << elapsedTime << "s.\r\n";
	
	if (checkSolutionCorrectness(W, D, PI, V) == true)
		cout << "Svi najkraći putevi su točno izračunati!\r\n";
	else
		cout << "Najkraći putevi nisu točno izračunati.\r\n";
	
	inputGraphFile.close();
	outputFile.close();
	free(W); free(D); free(PI);
	
	return 0;
}