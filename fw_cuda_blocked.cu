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

/* Kernel za device koji implementira prvu fazu blocked Floyd-Warshall algoritma. */
__global__ void FW_Cuda_Phase1(int B, int* D, int* PI, int dim, int primaryBlockDim) {	
	extern __shared__ int shrd[];
	
	int* sh_D = &shrd[0];
	int* sh_PI = &shrd[primaryBlockDim*primaryBlockDim];
	
	int i = B*primaryBlockDim + threadIdx.y;
	int j = B*primaryBlockDim + threadIdx.x;
	
	int sub_dim = primaryBlockDim;
	int sub_i = threadIdx.y;
	int sub_j = threadIdx.x;
	
	__syncthreads();
	
	if (sub_i < sub_dim && sub_j < sub_dim && sub_i*sub_dim+sub_j < sub_dim*sub_dim && i < max(sub_dim, dim) && j < max(sub_dim, dim)) {
		sh_D[sub_i*sub_dim+sub_j] = D[i*dim+j];
		sh_PI[sub_i*sub_dim+sub_j] = PI[i*dim+j];
	}
	else {
		sh_D[sub_i*sub_dim+sub_j] = INT_MAX;
		sh_PI[sub_i*sub_dim+sub_j] = -1;
	}
	
	__syncthreads();
	
	for (int sub_k = 0; sub_k < min(sub_dim, dim); sub_k++) {
		__syncthreads();
		if (i < max(sub_dim, dim) && j < max(sub_dim, dim) && sh_D[sub_i*sub_dim+sub_k] < INT_MAX && sh_D[sub_k*sub_dim+sub_j] < INT_MAX) {
			if (sh_D[sub_i*sub_dim+sub_j] > sh_D[sub_i*sub_dim+sub_k] + sh_D[sub_k*sub_dim+sub_j]) {
				sh_D[sub_i*sub_dim+sub_j] = sh_D[sub_i*sub_dim+sub_k] + sh_D[sub_k*sub_dim+sub_j];
				sh_PI[sub_i*sub_dim+sub_j] = sh_PI[sub_k*sub_dim+sub_j];
			}
		}
	}
	
	__syncthreads();
	
	if (sub_i < min(sub_dim, dim) && sub_j < min(sub_dim, dim) && sub_i*sub_dim+sub_j < sub_dim*sub_dim && i < max(sub_dim, dim) && j < max(sub_dim, dim)) {
		D[i*dim+j] = sh_D[sub_i*sub_dim+sub_j];
		PI[i*dim+j] = sh_PI[sub_i*sub_dim+sub_j];
	}
}


/* Kernel za device koji implementira drugu fazu blocked Floyd-Warshall algoritma. */
__global__ void FW_Cuda_Phase2(int B, int* D, int* PI, int dim, int primaryBlockDim) {
	extern __shared__ int shrd[];
	
	/* Sve varijable koje imaju prefiks "p_" pripadaju primarnom podbloku, sve koje imaju "c_" pripadaju trenutnom bloku. */
	int* p_sh_D = &shrd[0];
	int* p_sh_PI = &shrd[primaryBlockDim*primaryBlockDim];
	int* c_sh_D = &shrd[2*primaryBlockDim*primaryBlockDim];
	int* c_sh_PI = &shrd[3*primaryBlockDim*primaryBlockDim];
	
	int p_i = B*primaryBlockDim + threadIdx.y;
	int p_j = B*primaryBlockDim + threadIdx.x;
	
	/* Ako je trenutni blok prije primarnog, skipCenterBlock biti će 0.
	   Inače, ako je primarni ili neki nakon njega, biti će 1. */
	int skipCenterBlock = min((blockIdx.x+1)/(B+1), 1);
	
	int c_i, c_j;
	
	/* Ako je y koordinata bloka u gridu jednaka 0, onda on pripada istom retku kao i primarni blok.
	   Ako je y koordinata bloka u gridu jednaka 1, pripada istom stupcu kao i primarni blok. */
	if (blockIdx.y == 0) {
		c_i = p_i;
		c_j = (blockIdx.x+skipCenterBlock)*primaryBlockDim + threadIdx.x;
	}
	else {
		c_i = (blockIdx.x+skipCenterBlock)*primaryBlockDim + threadIdx.y;
		c_j = p_j;
	}
	
	int sub_dim = primaryBlockDim;
	int sub_i = threadIdx.y;
	int sub_j = threadIdx.x;
	
	__syncthreads();
	
	p_sh_D[sub_i*sub_dim+sub_j] = D[p_i*dim+p_j];
	p_sh_PI[sub_i*sub_dim+sub_j] = PI[p_i*dim+p_j];

	if (sub_i < sub_dim && sub_j < sub_dim && sub_i*sub_dim+sub_j < sub_dim*sub_dim && c_i < max(sub_dim, dim) && c_j < max(sub_dim, dim)) {
		c_sh_D[sub_i*sub_dim+sub_j] = D[c_i*dim+c_j];
		c_sh_PI[sub_i*sub_dim+sub_j] = PI[c_i*dim+c_j];
	}
	else {
		c_sh_D[sub_i*sub_dim+sub_j] = INT_MAX;
		c_sh_PI[sub_i*sub_dim+sub_j] = -1;
	}
	
	__syncthreads();
		
	for (int sub_k = 0; sub_k < min(sub_dim, dim); sub_k++) {
		__syncthreads();
		/* Pripada istom stupcu kao i primarni blok. */
		if (blockIdx.y == 1) {
			if (c_i < max(sub_dim, dim) && c_j < max(sub_dim, dim) && c_sh_D[sub_i*sub_dim+sub_k] < INT_MAX && p_sh_D[sub_k*sub_dim+sub_j] < INT_MAX) {
				if (c_sh_D[sub_i*sub_dim+sub_j] > c_sh_D[sub_i*sub_dim+sub_k] + p_sh_D[sub_k*sub_dim+sub_j]) {
					c_sh_D[sub_i*sub_dim+sub_j] = c_sh_D[sub_i*sub_dim+sub_k] + p_sh_D[sub_k*sub_dim+sub_j];
					c_sh_PI[sub_i*sub_dim+sub_j] = p_sh_PI[sub_k*sub_dim+sub_j];
				}
			}
		}
		/* Pripada istom retku kao i primarni blok. */
		if (blockIdx.y == 0) {
			if (c_i < max(sub_dim, dim) && c_j < max(sub_dim, dim) && p_sh_D[sub_i*sub_dim+sub_k] < INT_MAX && c_sh_D[sub_k*sub_dim+sub_j] < INT_MAX) {
				if (c_sh_D[sub_i*sub_dim+sub_j] > p_sh_D[sub_i*sub_dim+sub_k] + c_sh_D[sub_k*sub_dim+sub_j]) {
					c_sh_D[sub_i*sub_dim+sub_j] = p_sh_D[sub_i*sub_dim+sub_k] + c_sh_D[sub_k*sub_dim+sub_j];
					c_sh_PI[sub_i*sub_dim+sub_j] = c_sh_PI[sub_k*sub_dim+sub_j];
				}
			}
		}
		__syncthreads();
	}
	
	__syncthreads();
	
	if (sub_i < min(sub_dim, dim) && sub_j < min(sub_dim, dim) && sub_i*sub_dim+sub_j < sub_dim*sub_dim && c_i < max(sub_dim, dim) && c_j < max(sub_dim, dim)) {
		D[c_i*dim+c_j] = c_sh_D[sub_i*sub_dim+sub_j];
		PI[c_i*dim+c_j] = c_sh_PI[sub_i*sub_dim+sub_j];
	}
}

/* Kernel za device koji implementira treću fazu blocked Floyd-Warshall algoritma. */
__global__ void FW_Cuda_Phase3(int B, int* D, int* PI, int dim, int primaryBlockDim) {
	extern __shared__ int shrd[];
	
	/* Sve varijable koje imaju prefiks "p1_" pripadaju primarnom podbloku 1 izračunatom u fazi 2, 
	   sve koje imaju prefiks "p2_" pripadaju primarnom podbloku 2 izračunatom u fazi 2,
	   a sve koje imaju "c_" pripadaju trenutnom bloku. */
	int* p1_sh_D = &shrd[0];
	int* p1_sh_PI = &shrd[primaryBlockDim*primaryBlockDim];
	int* p2_sh_D = &shrd[2*primaryBlockDim*primaryBlockDim];
	int* p2_sh_PI = &shrd[3*primaryBlockDim*primaryBlockDim];
	int* c_sh_D = &shrd[4*primaryBlockDim*primaryBlockDim];
	int* c_sh_PI = &shrd[5*primaryBlockDim*primaryBlockDim];
	
	/* Ako je trenutni blok prije primarnog, skipCenterBlock biti će 0.
	   Inače, ako je primarni ili neki nakon njega, biti će 1. U ovoj fazi to radimo po obje osi. */
	int skipCenterBlockX = min((blockIdx.x+1)/(B+1), 1);
	int skipCenterBlockY = min((blockIdx.y+1)/(B+1), 1);
	
	int c_i = (blockIdx.y+skipCenterBlockY)*primaryBlockDim + threadIdx.y;
	int c_j = (blockIdx.x+skipCenterBlockX)*primaryBlockDim + threadIdx.x;
	
	int p1_i = c_i;
	int p1_j = B*primaryBlockDim + threadIdx.x;
	int p2_i = B*primaryBlockDim + threadIdx.y;
	int p2_j = c_j;
	
	int sub_dim = primaryBlockDim;
	int sub_i = threadIdx.y;
	int sub_j = threadIdx.x;
	
	__syncthreads();
	
	p1_sh_D[sub_i*sub_dim+sub_j] = D[p1_i*dim+p1_j];
	p1_sh_PI[sub_i*sub_dim+sub_j] = PI[p1_i*dim+p1_j];
	p2_sh_D[sub_i*sub_dim+sub_j] = D[p2_i*dim+p2_j];
	p2_sh_PI[sub_i*sub_dim+sub_j] = PI[p2_i*dim+p2_j];
	
	if (sub_i < sub_dim && sub_j < sub_dim && sub_i*sub_dim+sub_j < sub_dim*sub_dim && c_i < dim && c_j < dim) {
		c_sh_D[sub_i*sub_dim+sub_j] = D[c_i*dim+c_j];
		c_sh_PI[sub_i*sub_dim+sub_j] = PI[c_i*dim+c_j];
	}
	else {
		c_sh_D[sub_i*sub_dim+sub_j] = INT_MAX;
		c_sh_PI[sub_i*sub_dim+sub_j] = -1;
	}
	
	__syncthreads();
	
	for (int sub_k = 0; sub_k < min(sub_dim, dim); sub_k++) {
		__syncthreads();
		if (c_i < max(sub_dim, dim) && c_j < max(sub_dim, dim) && p1_sh_D[sub_i*sub_dim+sub_k] < INT_MAX && p2_sh_D[sub_k*sub_dim+sub_j] < INT_MAX) {
			if (c_sh_D[sub_i*sub_dim+sub_j] > p1_sh_D[sub_i*sub_dim+sub_k] + p2_sh_D[sub_k*sub_dim+sub_j]) {
				c_sh_D[sub_i*sub_dim+sub_j] = p1_sh_D[sub_i*sub_dim+sub_k] + p2_sh_D[sub_k*sub_dim+sub_j];
				c_sh_PI[sub_i*sub_dim+sub_j] = p2_sh_PI[sub_k*sub_dim+sub_j];
			}
		}
		__syncthreads();
	}
	
	__syncthreads();
	
	if (sub_i < min(sub_dim, dim) && sub_j < min(sub_dim, dim) && sub_i*sub_dim+sub_j < sub_dim*sub_dim && c_i < max(sub_dim, dim) && c_j < max(sub_dim, dim)) {
		D[c_i*dim+c_j] = c_sh_D[sub_i*sub_dim+sub_j];
		PI[c_i*dim+c_j] = c_sh_PI[sub_i*sub_dim+sub_j];
	}
}



void Blocked_Floyd_Warshall_Cuda (int* W, int* D, int* PI, unsigned int dim) {	
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
	int blockDim = 25;									/* Ukoliko je dimenzija bloka 32, imamo 1024 threadova po bloku.
														   U blocked verziji Floyd-Warshall algoritma ovaj broj ovisi o tome koliko
														   threadova se može pokrenuti u jednom bloku na nekoj grafičkoj kartici,
														   ali i o tome koliko shared memorije po multiprocesoru ima dotična grafička kartica.
														   Konkretno, za Teslu C2050/C2070 na kakvoj je rađen ovaj rad, maksimalno je 
														   1024 threada po bloku, te je maksimalna shareana memorija multiprocesora 48kb,
														   što bi značilo da možemo spremiti 4096 int elementa (jer u fazi 3 nam trebaju 3 bloka
														   pa zapravo imamo 16kb po bloku za spremanje integera). Kako moramo gledati broj koji je
														   manji (između 1024 zbog threadova/bloku i 4096 zbog shareane memorije), algoritam ćemo
														   pokretati tako da je jedan blok veličine najviše 32*32, pri čemu se za veličinu bloka
														   uzima prvi broj manji ili jednak broju 32 s kojim je dijeljiv broj vrhova grafa. */

	int numberOfBlocks = ceil((float)dim/(float)blockDim);		/* Broj (primarnih) blokova u blocked Floyd-Warshall algoritmu. */
	
	cout << "Blocked Floyd-Warshall algoritam se pokreće sa " << numberOfBlocks << " primarna bloka po dijagonali.\r\n";
	cout << "CUDA kerneli se pokreću kako slijedi: \r\n \t Faza 1: grid dimenzije 1x1 \r\n";
	cout << "\t Faza 2: grid dimenzije " << numberOfBlocks-1 << "x2";
	if (numberOfBlocks-1 == 0) cout << " (Faza 2 se neće izvršiti zbog dimenzija grida)";
	cout << "\r\n";
	cout << "\t Faza 3: grid dimenzije " << numberOfBlocks-1 << "x" << numberOfBlocks-1;
	if (numberOfBlocks-1 == 0) cout << " (Faza 3 se neće izvršiti zbog dimenzija grida)";
	cout << "\r\n";
	cout << "Svi blokovi se pokreću s " << blockDim*blockDim << " threada po bloku.\r\n";
	
	/* Iteriranje po blokovima radimo na CPU, ostalo paraleliziramo. */
	for (int B = 0; B < numberOfBlocks; B++) {
		
		/* Veličina shared memorije je blockDim*blockDim za matricu D i za matricu PI. */
		FW_Cuda_Phase1<<<dim3(1, 1, 1), dim3(blockDim, blockDim, 1), 2*blockDim*blockDim*sizeof(int)>>> (B, d_D, d_PI, dim, blockDim);
		
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			fprintf(stderr, "Neuspješno pokrenuta kernel metoda FW_Cuda_Phase1 (error code %s)!\n", cudaGetErrorString(err));
			cout << "\r\n B = " << B << "\r\n";
			exit(EXIT_FAILURE);
		}
		
		cudaThreadSynchronize();
		
		/* Veličina shared memorije je blockDim*blockDim za primarnu matricu D, trenutnu matricu D, primarnu i trenutnu za matricu PI. */
		if (numberOfBlocks-1 > 0) {
			FW_Cuda_Phase2<<<dim3(numberOfBlocks-1, 2, 1), dim3(blockDim, blockDim, 1), 4*blockDim*blockDim*sizeof(int)>>> (B, d_D, d_PI, dim, blockDim);
			
			err = cudaGetLastError();
			if (err != cudaSuccess) {
				fprintf(stderr, "Neuspješno pokrenuta kernel metoda FW_Cuda_Phase2 (error code %s)!\n", cudaGetErrorString(err));
				cout << "\r\n B = " << B << "\r\n";
				exit(EXIT_FAILURE);
			}
		}
		
		cudaThreadSynchronize();
		
		/* Veličina shared memorije je blockDim*blockDim za trenutnu matricu D, dvije primarne matrice D izračunate u fazi 2,
		   te za pripadne matrice PI. */
		if (numberOfBlocks-1 > 0) {
			FW_Cuda_Phase3<<<dim3(numberOfBlocks-1, numberOfBlocks-1, 1), dim3(blockDim, blockDim, 1), 6*blockDim*blockDim*sizeof(int)>>> (B, d_D, d_PI, dim, blockDim);
			
			err = cudaGetLastError();
			if (err != cudaSuccess) {
				fprintf(stderr, "Neuspješno pokrenuta kernel metoda FW_Cuda_Phase3 (error code %s)!\n", cudaGetErrorString(err));
				cout << "\r\n B = " << B << "\r\n";
				exit(EXIT_FAILURE);
			}
		}
		
		/* Sinkronizacija threadova kako bi se završila B-ta iteracija, te kako bi se prešlo na (B+1). iteraciju. */
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
	ofstream outputFile; outputFile.open("output_cuda_blocked.txt");
	
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
	Blocked_Floyd_Warshall_Cuda(W, D, PI, V);
	
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
	
	cout << "Vrijeme izvršavanja Blocked Floyd-Warshall algoritma: " << elapsedTime << "s.\r\n";
	
	if (checkSolutionCorrectness(W, D, PI, V) == true)
		cout << "Svi najkraći putevi su točno izračunati!\r\n";
	else
		cout << "Najkraći putevi nisu točno izračunati.\r\n";
	
	inputGraphFile.close();
	outputFile.close();
	free(W); free(D); free(PI);
	
	return 0;
}