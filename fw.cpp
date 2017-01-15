#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cstring>
#include <limits>		// radi definiranja beskonačnosti
#include <ctime>		// radi mjerenja vremena izvršavanja
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

void Floyd_Warshall (int* W, int* D, int* PI, unsigned int dim) {
	//cout << "-------- Floyd-Warshall --------" << "\r\n";
	for (int k = 0; k < dim; k++) {
		for (int i = 0; i < dim; i++)
			for (int j = 0; j < dim; j++) {
				/* Kako ne bismo dobili overflow, umjesto provjere:
						D[i*dim+j] > D[i*dim+k] + D[k*dim+j]
				   radimo donju provjeru. */
				if (D[i*dim+k] < infty && D[k*dim+j] < infty) {
					if (D[i*dim+j] > D[i*dim+k] + D[k*dim+j]) {
						D[i*dim+j] = D[i*dim+k] + D[k*dim+j];
						PI[i*dim+j] = PI[k*dim+j];
					}
				}
			}
		//cout << "------ k = " << k << " ------" << "\r\n";
		//printMatrix(D, dim); printMatrix(PI, dim);
	}
	//cout << "--------------------------------" << "\r\n";
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
	ofstream outputFile; outputFile.open("output.txt");
	
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
	Floyd_Warshall(W, D, PI, V);
	
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