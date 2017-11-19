/*
* Classic DMRG implementation.
* Compile with: cc dmrg.c -L/usr/grads/lib -llapack -lblas -lm -o dmrg 
*/


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h> 
#include <unistd.h> 
// TODO
// Smarter krons (use block in matrices)
// Smarter matrix multiplication (for example when I know one matrix is kroned with identity)
// Pack matrices
// Upate operators smarter

typedef struct { float re, im; } fcomplex;

fcomplex mult(fcomplex i, fcomplex j) {
	fcomplex result = {.re = (i.re*j.re - i.im*j.im), .im = (i.re*j.im + i.im*j.re)};
	return result;
}

int cgemm_(const char *transa, const char *transb, const int *m, const int *n, 
  const int *k, const fcomplex *alpha, const fcomplex *A, const int *lda, 
  const fcomplex *B, const int *ldb, const fcomplex *beta, fcomplex *C, const int *ldc); 

void cheev_( char* jobz, char* uplo, const int* n, fcomplex* A, const int* lda,
                double* w, fcomplex* work, int* lwork, double* rwork, int* info );


void cgesvd_( char* jobu, char* jobvt, int* m, int* n, fcomplex* a,
                int* lda, float* s, fcomplex* u, int* ldu, fcomplex* vt, int* ldvt,
                fcomplex* work, int* lwork, float* rwork, int* info );

// Global consts TODO static these
const int D = 50;
const int SITES_STATE_NUM = 2;
const fcomplex S_PLUS_SITE[2*2] = {
	{0, 0}, {1, 0},
	{0, 0}, {0, 0}
};
const fcomplex S_Z_SITE[2*2] = {
	{1, 0}, {0, 0},
	{0, 0}, {-1, 0}
};
const fcomplex h = {.re = 1, .im = 0}, j = {.re = 1, .im = 0} , jz = {.re = 1, .im = 0};
const fcomplex ONE = {.re = 1, .im = 0}, ZERO = {.re = 0, .im = 0};

typedef struct{
	int elementsNum;
	int basisSize;
 	fcomplex* sPlusLastSite;
	fcomplex* sZLastSite;
	fcomplex* H;
} Block;

void freeBlock(Block* block) {
	free(block->sPlusLastSite);
	free(block->sZLastSite);
	free(block->H);
	free(block);
} 

void printMatrix(char* desc, fcomplex* M, int r, int c);

void getEntryIndices(int i, int rowsNum, int colsNum, int *row, int *col) {
	*row = i / colsNum;
	*col = i % colsNum;
}

int getArrayIndex(int row, int col, int rowsNum, int colsNum) {
	return row * colsNum + col;
}

int getArraySize(int rowsNum, int colsNum) {
	return rowsNum * colsNum;
}

// Returns the KrONEcker product of A and B.
// [a11 * B a12*B ...]
// [a21 * B ...      ]
// Allocates the function!
fcomplex* kron(const fcomplex* restrict A, const fcomplex* restrict B, int rowsNumA, int colsNumA, int rowsNumB, int colsNumB) {
	int rowsNum = rowsNumA * rowsNumB, colsNum = colsNumA * colsNumB;
	int resultSize =  getArraySize(rowsNum, colsNum);
	fcomplex* result = (fcomplex*) malloc(sizeof(fcomplex) * resultSize);
	for (int i = 0; i < resultSize; i++) result[i] = ONE;
	for (int i = 0; i < resultSize; i++) {
		int row, col;
		getEntryIndices(i, rowsNum, colsNum, &row, &col);

		int aRow = row / colsNumB, aCol = col / rowsNumB;
		result[i] = mult(result[i], A[getArrayIndex(aRow, aCol, rowsNumA, colsNumA)]);
		int bRow = row % colsNumB, bCol = col % rowsNumB;
		result[i] = mult(result[i], B[getArrayIndex(bRow, bCol, rowsNumB, colsNumB)]);	
	}
	return result;
}

void printMatrix(char* desc, fcomplex* M, int r, int c) {
	printf("%s:\n", desc);
	for (int i = 0; i < r * c; i++) {
		printf("%.2f, %.2f\t", M[i].re, M[i].im);
		if ((i + 1) % c == 0) printf("\n");
	}
}

Block* dmrgStep(Block* block);

fcomplex* updateOperatorForNewBasis(fcomplex* basis, int newBasisSize, fcomplex* O, int oldBasisSize);

// TOO multiply by h, j , jz
Block * initBlock() {
	Block* result = (Block*) malloc(sizeof(Block));
	fcomplex* sPlus = (fcomplex*) malloc(sizeof(fcomplex) * getArraySize(SITES_STATE_NUM, SITES_STATE_NUM));
	memcpy(sPlus, S_PLUS_SITE, sizeof(fcomplex) * getArraySize(SITES_STATE_NUM, SITES_STATE_NUM));
	fcomplex* sZ = (fcomplex*) malloc(sizeof(fcomplex) * getArraySize(SITES_STATE_NUM, SITES_STATE_NUM));
	memcpy(sZ, S_Z_SITE, sizeof(fcomplex) * getArraySize(SITES_STATE_NUM, SITES_STATE_NUM));
	fcomplex* H = (fcomplex*) malloc(sizeof(fcomplex) * getArraySize(SITES_STATE_NUM, SITES_STATE_NUM));
	memcpy(H, S_Z_SITE, sizeof(fcomplex) * getArraySize(SITES_STATE_NUM, SITES_STATE_NUM));
	result->elementsNum = 1;
	result->basisSize = 2;
	result->sPlusLastSite = sPlus;
	result->sZLastSite = sZ;
	result->H = H;
	return result;
}

int main(int argc, char const *argv[])
{
	

	// dmrgStep(&b);
	fcomplex A[3*4] = {
		{0, 0}, {0, 0}, {0, 0}, {0, 0},
		{1, 0}, {0, 0}, {0, 0}, {0, 0},  
		{0, 0}, {0, 0}, {0, 0}, {0, 0}
	};
	fcomplex B[4*4] = {
		{1, 0}, {0, 0}, {0, 0}, {0, 0},
		{0, 0}, {3, 0}, {0, 0}, {0, 0},
		{0, 0}, {0, 0}, {2, 0}, {0, 0},
		{0, 0}, {0, 0}, {0, 0}, {4, 0}
	};
	updateOperatorForNewBasis(A, 3, B, 4);

	return 0;
}

fcomplex* identity(int n) {
	int resultSize = getArraySize(n, n);
	fcomplex* result = (fcomplex*) malloc(sizeof(fcomplex) * resultSize);
	for (int i = 0; i < resultSize; i++) {
		int row, col;
		getEntryIndices(i, n, n, &row, &col);
		if (row == col) {
			result[i] = ONE;
		} else {
			result[i] = ZERO;
		}
	}
	return result;
}

// Here we add the donation of the new site to the block Hamiltonian.
void updateBlockHForNewSite(Block* block, fcomplex* H, fcomplex* sPlusNewSite, fcomplex* sZNewSite, int blockStatesNum) {
	// cgemm assigns C = alpha * AB + beta * C
	// todo better way for this?
	fcomplex* I = identity(SITES_STATE_NUM);
	fcomplex* expandedSPlusLastSite = kron(I, block->sPlusLastSite, SITES_STATE_NUM, SITES_STATE_NUM, block->basisSize, block->basisSize);
	fcomplex* expandedSZLastSite = kron(I, block->sZLastSite, SITES_STATE_NUM, SITES_STATE_NUM, block->basisSize, block->basisSize);
	free(I);

	cgemm_("N", "T", &blockStatesNum, &blockStatesNum, &blockStatesNum, &j, expandedSPlusLastSite, &blockStatesNum, 
		sPlusNewSite, &blockStatesNum, &ONE, H, &blockStatesNum);	
	cgemm_("T", "N", &blockStatesNum, &blockStatesNum, &blockStatesNum, &j, expandedSPlusLastSite, &blockStatesNum, 
		sPlusNewSite, &blockStatesNum, &ONE, H, &blockStatesNum);	
	cgemm_("N", "T", &blockStatesNum, &blockStatesNum, &blockStatesNum, &j, expandedSZLastSite, &blockStatesNum, 
		sZNewSite, &blockStatesNum, &ONE, H, &blockStatesNum);	
	free(expandedSZLastSite);
	free(expandedSPlusLastSite);
}

// Returns the H of the full lattice.
fcomplex* getFullH(fcomplex* H_A, fcomplex* sPlusNewSite, fcomplex* sZNewSite, int blockStatesNum, int fullStatesNum) {
	fcomplex* H_B = (fcomplex*) malloc(sizeof(fcomplex) * getArraySize(blockStatesNum, blockStatesNum));
	memcpy(H_B, H_A, sizeof(fcomplex) * getArraySize(blockStatesNum, blockStatesNum));
	fcomplex* fullH = kron(H_A, H_B, blockStatesNum, blockStatesNum, blockStatesNum, blockStatesNum);
	free(H_B);

	// calculate the donation of the connection of the two new sites to H.
	fcomplex* I = identity(blockStatesNum);
	fcomplex* sPlusA = kron(sPlusNewSite, I, blockStatesNum, blockStatesNum, blockStatesNum, blockStatesNum);
	fcomplex* sPlusB = kron(I, sPlusNewSite, blockStatesNum, blockStatesNum, blockStatesNum, blockStatesNum);
	fcomplex* sZA = kron(sZNewSite, I, blockStatesNum, blockStatesNum, blockStatesNum, blockStatesNum);
	fcomplex* sZB = kron(I, sZNewSite, blockStatesNum, blockStatesNum, blockStatesNum, blockStatesNum);
	cgemm_("N", "T", &fullStatesNum, &fullStatesNum, &fullStatesNum, &j, 
		sPlusA, &fullStatesNum, sPlusB, &fullStatesNum, &ONE, fullH, &fullStatesNum);	
	cgemm_("T", "N", &fullStatesNum, &fullStatesNum, &fullStatesNum, &j, 
		sPlusA, &fullStatesNum, sPlusB, &fullStatesNum, &ONE, fullH, &fullStatesNum);	
	cgemm_("N", "N", &fullStatesNum, &fullStatesNum, &fullStatesNum, &jz, 
		sZA, &fullStatesNum, sZB, &fullStatesNum, &ONE, fullH, &fullStatesNum);	
	free(I);
	free(sPlusA);
	free(sPlusB);
	free(sZA);
	free(sZB);
	return fullH;
}

// Returns the eigenvector corresponding to the smallest eigenvalue of H, H is statesNum*statesNum.
fcomplex* getGroundState(fcomplex* H, int statesNum) {
 	// dgeev changes the matrix H, and we want to keep our H intact for later.
	fcomplex* tempH = (fcomplex*) malloc(sizeof(fcomplex) * pow(statesNum, 2));
	memcpy(tempH, H, sizeof(fcomplex) * pow(statesNum, 2));
	double eigenvalues[statesNum];
	fcomplex eigenVectors[(int) pow(statesNum, 2)];
	int lwork = -1;
    fcomplex wkopt;
    fcomplex* work;
    double rwork[3*statesNum - 2];
    int info;

    cheev_( "Vectors", "Lower", &statesNum, tempH, &statesNum, eigenvalues, &wkopt, &lwork, rwork, &info);
    lwork = (int)wkopt.re;
    work = (fcomplex*)malloc( lwork*sizeof(fcomplex) );
    /* Solve eigenproblem */
    cheev_( "Vectors", "Lower", &statesNum, tempH, &statesNum, eigenvalues, work, &lwork, rwork, &info);
    /* Check for convergence */
    if( info > 0 ) {
       printf( "getGroundState: cheev failed.\n" );
       exit(1);
    }
     
    fcomplex* groundState = (fcomplex*) malloc(sizeof(fcomplex) * statesNum);
	memcpy(groundState, tempH, sizeof(fcomplex) * statesNum);
	free(tempH);
    free(work);
	return groundState;
}

// Returns the reduced density matrix. Treats the ground state as a matrix:
// groundState = sum(psi_ij * |A_i>|B_j>).
// psi* psi^dagger is our density matrix.
fcomplex* getReducedDensityMatrix(fcomplex* groundState, int blockStatesNum) {
	fcomplex* rhoA = (fcomplex*) malloc(sizeof(fcomplex) * getArraySize(blockStatesNum, blockStatesNum));
	cgemm_("N", "T", &blockStatesNum, &blockStatesNum, &blockStatesNum, &ONE, 
		groundState, &blockStatesNum, groundState, &blockStatesNum, 
		&ZERO, rhoA, &blockStatesNum);	
	return rhoA;
}

// Get new basis using SVD on the ground state in matrix form.
// psi is a blockStatesNum*blockStatesNum matrix.
fcomplex* getNewBasis(fcomplex* psi, int blockStatesNum, int* newBasisSize) {
	// Perform SVD decomposition.
	int info;
	fcomplex wkopt;
    fcomplex* work;
    /* Local arrays */
    /* iwork dimension should be at least 8*min(m,n) */
    int iwork[8*blockStatesNum];
    /* rwork dimension should be at least 5*(min(m,n))**2 + 7*min(m,n)) */
    float S[blockStatesNum], rwork[5*blockStatesNum*blockStatesNum+7*blockStatesNum];
    fcomplex VT[blockStatesNum*blockStatesNum];
    fcomplex* U = (fcomplex*) malloc(sizeof(fcomplex) * blockStatesNum*blockStatesNum);
	int lwork = -1;
 	cgesvd_("All", "All", &blockStatesNum, &blockStatesNum, psi, &blockStatesNum, S, U, &blockStatesNum, VT, &blockStatesNum, &wkopt, &lwork, rwork, &info);
    lwork = (int)wkopt.re;
    work = (fcomplex*) malloc(lwork*sizeof(fcomplex));
    /* Compute SVD */
    cgesvd_("All", "All", &blockStatesNum, &blockStatesNum, psi, &blockStatesNum, S, U, &blockStatesNum, VT, &blockStatesNum, work, &lwork, rwork, &info );
    /* Check for convergence */
    if( info > 0 ) {
        printf( "The algorithm computing SVD failed to converge.\n" );
        exit( 1 );
    }

    // Use only min(D, blockStatesNum) rows of U corresponding to the largest singular values.
    *newBasisSize = blockStatesNum < D ? blockStatesNum : D;
    U = (fcomplex*) realloc(U, sizeof(fcomplex) * blockStatesNum * (*newBasisSize));

    free( (void*)work );
    return U;
}

// TODO bugged
fcomplex* updateOperatorForNewBasis(fcomplex* basis, int newBasisSize, fcomplex* O, int oldBasisSize) {
	printMatrix("basis", basis, oldBasisSize, newBasisSize);
	printMatrix("basis", basis, newBasisSize, oldBasisSize);
	printMatrix("O", O, oldBasisSize, oldBasisSize);

	fcomplex* result = (fcomplex* ) malloc(sizeof(fcomplex) * getArraySize(newBasisSize, newBasisSize));
	fcomplex* temp = (fcomplex* ) malloc(sizeof(fcomplex) * getArraySize(oldBasisSize, newBasisSize));

 	cgemm_("N", "N", &newBasisSize, &oldBasisSize, &oldBasisSize, &ONE,
		basis, &oldBasisSize, O, &oldBasisSize,
		&ZERO, temp, &newBasisSize);
	printMatrix("temp", temp, oldBasisSize, newBasisSize);
	// printMatrix("temp", temp, newBasisSize, oldBasisSize);
 
 //  	cgemm_("N", "T", &oldBasisSize, &newBasisSize, &oldBasisSize, &ONE,
	// 	O, &oldBasisSize, basis, &newBasisSize, 
	// 	&ZERO, temp, &oldBasisSize);
	// printMatrix("temp", temp, oldBasisSize, newBasisSize);
	// cgemm_("N", "N", &newBasisSize, &newBasisSize, &oldBasisSize, &ONE, basis, &newBasisSize, 
	// 	temp, &oldBasisSize, &ZERO, result, &newBasisSize);		
	// printMatrix("res", result, newBasisSize, newBasisSize);
	free(temp);
	return result;
}

Block* dmrgStep(Block* block) {
	int blockStatesNum = block->basisSize * SITES_STATE_NUM;
	
	fcomplex* I_block = identity(block->basisSize);
 	fcomplex* I_site = identity(SITES_STATE_NUM); 
	fcomplex* sPlusNewSite = kron(S_PLUS_SITE, I_block, SITES_STATE_NUM, SITES_STATE_NUM, block->basisSize, block->basisSize);
	fcomplex* sZNewSite = kron(S_Z_SITE, I_block, SITES_STATE_NUM, SITES_STATE_NUM, block->basisSize, block->basisSize);
	fcomplex* expandedH = kron(I_site, block->H, SITES_STATE_NUM, SITES_STATE_NUM, block->basisSize, block->basisSize);
	free(I_block);
	free(I_site);

	updateBlockHForNewSite(block, expandedH, sPlusNewSite, sZNewSite, blockStatesNum);
	int fullStatesNum = pow(blockStatesNum, 2);
	fcomplex* fullH = getFullH(expandedH, sPlusNewSite, sZNewSite, blockStatesNum, fullStatesNum);
	fcomplex* groundState = getGroundState(fullH, fullStatesNum);
	int newBasisSize;
	fcomplex* basis = getNewBasis(groundState, blockStatesNum, &newBasisSize);
  
	Block* result = (Block*) malloc(sizeof(Block));
	result->elementsNum = block->elementsNum + 1;
	result->basisSize = newBasisSize;
	result->sPlusLastSite = updateOperatorForNewBasis(basis, newBasisSize, sPlusNewSite, blockStatesNum);
	result->sZLastSite = updateOperatorForNewBasis(basis, newBasisSize, sZNewSite, blockStatesNum);
	result->H = updateOperatorForNewBasis(basis, newBasisSize, expandedH, blockStatesNum);
	free(expandedH);
	free(sZNewSite);
	free(sPlusNewSite);
	freeBlock(block);
	return result;
 
}
