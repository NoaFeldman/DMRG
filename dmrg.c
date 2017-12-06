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
// Smarter krons (use matrixBlock in matrices)
// Smarter matrix multiplication (for example when I know one matrix is kroned with identity)
// Pack matrices
// Upate operators smarter

typedef struct { float re, im; } fcomplex;

fcomplex mult(fcomplex i, fcomplex j) {
	fcomplex result = {.re = (i.re*j.re - i.im*j.im), .im = (i.re*j.im + i.im*j.re)};
	return result;
}

fcomplex add(fcomplex i, fcomplex j) {
	fcomplex result = {.re = i.re + j.re, .im = i.im + j.im};
	return result;
}

fcomplex conjugate(fcomplex f) {
	fcomplex result = {.re = f.re, .im = -1 * f.im};
	return result;
}

void printMatrix(char* desc, fcomplex* M, int rowsNum, int colsNum);

int getArrayIndex(int row, int col, int rowsNum, int colsNum) {
	return row * colsNum + col;
}
int getArraySize(int rowsNum, int colsNum) {
	return rowsNum * colsNum;
}

fcomplex* dagger(fcomplex* M, int rowsNum, int colsNum) {
	fcomplex* result = malloc(sizeof(fcomplex) * getArraySize(colsNum, rowsNum));
	for (int i = 0; i < rowsNum; i++) {
		for (int j = 0; j < colsNum; j++) {
			result[getArrayIndex(j, i, colsNum, rowsNum)] = conjugate(M[getArrayIndex(i, j, rowsNum, colsNum)]);
		}
	}
	return result;
}

void getEntryIndices(int i, int rowsNum, int colsNum, int *row, int *col) {
	*row = i / colsNum;
	*col = i % colsNum;
}

int cgemm_(const char *transa, const char *transb, const int *m, const int *n, 
  const int *k, const fcomplex *alpha, const fcomplex *A, const int *lda, 
  const fcomplex *B, const int *ldb, const fcomplex *beta, fcomplex *C, const int *ldc); 

void cheev_( char* jobz, char* uplo, const int* n, fcomplex* A, const int* lda,
                float* w, fcomplex* work, int* lwork, float* rwork, int* info );


void cgesvd_( char* jobu, char* jobvt, int* m, int* n, fcomplex* a,
                int* lda, float* s, fcomplex* u, int* ldu, fcomplex* vt, int* ldvt,
                fcomplex* work, int* lwork, float* rwork, int* info );

fcomplex* identity(int n);

typedef struct {
	int elementsNum;
	int basisSize;
 	fcomplex* sPlusLastSite; // represents the block acting on this site which will take us to the next block.
 	int nextBlockBasisSize; // this is the number of rows in sPlusLastSite.
 	fcomplex* sZLastSite;
	fcomplex* H;
	float sZ;
} MatrixBlock;

void freeMatrixBlock(MatrixBlock* matrixBlock) {
	free(matrixBlock->sPlusLastSite);
	free(matrixBlock->sZLastSite);
	free(matrixBlock->H);
}

// Global consts TODO static these
const int D = 50;
const int SITE_STATES_NUM = 2;
const fcomplex M = {.re = 1, .im = 0}, JPM = {.re = 1, .im = 0} , JZ = {.re = 1, .im = 0};
const fcomplex ONE = {.re = 1, .im = 0}, ZERO = {.re = 0, .im = 0};
fcomplex S_PLUS_DOWN_STATE[1];
fcomplex S_Z_DOWN_STATE[1];
fcomplex H_DOWN_STATE[1];
fcomplex S_Z_UP_STATE[1];
fcomplex H_UP_STATE[1];
MatrixBlock NEW_STATE_MATRIX_BLOCK[2];

void initSingleSite() {
	S_PLUS_DOWN_STATE[0] = ONE;
	S_Z_DOWN_STATE[0] = (fcomplex) {-0.5, 0};
	H_DOWN_STATE[0] = mult(M, S_Z_DOWN_STATE[0]);
	S_Z_UP_STATE[0] = (fcomplex)  {0.5, 0};
	H_UP_STATE[0] = mult(M, S_Z_UP_STATE[0]);
	NEW_STATE_MATRIX_BLOCK[0] = (MatrixBlock) {.elementsNum = 1, .basisSize = 1, .sPlusLastSite = S_PLUS_DOWN_STATE, .nextBlockBasisSize = 1, .sZLastSite = S_Z_DOWN_STATE, .H = H_DOWN_STATE, .sZ = -0.5};
	NEW_STATE_MATRIX_BLOCK[1] = (MatrixBlock) {.elementsNum = 1, .basisSize = 1, .sPlusLastSite = NULL, .sZLastSite = S_Z_UP_STATE, .H = H_UP_STATE, .sZ = 0.5};
}

fcomplex* multiplyMatrices(fcomplex* A, int ARows, int ACols, char* daggerA, fcomplex* B, int BRows, int BCols, char* daggerB) {
	int rows, cols, midDim;
	if (*daggerA == 'n' || *daggerA == 'N') {
		rows = ARows;
		midDim = ACols;
	}
	else if (*daggerA == 'd' || *daggerA == 'D') {
		rows = ACols;
		midDim = ARows;
	}
	else {
		printf("Invalid parameter daggerA for multiplyMatrices: %s\n", daggerA);
	}
	if (*daggerB == 'n' || *daggerB == 'N')
		cols = BCols;
	else if (*daggerB == 'd' || *daggerB == 'D')
		cols = BRows;
	else {
		printf("Invalid parameter daggerA for multiplyMatrices: %s\n", daggerA);
	}
	fcomplex* result = malloc(sizeof(fcomplex) * getArraySize(rows, cols));
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result[getArrayIndex(i, j, rows, cols)] = ZERO;
			for (int k = 0; k < midDim; k++) {
				fcomplex AElem = (*daggerA == 'n' || *daggerA == 'N') ? A[getArrayIndex(i, k, ARows, ACols)]
																	  : conjugate(A[getArrayIndex(k, i, ARows, ACols)]);
				fcomplex BElem = (*daggerB == 'n' || *daggerB == 'N') ? B[getArrayIndex(k, j, BRows, BCols)]
																	  : conjugate(B[getArrayIndex(j, k, BRows, BCols)]);
				result[getArrayIndex(i, j, rows, cols)] = add(result[getArrayIndex(i, j, rows, cols)], 
															  mult(AElem, BElem));																	 
			}
		}
	}
	return result;
}

fcomplex* kron(const fcomplex* restrict A, const fcomplex* restrict B, 
	int rowsNumA, int colsNumA, int rowsNumB, int colsNumB, int* resultRowsNum, int* resultsColsNum);

// TODO this would be cleaner and a little fatser with memcpy for eack row.
fcomplex* toSingleMatrix(fcomplex* A, int dimA, fcomplex* B, int dimB) {
	int dimResult = dimA + dimB;
	int arraySize = pow(dimResult, 2);
	fcomplex* result = malloc(sizeof(fcomplex) * arraySize);
	for (int i = 0; i < dimResult * dimA; i++) {
		if (i % dimResult < dimA) {
			int row = i / dimResult;
			int col = i % dimResult;
			result[i] = A[row * dimA + col];
		}
		else result[i] = ZERO;
	}
	for (int i = dimResult * dimA; i < arraySize; i++) {
		if (i % dimResult >= dimA) {
			int row = ((i / dimResult) - dimA);
			int col = i % dimResult - dimA;
			result[i] = B[row *dimB + col];
		}
		else result[i] = ZERO;
	}
	return result;
}

// TODO rename stuff
typedef struct {
	int ABlockIndex;
	int BBlockIndex;
	int subBlockSize;
} StatesDoc;

void printStatesDoc(StatesDoc* doc) {
	printf("A block index = %d, B block index = %d\n", doc->ABlockIndex, doc->BBlockIndex);
}

typedef struct {
	StatesDoc* statesDocs;
	int statesDocsNum;
} BlockStatesDoc;
void freeBlockStatesDoc(BlockStatesDoc* doc) {
	free(doc->statesDocs);
}

void printBlockStatesDoc(char* desc, BlockStatesDoc* doc) {
	printf("%s\n", desc);
	for (int i = 0; i < doc->statesDocsNum; i++) {
		printStatesDoc(&(doc->statesDocs[i]));
	}
}

void printMatrixBlock(char* desc, MatrixBlock* M);

MatrixBlock* toSingleMatrixBlock(MatrixBlock* A, MatrixBlock* B) {
	MatrixBlock* result = malloc(sizeof(MatrixBlock));
	*result = (MatrixBlock) {
		.elementsNum = A->elementsNum,
		.basisSize = A->basisSize + B->basisSize,
		.H = toSingleMatrix(A->H, A->basisSize, B->H, B->basisSize),
		.sZLastSite = toSingleMatrix(A->sZLastSite, A->basisSize, B->sZLastSite, B->basisSize),
		.sZ = A->sZ
	};
	return result;
}

void addSubBlock(fcomplex* M, int MRowsNum, int MColsNum, fcomplex alpha, fcomplex* m, int mRowsNum, int mColsNum, int firstRow, int firstCol, char* dagger);

void addSPMFromSubBlocksToH(MatrixBlock* block, char* oldAOperator, 
		MatrixBlock* oldSubBlockA, MatrixBlock* oldSubBlockB,
		MatrixBlock* newSubBlockA, MatrixBlock* newSubBlockB,
		int oldSubBlockIndex, int newSubBlockIndex) {
	fcomplex *sPM, *sMinus;
	int sPMRows, sPMCols;
	if (*oldAOperator == 'P' || *oldAOperator == 'p') {
		sMinus = dagger(newSubBlockB->sPlusLastSite, newSubBlockB->nextBlockBasisSize, newSubBlockB->basisSize);
		sPM = kron(oldSubBlockA->sPlusLastSite, sMinus, 
				   oldSubBlockA->nextBlockBasisSize, oldSubBlockA->basisSize,
				   newSubBlockB->basisSize, newSubBlockB->nextBlockBasisSize, &sPMRows, &sPMCols);
		free(sMinus);
	} else if (*oldAOperator == 'M' || *oldAOperator == 'm') {
		sMinus = dagger(newSubBlockA->sPlusLastSite, newSubBlockA->nextBlockBasisSize, newSubBlockA->basisSize);
		sPM = kron(sMinus, oldSubBlockB->sPlusLastSite, 
				   newSubBlockA->basisSize, newSubBlockA->nextBlockBasisSize,
				   oldSubBlockB->nextBlockBasisSize, oldSubBlockB->basisSize, &sPMRows, &sPMCols);		
		free(sMinus);
	} else {
		printf("Unknown oldAOperator for addSPMFromSubBlocksToH : %s.\n", oldAOperator);
	}
    addSubBlock(block->H, block->basisSize, block->basisSize, JPM, sPM, sPMRows, sPMCols, newSubBlockIndex, oldSubBlockIndex, "N");
    addSubBlock(block->H, block->basisSize, block->basisSize, JPM, sPM, sPMRows, sPMCols, oldSubBlockIndex, newSubBlockIndex, "Dagger");

	free(sPM);
}

// After adding a new sub-block to block, (kronned fron A[newSubInA], B[newSubInB]),
// Add its donation to the s+- matrix by going over all sub-blocks, finding the ones which connect
// with our sub-slock and adding the operator to block->H.
void addSPMDonationToH(MatrixBlock* block, BlockStatesDoc doc, 
		int newSubBlockIndex,
		MatrixBlock* A, int newSubInA, MatrixBlock* B, int newSubInB) {
	int currSubBlockIndex = 0;
	for (int l = 0; l < doc.statesDocsNum; l++) {
		MatrixBlock* currSubBlockA = &A[doc.statesDocs[l].ABlockIndex];
		MatrixBlock* currSubBlockB = &B[doc.statesDocs[l].BBlockIndex];
		if(currSubBlockA->sZ == A[newSubInA].sZ + 1) { // Add sPlus_A * sMinus_B donation to unified_H.
			addSPMFromSubBlocksToH(block, "M", currSubBlockA, currSubBlockB,
				&A[newSubInA], &B[newSubInB], currSubBlockIndex, newSubBlockIndex);
		} else if (currSubBlockA->sZ == A[newSubInA].sZ - 1) {
			addSPMFromSubBlocksToH(block, "P", currSubBlockA, currSubBlockB,
				&A[newSubInA], &B[newSubInB], currSubBlockIndex, newSubBlockIndex);
		}
		currSubBlockIndex += doc.statesDocs[l].subBlockSize;
	}
}


fcomplex* getsZLastSite(float sZ, int basisSize) {
	fcomplex* result = identity(basisSize);
	fcomplex sZComp = {.re = sZ, .im = 0};
	for (int i = 0; i < getArraySize(basisSize, basisSize); i++) { 
		result[i] = mult(sZComp, result[i]); 
	}
	return result;
}

// TODO restrict
// Creates a block of A x B states, Does not add the connection operators to the Hamiltonian.
void combineStates(MatrixBlock* A, int ABlocksNum, MatrixBlock* B, int BBlocksNum, 
		MatrixBlock** result, int* resultSize, BlockStatesDoc** doc) {
	MatrixBlock* blocks = malloc(sizeof(MatrixBlock) * ABlocksNum * BBlocksNum);
	int blocksCounter = 0;
	*doc = malloc(sizeof(BlockStatesDoc) * ABlocksNum * BBlocksNum);
	for (int i = 0; i < ABlocksNum; i++) {
		for (int j = 0; j < BBlocksNum; j++) {
			int rows, cols;
			// TODO to separate func
			fcomplex* I_A = identity(A[i].basisSize);
			fcomplex* I_B = identity(B[j].basisSize);
			fcomplex* H_A = kron(A[i].H, I_B, A[i].basisSize, A[i].basisSize, B[j].basisSize, B[j].basisSize, &rows, &cols);	
			fcomplex* H_B = kron(I_A, B[j].H, A[i].basisSize, A[i].basisSize, B[j].basisSize, B[j].basisSize, &rows, &cols);
			fcomplex* H = malloc(sizeof(fcomplex) * getArraySize(rows, cols));
			for (int k = 0; k < getArraySize(rows, cols); k++) H[k] = add(H_A[k], H_B[k]);
			fcomplex* sZLastSite = getsZLastSite(A[i].sZ, A[i].basisSize * B[j].basisSize); // This matrix will be used only if A is the new site, so we assume A[i] here is a single site.
			fcomplex* sZsZ = kron(A[i].sZLastSite, B[j].sZLastSite, A[i].basisSize, A[i].basisSize, B[j].basisSize, B[j].basisSize, &rows, &cols);
			for (int k = 0; k < getArraySize(rows, cols); k++) H[k] = add(H[k], mult(sZsZ[k], JZ));
	
			blocks[blocksCounter] = (MatrixBlock) {
				.elementsNum = A[i].elementsNum + B[j].elementsNum, 
				.basisSize = A[i].basisSize * B[j].basisSize,
				.H = H,
				.sZLastSite = sZLastSite,
				.sZ = A[i].sZ + B[j].sZ
			};
			MatrixBlock* block = blocks + blocksCounter;
			free(I_A);
			free(I_B);
			free(H_A);
			free(H_B);
			free(sZsZ);

			int foundMatching = 0;
			for (int k = 0; k < blocksCounter; k++) { // TODO better way for this?
				if (block->sZ == blocks[k].sZ) {
					MatrixBlock* unified = toSingleMatrixBlock(&blocks[k], block);
					addSPMDonationToH(unified, (*doc)[k], blocks[k].basisSize, A, i, B, j);

					// TODO to separate func
					int index = (*doc)[k].statesDocsNum;
					StatesDoc* statesDocs = malloc(sizeof(StatesDoc) * (index + 1));
					memcpy(statesDocs, (*doc)[k].statesDocs, sizeof(StatesDoc) * (index));
					free((*doc)[k].statesDocs);
					(*doc)[k].statesDocs = statesDocs;
					// (*doc)[k].statesDocs = realloc((*doc)[k].statesDocs, sizeof(StatesDoc) * (index + 1));
					(*doc)[k].statesDocs[index] = (StatesDoc) {.ABlockIndex = i, .BBlockIndex = j, .subBlockSize = block->basisSize};
					(*doc)[k].statesDocsNum++;

					freeMatrixBlock(&blocks[k]);
					freeMatrixBlock(block);
					blocks[k] = *unified;
					foundMatching = 1;
					break;
				}
			}
			if (!foundMatching) {
				// TODO to separate func
				StatesDoc* statesDocs = malloc(sizeof(StatesDoc));
				statesDocs[0] = (StatesDoc) {.ABlockIndex = i, .BBlockIndex = j, .subBlockSize = block->basisSize}; 
				(*doc)[blocksCounter] = (BlockStatesDoc) {.statesDocs = statesDocs, .statesDocsNum = 1};

				blocksCounter++;				
			}
		}
	}
	*result = blocks;
	*resultSize = blocksCounter;
}

void printMatrixBlock(char* desc, MatrixBlock* M) {
	printf("%s:\n", desc);
	printf("elementsNum: %d\n", M->elementsNum);
	printf("basisSize: %d\n", M->basisSize);
	printMatrix("H", M->H, M->basisSize, M->basisSize);
	printMatrix("sPlusLastSite", M->sPlusLastSite, M->nextBlockBasisSize, M->basisSize);
	printMatrix("sZLastSite", M->sZLastSite, M->basisSize, M->basisSize);
	printf("sZ: %.1f\n", M->sZ);
	printf("\n");
}

void addSubBlock(fcomplex* M, int MRowsNum, int MColsNum, fcomplex alpha, fcomplex* m, int mRowsNum, int mColsNum, int firstRow, int firstCol, char* dagger) {
	if (m == NULL) return;
	for (int i = 0; i < mRowsNum; i++) {
		for (int j = 0; j < mColsNum; j++) {
			fcomplex mEntry = m[getArrayIndex(i, j, mRowsNum, mColsNum)];
			int row, col;
			fcomplex MEntry;
			if (*dagger == 'D' || *dagger == 'd') {
				row = firstRow + j;
				col = firstCol + i;
				MEntry = conjugate(mEntry); 
			} else if (*dagger == 'N' || *dagger == 'n') {
				row = firstRow + i;
				col = firstCol + j;
				MEntry = mEntry;
			} else {
				printf("Unknown dagger option to addSubBlock.\n");
			}
			int currIndex = getArrayIndex(row, col, MRowsNum, MColsNum);
			M[currIndex] = add(mult(MEntry, alpha), M[currIndex]);
		}
	}
}

// Creates the matrix that transfers thisBlock -> next Block by applying S+ on thisBlock.
// preassumptions:
// 1. thisBlock.sZ = nextBlock.sZ - 1
// 2. In both blocks, the A block (left matrix in the kron operator) is our new site.
// 3. S+ for the subBlocks in thisBlock are not NULL unless this is the one with highest sz for the new site.
// 4. S+ for the subBlocks in thisBlock are in the proper dimensions.
void getSPlusLastSite(MatrixBlock* thisBlock, BlockStatesDoc* thisDoc, 
		MatrixBlock* nextBlock, BlockStatesDoc *nextDoc, MatrixBlock* A,
		fcomplex** result, int* resultRows, int* resultCols) {
	*resultCols = thisBlock->basisSize;
	*resultRows = nextBlock->basisSize;
	*result = malloc(sizeof(fcomplex) * getArraySize(*resultRows, *resultCols));
	for (int i = 0; i < getArraySize(*resultRows, *resultCols); i++) (*result)[i] = ZERO;
	int currSubBlockIndex = 0;
	for (int i = 0; i < thisDoc->statesDocsNum; i++) {
		MatrixBlock* thisSubBlock = &A[thisDoc->statesDocs[i].ABlockIndex];
		int thisSubBasisSize = thisDoc->statesDocs[i].subBlockSize;
		int nextSubBlockIndex = 0;
		for (int j = 0; j < nextDoc->statesDocsNum; j++) {
			MatrixBlock* nextSubBlock = &A[nextDoc->statesDocs[j].ABlockIndex];
			int nextSubBasisSize = nextDoc->statesDocs[j].subBlockSize;
			if (thisSubBlock->sZ == nextSubBlock->sZ - 1) {
				int oldSiteSize = thisSubBasisSize / thisSubBlock->basisSize;
				fcomplex* identity_oldSites = identity(oldSiteSize);
				int subSPlusRows, subSPlusCols;
				fcomplex* subSPlus = kron(thisSubBlock->sPlusLastSite, identity_oldSites,
					thisSubBlock->nextBlockBasisSize, thisSubBlock->basisSize,
					oldSiteSize, oldSiteSize,
					&subSPlusRows, &subSPlusCols);
				addSubBlock(*result, *resultRows, *resultCols, 
					ONE, subSPlus, subSPlusRows, subSPlusCols,
					nextSubBlockIndex, currSubBlockIndex, "N");
				free(identity_oldSites);
				free(subSPlus);
				break;
			}
			nextSubBlockIndex += nextSubBasisSize;
		}
		currSubBlockIndex += thisSubBasisSize;
	}
}

MatrixBlock* dmrgStep(MatrixBlock* A, int ABlocksNum, int* resultSize);

// Returns the eigenvector corresponding to the smallest eigenvalue of H, H is statesNum*statesNum.
fcomplex* getGroundStateForBlock(fcomplex* H, int statesNum, float* groundStateEnergy) {
 	// cheev changes the matrix H, and we want to keep our H intact for later.
	fcomplex* tempH = (fcomplex*) malloc(sizeof(fcomplex) * pow(statesNum, 2));
	memcpy(tempH, H, sizeof(fcomplex) * pow(statesNum, 2));
	float eigenvalues[statesNum];
	fcomplex eigenVectors[(int) pow(statesNum, 2)];
	int lwork = -1;
    fcomplex wkopt;
    fcomplex* work;
    float rwork[3*statesNum - 2];
    int info;

    cheev_( "Vectors", "Lower", &statesNum, tempH, &statesNum, eigenvalues, &wkopt, &lwork, rwork, &info);
    lwork = (int)wkopt.re;
    work = (fcomplex*)malloc( lwork*sizeof(fcomplex) );
    /* Solve eigenproblem */
    cheev_( "Vectors", "Lower", &statesNum, tempH, &statesNum, eigenvalues, work, &lwork, rwork, &info);
    /* Check for convergence */
    if( info > 0 ) {
       printf( "getGroundStateForBlock: cheev failed.\n" );
       exit(1);
    }
    *groundStateEnergy = eigenvalues[0];
    fcomplex* groundState = (fcomplex*) malloc(sizeof(fcomplex) * statesNum);
	memcpy(groundState, tempH, sizeof(fcomplex) * statesNum);
	free(tempH);
    free(work);
	return groundState;
}

void getDensityMatrixEigenVectors(fcomplex* rhoA, int rhoADim, float** eigenvalues) {
	*eigenvalues = malloc(sizeof(float) * rhoADim);
	int lwork = -1;
    fcomplex wkopt;
    fcomplex* work;
    float rwork[3*rhoADim - 2];
    int info;

    cheev_( "Vectors", "Lower", &rhoADim, rhoA, &rhoADim, *eigenvalues, &wkopt, &lwork, rwork, &info);
    lwork = (int)wkopt.re;
    work = (fcomplex*)malloc( lwork*sizeof(fcomplex) );
    /* Solve eigenproblem */
    cheev_( "Vectors", "Lower", &rhoADim, rhoA, &rhoADim, *eigenvalues, work, &lwork, rwork, &info);
    /* Check for convergence */
    if( info > 0 ) {
    	printf( "getDensityMatrixEigenVectors: cheev failed.\n" );
    	free(work);
       	exit(1);
    }
    free(work);
}

void getGroundState(MatrixBlock* blocks, int blocksNum, BlockStatesDoc* docs, int* groundStateBlockIndex, fcomplex** groundState) {
	float lowestEnergy;
	fcomplex* bestVec;
	int bestBlockIndex = 0;
	for (int i = 0; i < blocksNum; i++) {
		float blockGroundStateEnergy;
		fcomplex* blockGroundState = getGroundStateForBlock(blocks[i].H, blocks[i].basisSize, &blockGroundStateEnergy);
		if (i == 0 || blockGroundStateEnergy < lowestEnergy) {
			lowestEnergy = blockGroundStateEnergy;
			if (i != 0) free(bestVec);
			bestVec = blockGroundState;
			bestBlockIndex = i;
		}
	}
	*groundStateBlockIndex = bestBlockIndex;
	*groundState = bestVec;
}

// void getGroundState(MatrixBlock* blocks, int blocksNum, BlockStatesDoc* docs, int* groundStateBlockIndex, fcomplex*** groundState) {
// 	float lowestEnergy;
// 	fcomplex* bestVec;
// 	int bestBlockIndex = 0;
// 	for (int i = 0; i < blocksNum; i++) {
// 		float blockGroundStateEnergy;
// 		fcomplex* blockGroundState = getGroundStateForBlock(blocks[i].H, blocks[i].basisSize, &blockGroundStateEnergy);
// 		if (i == 0 || blockGroundStateEnergy < lowestEnergy) {
// 			lowestEnergy = blockGroundStateEnergy;
// 			if (i != 0) free(bestVec);
// 			bestVec = blockGroundState;
// 			bestBlockIndex = i;
// 		}
// 	}
// 	*groundStateBlockIndex = bestBlockIndex;
// 	fcomplex* result[docs[*groundStateBlockIndex].statesDocsNum];
// 	int currIndex = 0;
// 	for (int i = 0; i < docs[*groundStateBlockIndex].statesDocsNum; i++) {
// 		result[i] = bestVec + currIndex;
// 		currIndex += docs[*groundStateBlockIndex].statesDocs[i].subBlockSize;
// 	}
// 	*groundState = result;
// }

void copyMatrixBlock(MatrixBlock* source, MatrixBlock* dest);

int main() {
	initSingleSite();
	MatrixBlock* blocks = malloc(sizeof(MatrixBlock) * 2);
	for (int i = 0; i < 2; i++) copyMatrixBlock(NEW_STATE_MATRIX_BLOCK + i, blocks + i);

	int nextStepBlocksNum = 2;
	for (int i = 0; i <= 7; i++) {
		int thisStepBlocksNum = nextStepBlocksNum;
		blocks = dmrgStep(blocks, thisStepBlocksNum, &nextStepBlocksNum);
		printf("nextStepBlocksNum = %d\n", nextStepBlocksNum);
	}
	// exact(5);
}

// Auxilary funcs

// Returns the KrONEcker product of A and B.
// [a11 * B a12*B ...]
// [a21 * B ...      ]
// Allocates the matrix!
fcomplex* kron(const fcomplex* restrict A, const fcomplex* restrict B, 
	int rowsNumA, int colsNumA, int rowsNumB, int colsNumB,
	int* resultRowsNum, int* resultsColsNum) {
	*resultRowsNum = rowsNumA * rowsNumB;
	*resultsColsNum = colsNumA * colsNumB;
	int resultSize =  getArraySize(*resultRowsNum, *resultsColsNum);
	fcomplex* result = (fcomplex*) malloc(sizeof(fcomplex) * resultSize);
	for (int i = 0; i < resultSize; i++) result[i] = ONE;
	for (int i = 0; i < resultSize; i++) {
		int row, col;
		getEntryIndices(i, *resultRowsNum, *resultsColsNum, &row, &col);
		int aRow = row / rowsNumB, aCol = col / colsNumB;
		result[i] = mult(result[i], A[getArrayIndex(aRow, aCol, rowsNumA, colsNumA)]);
		int bRow = row % rowsNumB, bCol = col % colsNumB;

		result[i] = mult(result[i], B[getArrayIndex(bRow, bCol, rowsNumB, colsNumB)]);	
	}
	return result;
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


void printMatrix(char* desc, fcomplex* M, int rowsNum, int colsNum) {
	printf("%s:\n", desc);
	for (int row = 0; row < rowsNum; row++) {
		for (int col = 0; col < colsNum; col++) {
			// printf("%.2f + %.2fi\t", M[row*colsNum + col].re, M[row*colsNum + col].im);
			printf("%.2f\t", M[row*colsNum + col].re);
		}
		printf("\n");
	}
}

// Here we add the donation of the new site to the matrixBlock Hamiltonian.
// void updateMatrixBlockHForNewSite(MatrixBlock* matrixBlock, fcomplex* H, fcomplex* sPlusNewSite, fcomplex* sZNewSite, int matrixBlockStatesNum) {
// 	// cgemm assigns C = alpha * AB + beta * C
// 	// todo better way for this?
// 	fcomplex* I = identity(SITES_STATE_NUM);
// 	fcomplex* expandedSPlusLastSite = kron(I, matrixBlock->sPlusLastSite, SITES_STATE_NUM, SITES_STATE_NUM, matrixBlock->basisSize, matrixBlock->basisSize);
// 	fcomplex* expandedSZLastSite = kron(I, matrixBlock->sZLastSite, SITES_STATE_NUM, SITES_STATE_NUM, matrixBlock->basisSize, matrixBlock->basisSize);
// 	free(I);

// 	cgemm_("N", "T", &matrixBlockStatesNum, &matrixBlockStatesNum, &matrixBlockStatesNum, &j, expandedSPlusLastSite, &matrixBlockStatesNum, 
// 		sPlusNewSite, &matrixBlockStatesNum, &ONE, H, &matrixBlockStatesNum);	
// 	cgemm_("T", "N", &matrixBlockStatesNum, &matrixBlockStatesNum, &matrixBlockStatesNum, &j, expandedSPlusLastSite, &matrixBlockStatesNum, 
// 		sPlusNewSite, &matrixBlockStatesNum, &ONE, H, &matrixBlockStatesNum);	
// 	cgemm_("N", "T", &matrixBlockStatesNum, &matrixBlockStatesNum, &matrixBlockStatesNum, &j, expandedSZLastSite, &matrixBlockStatesNum, 
// 		sZNewSite, &matrixBlockStatesNum, &ONE, H, &matrixBlockStatesNum);	
// 	free(expandedSZLastSite);
// 	free(expandedSPlusLastSite);
// }

// // Returns the H of the full lattice.
// fcomplex* getFullH(fcomplex* H_A, fcomplex* sPlusNewSite, fcomplex* sZNewSite, int matrixBlockStatesNum, int fullStatesNum) {
// 	fcomplex* H_B = (fcomplex*) malloc(sizeof(fcomplex) * getArraySize(matrixBlockStatesNum, matrixBlockStatesNum));
// 	memcpy(H_B, H_A, sizeof(fcomplex) * getArraySize(matrixBlockStatesNum, matrixBlockStatesNum));
// 	fcomplex* fullH = kron(H_A, H_B, matrixBlockStatesNum, matrixBlockStatesNum, matrixBlockStatesNum, matrixBlockStatesNum);
// 	free(H_B);

// 	// calculate the donation of the connection of the two new sites to H.
// 	fcomplex* I = identity(matrixBlockStatesNum);
// 	fcomplex* sPlusA = kron(sPlusNewSite, I, matrixBlockStatesNum, matrixBlockStatesNum, matrixBlockStatesNum, matrixBlockStatesNum);
// 	fcomplex* sPlusB = kron(I, sPlusNewSite, matrixBlockStatesNum, matrixBlockStatesNum, matrixBlockStatesNum, matrixBlockStatesNum);
// 	fcomplex* sZA = kron(sZNewSite, I, matrixBlockStatesNum, matrixBlockStatesNum, matrixBlockStatesNum, matrixBlockStatesNum);
// 	fcomplex* sZB = kron(I, sZNewSite, matrixBlockStatesNum, matrixBlockStatesNum, matrixBlockStatesNum, matrixBlockStatesNum);
// 	cgemm_("N", "T", &fullStatesNum, &fullStatesNum, &fullStatesNum, &j, 
// 		sPlusA, &fullStatesNum, sPlusB, &fullStatesNum, &ONE, fullH, &fullStatesNum);	
// 	cgemm_("T", "N", &fullStatesNum, &fullStatesNum, &fullStatesNum, &j, 
// 		sPlusA, &fullStatesNum, sPlusB, &fullStatesNum, &ONE, fullH, &fullStatesNum);	
// 	cgemm_("N", "N", &fullStatesNum, &fullStatesNum, &fullStatesNum, &jz, 
// 		sZA, &fullStatesNum, sZB, &fullStatesNum, &ONE, fullH, &fullStatesNum);	
// 	free(I);
// 	free(sPlusA);
// 	free(sPlusB);
// 	free(sZA);
// 	free(sZB);
// 	return fullH;
// 

// // Returns the reduced density matrix. Treats the ground state as a matrix:
// // groundState = sum(psi_ij * |A_i>|B_j>).
// // psi* psi^dagger is our density matrix.
// fcomplex* getReducedDensityMatrix(fcomplex* groundState, int matrixBlockStatesNum) {
// 	fcomplex* rhoA = (fcomplex*) malloc(sizeof(fcomplex) * getArraySize(matrixBlockStatesNum, matrixBlockStatesNum));
// 	cgemm_("N", "T", &matrixBlockStatesNum, &matrixBlockStatesNum, &matrixBlockStatesNum, &ONE, 
// 		groundState, &matrixBlockStatesNum, groundState, &matrixBlockStatesNum, 
// 		&ZERO, rhoA, &matrixBlockStatesNum);	
// 	return rhoA;
// }

// Get new basis using SVD on the ground state in matrix form.
// psi is a ABlockStatesNum*BBlockStatesNum matrix.
fcomplex* getNewBasis(fcomplex* psi, int ABlockStatesNum, int BBlockStatesNum, int* newBasisSize) {
	// Perform SVD decomposition.
	int info;
	fcomplex wkopt;
    fcomplex* work;
    /* Local arrays */
    /* iwork dimension should be at least 8*min(m,n) */
    int minDim = ABlockStatesNum < BBlockStatesNum ? ABlockStatesNum : BBlockStatesNum;
    int iwork[8*minDim];
    /* rwork dimension should be at least 5*(min(m,n))**2 + 7*min(m,n)) */
    float S[minDim], rwork[5 * minDim * minDim + 7 * minDim];
    fcomplex VT[BBlockStatesNum * minDim];
    fcomplex U[ABlockStatesNum * minDim];
	int lwork = -1;
	cgesvd_("All", "All", &ABlockStatesNum, &BBlockStatesNum, psi, &ABlockStatesNum, S, U, &ABlockStatesNum, VT, &BBlockStatesNum, &wkopt, &lwork, rwork, &info);
    lwork = (int)wkopt.re;
    work = (fcomplex*) malloc(lwork*sizeof(fcomplex));
    /* Compute SVD */
    cgesvd_("All", "All", &ABlockStatesNum, &BBlockStatesNum, psi, &ABlockStatesNum, S, U, &ABlockStatesNum, VT, &BBlockStatesNum, work, &lwork, rwork, &info);
    /* Check for convergence */
    if( info > 0 ) {
        printf( "The algorithm computing SVD failed to converge.\n" );
        exit( 1 );
    }

    // Use only min(D, matrixBlockStatesNum) rows of U corresponding to the largest singular values.
    *newBasisSize = ABlockStatesNum < D ? ABlockStatesNum : D;
    fcomplex* UDagger = dagger(U, ABlockStatesNum, minDim);
    UDagger = (fcomplex*) realloc(U, sizeof(fcomplex) * ABlockStatesNum * (*newBasisSize));

    free( (void*)work );
    return UDagger;
}



// fcomplex* multiplyMatrices(fcomplex* A, int ARows, int ACols, fcomplex* B, int BCols) {
// 	fcomplex* result = malloc(sizeof(fcomplex) * getArraySize(ARows, BCols));

//  	cgemm_("N", "N", &ARows, &oldBasisSize, &oldBasisSize, &ONE, basis, &oldBasisSize, O, &oldBasisSize, &ZERO, result, &newBasisSize);
// 	return result;
// }

void copyMatrixBlock(MatrixBlock* source, MatrixBlock* dest) {
	fcomplex* H = malloc(sizeof(fcomplex) * getArraySize(source->basisSize, source->basisSize));
	memcpy(H, source->H, sizeof(fcomplex) * getArraySize(source->basisSize, source->basisSize));
	fcomplex* sZLastSite = malloc(sizeof(fcomplex) * getArraySize(source->basisSize, source->basisSize));
	memcpy(sZLastSite, source->sZLastSite, sizeof(fcomplex) * getArraySize(source->basisSize, source->basisSize));
	*dest = (MatrixBlock) {
		.elementsNum = source->elementsNum,
		.basisSize = source->basisSize,
 		.nextBlockBasisSize = source->nextBlockBasisSize,
 		.sZLastSite = sZLastSite,
		.H = H,
		.sZ = source->sZ
	};
	if (source->nextBlockBasisSize != 0) {
		fcomplex* sPlusLastSite = malloc(sizeof(fcomplex) * getArraySize(source->nextBlockBasisSize, source->basisSize));	
		memcpy(sPlusLastSite, source->sPlusLastSite, sizeof(fcomplex) * getArraySize(source->nextBlockBasisSize, source->basisSize));
		dest->sPlusLastSite = sPlusLastSite;
	}
}

fcomplex* updateOperatorForNewBasis(fcomplex* O, int ORows, int OCols, 
		fcomplex* newBasisRowMatrixLeft, int newBasisLeftSize, fcomplex* newBasisRowMatrixRight, int newBasisRightSize) {
	fcomplex* temp = multiplyMatrices(newBasisRowMatrixLeft, newBasisLeftSize, ORows, "N", O, ORows, OCols, "N");
	fcomplex* result = multiplyMatrices(temp, newBasisLeftSize, OCols, "N", newBasisRowMatrixRight, newBasisRightSize, OCols, "Dagger");
	free(temp);
	return result;
}

MatrixBlock* dmrgStep(MatrixBlock* A, int ABlocksNum, int* resultSize) {

	MatrixBlock* expandedBlockA;
	int expandedBlocksNum;
	BlockStatesDoc* expandedBlockDoc;
	combineStates(NEW_STATE_MATRIX_BLOCK, SITE_STATES_NUM, A, ABlocksNum, &expandedBlockA, &expandedBlocksNum, &expandedBlockDoc);
	for (int i= 0; i < expandedBlocksNum; i++)  {
		 //TODO to separate func.
		MatrixBlock* thisBlock = &expandedBlockA[i];
		for (int j = 0; j < expandedBlocksNum; j++) {
			MatrixBlock* nextBlock = &expandedBlockA[j];
			if (thisBlock->sZ == nextBlock->sZ - 1) {
				fcomplex* sPlusLastSite;
				int sPlusLastSiteRows, sPlusLastSiteCols;
				getSPlusLastSite(thisBlock, &expandedBlockDoc[i], nextBlock, &expandedBlockDoc[j], 
					NEW_STATE_MATRIX_BLOCK, &sPlusLastSite, &sPlusLastSiteRows, &sPlusLastSiteCols);
				thisBlock->sPlusLastSite = sPlusLastSite;
				thisBlock->nextBlockBasisSize = sPlusLastSiteRows;
				break;
			}
			thisBlock->nextBlockBasisSize = 0;
		}
	}

	MatrixBlock expandedBlockB[expandedBlocksNum];
	for (int i = 0; i < expandedBlocksNum; i++) copyMatrixBlock(expandedBlockA + i, expandedBlockB + i);

	MatrixBlock* fullLattice;
	int fullLaticeBlocksNum;
	BlockStatesDoc* fullLatticeDoc;
	combineStates(expandedBlockA, expandedBlocksNum, expandedBlockB, expandedBlocksNum, 
		&fullLattice, &fullLaticeBlocksNum, &fullLatticeDoc);

	int groundStateBlockIndex;
	fcomplex* groundState;
	getGroundState(fullLattice, fullLaticeBlocksNum, fullLatticeDoc, &groundStateBlockIndex, &groundState);
	printf("line 755\n");
	printf("groundStateBlockIndex = %d\n", groundStateBlockIndex);
	printf("fullLatticeDoc[groundStateBlockIndex].statesDocsNum = %d\n", fullLatticeDoc[groundStateBlockIndex].statesDocsNum);
	int subBlocksNum = fullLatticeDoc[groundStateBlockIndex].statesDocsNum;
	int currIndex = 0;
	float* eigenvalues[subBlocksNum];
	fcomplex* eigenvectors[subBlocksNum];
	int subBlockStatesNum[subBlocksNum];
	int fullBasisSize = 0;
	for (int i = 0; i < subBlocksNum; i++) {
		int AStatesNum = expandedBlockA[fullLatticeDoc[groundStateBlockIndex].statesDocs[i].ABlockIndex].basisSize;
		int BStatesNum = expandedBlockB[fullLatticeDoc[groundStateBlockIndex].statesDocs[i].BBlockIndex].basisSize;
		fcomplex* rhoA = multiplyMatrices(groundState + currIndex, AStatesNum, BStatesNum, "N", groundState + currIndex, AStatesNum, BStatesNum, "D");
		getDensityMatrixEigenVectors(rhoA, AStatesNum, &eigenvalues[i]);
		eigenvectors[i] = rhoA;
		subBlockStatesNum[i] = AStatesNum;
		currIndex += AStatesNum * BStatesNum;
		fullBasisSize += AStatesNum;
	}
	free(groundState);

	int newBasisSize = fullBasisSize < D ? fullBasisSize : D;
	int currEigenvector[subBlocksNum];
	for (int i = 0; i < subBlocksNum; i++) currEigenvector[i] = subBlockStatesNum[i] - 1;
	for (int i = 0; i < newBasisSize; i++) {
		float largestEigenvalue = 0;
		int largestEigenvalueBlock = 0;
		for (int j = 1; j <  subBlocksNum; j++) 
			if (currEigenvector[j] >=0 && eigenvalues[j][currEigenvector[j]] >= largestEigenvalue) {
				largestEigenvalue = eigenvalues[j][currEigenvector[j]];
				largestEigenvalueBlock = j;
			}
		currEigenvector[largestEigenvalueBlock]--;
	}

	MatrixBlock* result = malloc(sizeof(MatrixBlock));
	*resultSize = 0;
	for (int i = 0; i < subBlocksNum; i++) {
		int ABlockIndex = fullLatticeDoc[groundStateBlockIndex].statesDocs[i].ABlockIndex;
		if (currEigenvector[i] < subBlockStatesNum[i] - 1) { // We are using more than 0 vectors from this subblock.
			int basisSize = subBlockStatesNum[i] - 1 - currEigenvector[i];
			fcomplex* basis = eigenvectors[i] + (expandedBlockA[ABlockIndex].basisSize * (currEigenvector[i] + 1));
printf("line 798 basisSize = %d\n", basisSize);
printMatrix("H", expandedBlockA[ABlockIndex].H, expandedBlockA[ABlockIndex].basisSize, expandedBlockA[ABlockIndex].basisSize);
printMatrix("basis", basis, basisSize, expandedBlockA[ABlockIndex].basisSize);
			fcomplex* H = updateOperatorForNewBasis(expandedBlockA[ABlockIndex].H, expandedBlockA[ABlockIndex].basisSize, expandedBlockA[ABlockIndex].basisSize, 
				basis, basisSize, basis, basisSize);
			fcomplex* sZLastSite = updateOperatorForNewBasis(expandedBlockA[ABlockIndex].sZLastSite, expandedBlockA[ABlockIndex].basisSize, expandedBlockA[ABlockIndex].basisSize,
				basis, basisSize, basis, basisSize);
			result[*resultSize] = (MatrixBlock) {
				.elementsNum = expandedBlockA[ABlockIndex].elementsNum,
				.basisSize = basisSize,
 				// .sPlusLastSite = sPlusLastSite,
 				// .nextBlockBasisSize
 				.sZLastSite = sZLastSite,
				.H = H,
				.sZ = expandedBlockA[ABlockIndex].sZ
			};
			for (int j = 0; j < subBlocksNum; j++) {
				int nextBlockIndex = fullLatticeDoc[groundStateBlockIndex].statesDocs[j].ABlockIndex;
				if (expandedBlockA[ABlockIndex].sZ == expandedBlockA[nextBlockIndex].sZ - 1) {
					int nextBasisSize = subBlockStatesNum[j] - 1 - currEigenvector[j];
					fcomplex* nextBasis = eigenvectors[j] + (expandedBlockA[nextBlockIndex].basisSize * (currEigenvector[j] + 1));
					fcomplex* sPlusLastSite = updateOperatorForNewBasis(expandedBlockA[ABlockIndex].sPlusLastSite, 
						expandedBlockA[ABlockIndex].nextBlockBasisSize, expandedBlockA[ABlockIndex].basisSize,
						nextBasis, nextBasisSize, basis, basisSize);
					result[*resultSize].sPlusLastSite = sPlusLastSite;
					result[*resultSize].nextBlockBasisSize = nextBasisSize;
					break;		
				}
				result[*resultSize].sPlusLastSite = NULL;
				result[*resultSize].nextBlockBasisSize = 0;
			}
			*resultSize = *resultSize + 1;
			result = realloc(result, sizeof(MatrixBlock) * (*resultSize + 1));
		}
	}


	
	for (int i = 0; i < expandedBlocksNum; i++) freeMatrixBlock(expandedBlockA + i);
	for (int i = 0; i < expandedBlocksNum; i++) freeMatrixBlock(expandedBlockB + i);
	for (int i = 0; i < expandedBlocksNum; i++) freeBlockStatesDoc(expandedBlockDoc + i);
	for (int i = 0; i < ABlocksNum; i++) freeMatrixBlock(A + i);
	for (int i = 0; i < fullLaticeBlocksNum; i++) freeMatrixBlock(fullLattice + i);
	for (int i = 0; i < fullLaticeBlocksNum; i++) freeBlockStatesDoc(fullLatticeDoc + i);
	for (int i = 0; i < subBlocksNum; i++) free(eigenvectors[i]);
	for (int i = 0; i < subBlocksNum; i++) free(eigenvalues[i]);
	free(A);
	free(expandedBlockDoc);
	free(fullLatticeDoc);
	free(expandedBlockA);
	free(fullLattice);
	
	return result;
 
}

void exact(int size) {
	int basisSize = pow(2, size);
	fcomplex H[basisSize*basisSize];
	for (int i = 0; i < basisSize; i++) {
		for (int j = 0; j < basisSize; j++) {
			int index = i*basisSize + j;
			H[index] = ZERO;
			if (i == j) {
				int state = i;
				for (int k = 0; k < size; k++) {
					float sZI = (state & 1) - 0.5;
					fcomplex c = {.re = sZI, .im = 0};
					H[index] = add(H[index], mult(M, c));
					float sZI1 = ((state & 2) / 2) - 0.5;
					c.re *= sZI1;
					H[index] = add(H[index], mult(JZ, c));
					state >>= 1;
				}
			} else {
				int xor = i ^ j; // We want the difference between i and j to be neighbours.
				int xori = (xor & i); // Each state should have one up and one down. 
				int xorj = (xor & j);
				if (i == 4 && j == 7) printf("xor = %d, xori = %d, xorj = %d\n", xor, xori, xorj);
				if (xori !=0 && xorj != 0) {
					while (xor > 0) {
						if (xor & 1 > 0) {
							xor >>= 1;
							if (xor & 1 > 0) {
								xor >>= 1;
								if (xor == 0) {
									H[index] = add(H[index], JPM);
								}
								else xor = 0;
							}
							else xor = 0;
						}
						xor >>= 1;
					}
				}
			}
		}
	}
	printMatrix("H exact", H, basisSize, basisSize);
}
