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

int fcomplexCounter = 0;
int matrixBlockCounter = 0;
int docCounter = 0;
int sDocCounter = 0;
int floatCounter = 0;

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

typedef struct {
	int elementsNum;
	int basisSize;
 	fcomplex* sPlusLastSite; // represents the block acting on this site which will take us to the next block.
 	int nextBlockBasisSize; // this is the number of rows in sPlusLastSite.
 	fcomplex* sZLastSite;
	fcomplex* H;
	float sZ;
} MatrixBlock;

// TODO rename stuff
typedef struct {
	int ABlockIndex;
	int BBlockIndex;
	int subBlockSize;
} StatesDoc;

typedef struct {
	StatesDoc* statesDocs;
	int statesDocsNum;
} BlockStatesDoc;

// Global consts TODO static these
const int D = 100;
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
	NEW_STATE_MATRIX_BLOCK[1] = (MatrixBlock) {.elementsNum = 1, .basisSize = 1, .sPlusLastSite = NULL, .nextBlockBasisSize = 0, .sZLastSite = S_Z_UP_STATE, .H = H_UP_STATE, .sZ = 0.5};
}

float* mallocFloatArr(int size) {
	floatCounter++;
	return (float*) malloc(sizeof(float) * size);
}

void freeFlaotArr(float* v) {
	free(v);
	floatCounter--;
}

fcomplex* mallocFcomplexArr(int size) {
	fcomplexCounter++;
	return (fcomplex*) malloc(sizeof(fcomplex) * size);
}

void freeFcomplexArr(fcomplex* v) {
	free(v);
	fcomplexCounter--;
}

MatrixBlock* mallocMatrixBlockArr(int size) {
	matrixBlockCounter++;
	return (MatrixBlock*) malloc(sizeof(MatrixBlock) * size);
}

void freeMatrixBlockArr(MatrixBlock* v) {
	free(v);
	matrixBlockCounter--;
}

void freeMatrixBlock(MatrixBlock* matrixBlock) {
	if (matrixBlock->sPlusLastSite != NULL) freeFcomplexArr(matrixBlock->sPlusLastSite);
	freeFcomplexArr(matrixBlock->sZLastSite);
	freeFcomplexArr(matrixBlock->H);
}

StatesDoc* mallocStatesDocArr(int size) {
	sDocCounter++;
	return (StatesDoc*) malloc(sizeof(StatesDoc) * size);
}

void freeStatesDocArr(StatesDoc* v) {
	free(v);
	sDocCounter--;
}

void printStatesDoc(StatesDoc* doc) {
	printf("A block index = %d, B block index = %d\n", doc->ABlockIndex, doc->BBlockIndex);
}

BlockStatesDoc* mallocBlockStatesDocArr(int size) {
	docCounter++;
	return (BlockStatesDoc*) malloc(sizeof(BlockStatesDoc) * size);
}

void freeBlockStatesDocArr(BlockStatesDoc* v) {
	free(v);
	docCounter--;
}


void freeBlockStatesDoc(BlockStatesDoc* doc) {
	freeStatesDocArr(doc->statesDocs);
}

void identity(fcomplex* result, int n);
void printMatrix(char* desc, fcomplex* M, int rowsNum, int colsNum);
void dagger(fcomplex* M, int rowsNum, int colsNum, fcomplex* result);
void kron(const fcomplex* restrict A, const fcomplex* restrict B, 
	int rowsNumA, int colsNumA, int rowsNumB, int colsNumB, fcomplex* result);
void multiplyMatrices(fcomplex* A, int ARows, int ACols, char* daggerA, fcomplex* B, int BRows, int BCols, char* daggerB, 
		fcomplex* result);
void addSubBlock(fcomplex* M, int MRowsNum, int MColsNum, fcomplex alpha, fcomplex* m, int mRowsNum, int mColsNum, int firstRow, int firstCol, char* dagger);
// Returns a matrix constructed of A and B as blocks, i.e.
// ( A 0 )
// ( 0 B )
void toSingleMatrix(fcomplex* A, int dimA, fcomplex* B, int dimB, fcomplex* result);
void toSingleMatrixBlock(MatrixBlock* A, MatrixBlock* B, MatrixBlock* result);

int getArrayIndex(int row, int col, int rowsNum, int colsNum) {
	return row * colsNum + col;
}

int getArraySize(int rowsNum, int colsNum) {
	return rowsNum * colsNum;
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

void printBlockStatesDoc(char* desc, BlockStatesDoc* doc) {
	printf("%s\n", desc);
	for (int i = 0; i < doc->statesDocsNum; i++) {
		printStatesDoc(&(doc->statesDocs[i]));
	}
}

void printMatrixBlock(char* desc, MatrixBlock* M);

void addSPMFromSubBlocksToH(MatrixBlock* block, char* oldAOperator, 
		MatrixBlock* oldSubBlockA, MatrixBlock* oldSubBlockB,
		MatrixBlock* newSubBlockA, MatrixBlock* newSubBlockB,
		int oldSubBlockIndex, int newSubBlockIndex) {
	int sPMRows = oldSubBlockA->nextBlockBasisSize * newSubBlockB->basisSize;
	int sPMCols = oldSubBlockA->basisSize * newSubBlockB->nextBlockBasisSize;
	fcomplex sPM[sPMRows * sPMCols];
	if (*oldAOperator == 'P' || *oldAOperator == 'p') {
		fcomplex sMinus[getArraySize(newSubBlockB->nextBlockBasisSize, newSubBlockB->basisSize)];
		dagger(newSubBlockB->sPlusLastSite, newSubBlockB->nextBlockBasisSize, newSubBlockB->basisSize, sMinus);
		kron(oldSubBlockA->sPlusLastSite, sMinus, 
				   oldSubBlockA->nextBlockBasisSize, oldSubBlockA->basisSize,
				   newSubBlockB->basisSize, newSubBlockB->nextBlockBasisSize, sPM);
	} else if (*oldAOperator == 'M' || *oldAOperator == 'm') {
		fcomplex sMinus[getArraySize(newSubBlockA->nextBlockBasisSize, newSubBlockA->basisSize)];
		dagger(newSubBlockA->sPlusLastSite, newSubBlockA->nextBlockBasisSize, newSubBlockA->basisSize, sMinus);
		kron(sMinus, oldSubBlockB->sPlusLastSite, 
				   newSubBlockA->basisSize, newSubBlockA->nextBlockBasisSize,
				   oldSubBlockB->nextBlockBasisSize, oldSubBlockB->basisSize, sPM);		
	} else {
		printf("Unknown oldAOperator for addSPMFromSubBlocksToH : %s.\n", oldAOperator);
	}
    addSubBlock(block->H, block->basisSize, block->basisSize, JPM, sPM, sPMRows, sPMCols, newSubBlockIndex, oldSubBlockIndex, "N");
    addSubBlock(block->H, block->basisSize, block->basisSize, JPM, sPM, sPMRows, sPMCols, oldSubBlockIndex, newSubBlockIndex, "Dagger");
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
	fcomplex* result = mallocFcomplexArr(getArraySize(basisSize, basisSize));
	identity(result, basisSize);
	fcomplex sZComp = {.re = sZ, .im = 0};
	for (int i = 0; i < getArraySize(basisSize, basisSize); i++) { 
		result[i] = mult(sZComp, result[i]); 
	}
	return result;
}

void copyMatrixBlock(MatrixBlock* source, MatrixBlock* dest);

// TODO restrict
// Creates a block of A x B states, Does not add the connection operators to the Hamiltonian.
void combineStates(MatrixBlock* restrict A, int ABlocksNum, MatrixBlock* restrict B, int BBlocksNum, 
		MatrixBlock** result, int* resultSize, BlockStatesDoc** doc) {
	MatrixBlock* blocks = mallocMatrixBlockArr(ABlocksNum * BBlocksNum);
	int blocksCounter = 0;
	*doc = mallocBlockStatesDocArr(ABlocksNum * BBlocksNum);
	for (int i = 0; i < ABlocksNum; i++) {
		for (int j = 0; j < BBlocksNum; j++) {
			// TODO to separate func
			int combinedBasisSize = A[i].basisSize * B[j].basisSize;
			int combinedArraySize = getArraySize(combinedBasisSize, combinedBasisSize);
			fcomplex I_A[getArraySize(A[i].basisSize, A[i].basisSize)], I_B[getArraySize(B[j].basisSize, B[j].basisSize)],
				H_A[combinedArraySize], H_B[combinedArraySize];
			identity(I_A, A[i].basisSize);
			identity(I_B, B[j].basisSize);
			kron(A[i].H, I_B, A[i].basisSize, A[i].basisSize, B[j].basisSize, B[j].basisSize, H_A);	
			kron(I_A, B[j].H, A[i].basisSize, A[i].basisSize, B[j].basisSize, B[j].basisSize, H_B);
			fcomplex* H = mallocFcomplexArr(combinedArraySize);
			for (int k = 0; k < combinedArraySize; k++) H[k] = add(H_A[k], H_B[k]);
			fcomplex* sZLastSite = getsZLastSite(A[i].sZ, A[i].basisSize * B[j].basisSize); // This matrix will be used only if A is the new site, so we assume A[i] here is a single site.
			fcomplex sZsZ[getArraySize(combinedBasisSize, combinedBasisSize)]; 
			kron(A[i].sZLastSite, B[j].sZLastSite, A[i].basisSize, A[i].basisSize, B[j].basisSize, B[j].basisSize, sZsZ);
			for (int k = 0; k < combinedArraySize; k++) H[k] = add(H[k], mult(sZsZ[k], JZ));
	
			blocks[blocksCounter] = (MatrixBlock) {
				.elementsNum = A[i].elementsNum + B[j].elementsNum, 
				.basisSize = A[i].basisSize * B[j].basisSize,
				.H = H,
				.sZLastSite = sZLastSite,
				.sPlusLastSite = NULL,
				.nextBlockBasisSize = 0,
				.sZ = A[i].sZ + B[j].sZ
			};
			MatrixBlock* block = blocks + blocksCounter;
		
			int foundMatching = 0;
			for (int k = 0; k < blocksCounter; k++) { // TODO better way for this?
				if (block->sZ == blocks[k].sZ) {
					MatrixBlock oldBlock;
					copyMatrixBlock(blocks + k, &oldBlock);
					freeMatrixBlock(blocks + k);
					toSingleMatrixBlock(&oldBlock, block, blocks + k);
					addSPMDonationToH(blocks + k, (*doc)[k], oldBlock.basisSize, A, i, B, j);

					// TODO to separate func
					int index = (*doc)[k].statesDocsNum;
					StatesDoc* statesDocs = mallocStatesDocArr(index + 1);
					memcpy(statesDocs, (*doc)[k].statesDocs, sizeof(StatesDoc) * (index));
					freeBlockStatesDoc((*doc) + k);
					(*doc)[k].statesDocs = statesDocs;
					// (*doc)[k].statesDocs = realloc((*doc)[k].statesDocs, sizeof(StatesDoc) * (index + 1));
					(*doc)[k].statesDocs[index] = (StatesDoc) {.ABlockIndex = i, .BBlockIndex = j, .subBlockSize = block->basisSize};
					(*doc)[k].statesDocsNum++;

					freeMatrixBlock(&oldBlock);
					freeMatrixBlock(block);
					foundMatching = 1;
					break;
				}
			}
			if (!foundMatching) {
				// TODO to separate func
				StatesDoc* statesDocs = mallocStatesDocArr(1);
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
	*result = mallocFcomplexArr(getArraySize(*resultRows, *resultCols));
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
				int oldSiteSize = thisSubBasisSize / thisSubBlock->basisSize;  // TODO rename

				fcomplex identity_oldSites[getArraySize(oldSiteSize, oldSiteSize)]; 
				identity(identity_oldSites, oldSiteSize);
				int subSPlusRows = thisSubBlock->nextBlockBasisSize * oldSiteSize, subSPlusCols =  thisSubBlock->basisSize * oldSiteSize;
				fcomplex subSPlus[getArraySize(subSPlusRows, subSPlusCols)];
				kron(thisSubBlock->sPlusLastSite, identity_oldSites,
					thisSubBlock->nextBlockBasisSize, thisSubBlock->basisSize,
					oldSiteSize, oldSiteSize, subSPlus);
				addSubBlock(*result, *resultRows, *resultCols, 
					ONE, subSPlus, subSPlusRows, subSPlusCols,
					nextSubBlockIndex, currSubBlockIndex, "N");
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
	fcomplex* tempH = mallocFcomplexArr(pow(statesNum, 2));
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
    work = mallocFcomplexArr(lwork);
    /* Solve eigenproblem */
    cheev_( "Vectors", "Lower", &statesNum, tempH, &statesNum, eigenvalues, work, &lwork, rwork, &info);
    /* Check for convergence */
    if( info > 0 ) {
       printf( "getGroundStateForBlock: cheev failed.\n" );
       exit(1);
    }
    *groundStateEnergy = eigenvalues[0];
    fcomplex* groundState = mallocFcomplexArr(statesNum);
	memcpy(groundState, tempH, sizeof(fcomplex) * statesNum);
	freeFcomplexArr(tempH);
    freeFcomplexArr(work);
	return groundState;
}

void getDensityMatrixEigenVectors(fcomplex* rhoA, int rhoADim, float** eigenvalues) {
	*eigenvalues = mallocFloatArr(rhoADim);
	int lwork = -1;
    fcomplex wkopt;
    fcomplex* work;
    float rwork[3*rhoADim - 2];
    int info;

    cheev_( "Vectors", "Lower", &rhoADim, rhoA, &rhoADim, *eigenvalues, &wkopt, &lwork, rwork, &info);
    lwork = (int)wkopt.re;
    work = mallocFcomplexArr(lwork);
    /* Solve eigenproblem */
    cheev_( "Vectors", "Lower", &rhoADim, rhoA, &rhoADim, *eigenvalues, work, &lwork, rwork, &info);
    /* Check for convergence */
    if( info > 0 ) {
    	printf( "getDensityMatrixEigenVectors: cheev failed.\n" );
    	freeFcomplexArr(work);
       	exit(1);
    }
    freeFcomplexArr(work);
}

void getGroundState(MatrixBlock* blocks, int blocksNum, BlockStatesDoc* docs, int* groundStateBlockIndex, fcomplex** groundState) {
	// for (int i = 0; i < blocksNum; i++)printMatrixBlock("", &blocks[i]);
	float lowestEnergy;
	fcomplex* bestVec;
	int bestBlockIndex = 0;
	for (int i = 0; i < blocksNum; i++) {
		float blockGroundStateEnergy;
		fcomplex* blockGroundState = getGroundStateForBlock(blocks[i].H, blocks[i].basisSize, &blockGroundStateEnergy);
		if (i == 0 || blockGroundStateEnergy < lowestEnergy) {
			lowestEnergy = blockGroundStateEnergy;
			if (i != 0) freeFcomplexArr(bestVec);
			bestVec = blockGroundState;
			bestBlockIndex = i;
		} else freeFcomplexArr(blockGroundState);
	}
	printf("E0 = %f\n", lowestEnergy);
	*groundStateBlockIndex = bestBlockIndex;
	*groundState = bestVec;
}

int main() {
	initSingleSite();
	MatrixBlock* blocks = mallocMatrixBlockArr(2);
	for (int i = 0; i < 2; i++) copyMatrixBlock(NEW_STATE_MATRIX_BLOCK + i, blocks + i);

	int nextStepBlocksNum = 2;
	for (int i = 0; i < 10; i++) {
		printf("--------------------------------------------------------------------\n");
		int thisStepBlocksNum = nextStepBlocksNum;
		blocks = dmrgStep(blocks, thisStepBlocksNum, &nextStepBlocksNum);
		exact(nextStepBlocksNum * 2);
		printf("nextStepBlocksNum = %d\n", nextStepBlocksNum);
		// for (int j = 0; j < nextStepBlocksNum; j++) printMatrixBlock("", &blocks[j]);
	}
	for (int i = 0; i < nextStepBlocksNum; i++) freeMatrixBlock(blocks + i);
	freeMatrixBlockArr(blocks);
	printf("matrixBlockCounter = %d, fcomplexCounter = %d, docCounter = %d, sDocCounter = %d\n", matrixBlockCounter, fcomplexCounter, docCounter, sDocCounter);
}

// Auxilary funcs

// Returns the KrONEcker product of A and B.
// [a11 * B a12*B ...]
// [a21 * B ...      ]
void kron(const fcomplex* restrict A, const fcomplex* restrict B, 
	int rowsNumA, int colsNumA, int rowsNumB, int colsNumB, fcomplex* result) {
	int resultRowsNum = rowsNumA * rowsNumB, resultColsNum = colsNumA * colsNumB;
	int resultSize =  getArraySize(resultRowsNum, resultColsNum);
	for (int i = 0; i < resultSize; i++) result[i] = ONE;
	for (int i = 0; i < resultSize; i++) {
		int row, col;
		getEntryIndices(i, resultRowsNum, resultColsNum, &row, &col);
		int aRow = row / rowsNumB, aCol = col / colsNumB;
		result[i] = mult(result[i], A[getArrayIndex(aRow, aCol, rowsNumA, colsNumA)]);
		int bRow = row % rowsNumB, bCol = col % colsNumB;

		result[i] = mult(result[i], B[getArrayIndex(bRow, bCol, rowsNumB, colsNumB)]);	
	}
}

void identity(fcomplex* result, int n) {
	int resultSize = getArraySize(n, n);
	for (int i = 0; i < resultSize; i++) {
		int row, col;
		getEntryIndices(i, n, n, &row, &col);
		if (row == col) {
			result[i] = ONE;
		} else {
			result[i] = ZERO;
		}
	}
}

void dagger(fcomplex* M, int rowsNum, int colsNum, fcomplex* result) {
	for (int i = 0; i < rowsNum; i++) {
		for (int j = 0; j < colsNum; j++) {
			result[getArrayIndex(j, i, colsNum, rowsNum)] = conjugate(M[getArrayIndex(i, j, rowsNum, colsNum)]);
		}
	}
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

void multiplyMatrices(fcomplex* A, int ARows, int ACols, char* daggerA, fcomplex* B, int BRows, int BCols, char* daggerB, 
		fcomplex* result) {
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
}

// Returns a matrix constructed of A and B as blocks, i.e.
// ( A 0 )
// ( 0 B )
// TODO this would be cleaner and a little fatser with memcpy for each row.
void toSingleMatrix(fcomplex* A, int dimA, fcomplex* B, int dimB, fcomplex* result) {
	int dimResult = dimA + dimB;
	int arraySize = pow(dimResult, 2);
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
}

void toSingleMatrixBlock(MatrixBlock* A, MatrixBlock* B, MatrixBlock* result) {
	int arraySize = getArraySize(A->basisSize + B->basisSize, A->basisSize + B->basisSize);
	fcomplex* H = mallocFcomplexArr(arraySize);
	toSingleMatrix(A->H, A->basisSize, B->H, B->basisSize, H);
	fcomplex* sZLastSite = mallocFcomplexArr(arraySize);
	toSingleMatrix(A->sZLastSite, A->basisSize, B->sZLastSite, B->basisSize, sZLastSite);
	*result = (MatrixBlock) {
		.elementsNum = A->elementsNum,
		.basisSize = A->basisSize + B->basisSize,
		.H = H,
		.sZLastSite = sZLastSite,
		.sPlusLastSite = NULL,
		.nextBlockBasisSize = 0,
		.sZ = A->sZ
	};
}

void copyMatrixBlock(MatrixBlock* source, MatrixBlock* dest) {
	fcomplex* H = mallocFcomplexArr(getArraySize(source->basisSize, source->basisSize));
	memcpy(H, source->H, sizeof(fcomplex) * getArraySize(source->basisSize, source->basisSize));
	fcomplex* sZLastSite = mallocFcomplexArr(getArraySize(source->basisSize, source->basisSize));
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
		fcomplex* sPlusLastSite = mallocFcomplexArr(getArraySize(source->nextBlockBasisSize, source->basisSize));	
		memcpy(sPlusLastSite, source->sPlusLastSite, sizeof(fcomplex) * getArraySize(source->nextBlockBasisSize, source->basisSize));
		dest->sPlusLastSite = sPlusLastSite;
		} else {
		dest->sPlusLastSite = NULL;
	}
}

fcomplex* updateOperatorForNewBasis(fcomplex* O, int ORows, int OCols, 
	fcomplex* newBasisRowMatrixLeft, int newBasisLeftSize, fcomplex* newBasisRowMatrixRight, int newBasisRightSize) {
	fcomplex temp[getArraySize(newBasisLeftSize, OCols)];
	multiplyMatrices(newBasisRowMatrixLeft, newBasisLeftSize, ORows, "N", O, ORows, OCols, "N", temp);
	fcomplex* result = mallocFcomplexArr(getArraySize(newBasisLeftSize, newBasisRightSize));
	multiplyMatrices(temp, newBasisLeftSize, OCols, "N", newBasisRowMatrixRight, newBasisRightSize, OCols, "Dagger", result);
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
	int subBlocksNum = fullLatticeDoc[groundStateBlockIndex].statesDocsNum;
	int currIndex = 0;
	float* eigenvalues[subBlocksNum];
	fcomplex* eigenvectors[subBlocksNum];
	int subBlockStatesNum[subBlocksNum];
	int fullBasisSize = 0;
	printf("subBlocksNum = %d\n", subBlocksNum);
	for (int i = 0; i < subBlocksNum; i++) {
		int AStatesNum = expandedBlockA[fullLatticeDoc[groundStateBlockIndex].statesDocs[i].ABlockIndex].basisSize;
		int BStatesNum = expandedBlockB[fullLatticeDoc[groundStateBlockIndex].statesDocs[i].BBlockIndex].basisSize;
		fcomplex* rhoA = mallocFcomplexArr(getArraySize(AStatesNum, AStatesNum));
		multiplyMatrices(groundState + currIndex, AStatesNum, BStatesNum, "N", groundState + currIndex, AStatesNum, BStatesNum, "D", rhoA);
		getDensityMatrixEigenVectors(rhoA, AStatesNum, &eigenvalues[i]);
		eigenvectors[i] = rhoA;
		subBlockStatesNum[i] = AStatesNum;
		currIndex += AStatesNum * BStatesNum;
		fullBasisSize += AStatesNum;
	}
	freeFcomplexArr(groundState);

	int newBasisSize = fullBasisSize < D ? fullBasisSize : D;
	printf("new basis size = %d\n", newBasisSize);
	int currEigenvector[subBlocksNum];
	for (int i = 0; i < subBlocksNum; i++) currEigenvector[i] = subBlockStatesNum[i] - 1;
	for (int i = 0; i < newBasisSize; i++) {
		float largestEigenvalue = -1;
		int largestEigenvalueBlock = 0;
		for (int j = 1; j <  subBlocksNum; j++) {
			if (currEigenvector[j] >=0 && eigenvalues[j][currEigenvector[j]] >= largestEigenvalue) {
				largestEigenvalue = eigenvalues[j][currEigenvector[j]];
				largestEigenvalueBlock = j;
			}
		}
		currEigenvector[largestEigenvalueBlock]--;
	}

	MatrixBlock* result = mallocMatrixBlockArr(1);
	*resultSize = 0;
	for (int i = 0; i < subBlocksNum; i++) {
		int ABlockIndex = fullLatticeDoc[groundStateBlockIndex].statesDocs[i].ABlockIndex;
		if (currEigenvector[i] < subBlockStatesNum[i] - 1) { // We are using more than 0 vectors from this subblock.
			int basisSize = subBlockStatesNum[i] - 1 - currEigenvector[i];
			fcomplex* basis = eigenvectors[i] + (expandedBlockA[ABlockIndex].basisSize * (currEigenvector[i] + 1));
			fcomplex* H = updateOperatorForNewBasis(expandedBlockA[ABlockIndex].H, expandedBlockA[ABlockIndex].basisSize, expandedBlockA[ABlockIndex].basisSize, 
				basis, basisSize, basis, basisSize);
			fcomplex* sZLastSite = updateOperatorForNewBasis(expandedBlockA[ABlockIndex].sZLastSite, expandedBlockA[ABlockIndex].basisSize, expandedBlockA[ABlockIndex].basisSize,
				basis, basisSize, basis, basisSize);
			result[*resultSize] = (MatrixBlock) {
				.elementsNum = expandedBlockA[ABlockIndex].elementsNum,
				.basisSize = basisSize,
 				.sZLastSite = sZLastSite,
 				.sPlusLastSite = NULL,
 				.nextBlockBasisSize = 0,
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
	for (int i = 0; i < subBlocksNum; i++) freeFcomplexArr(eigenvectors[i]); 
	for (int i = 0; i < subBlocksNum; i++) freeFlaotArr(eigenvalues[i]);
	freeMatrixBlockArr(A);
	freeBlockStatesDocArr(expandedBlockDoc);
	freeBlockStatesDocArr(fullLatticeDoc);
	freeMatrixBlockArr(expandedBlockA);
	freeMatrixBlockArr(fullLattice);
	
	return result;
 
}

void exact(int size) {
	int basisSize = (int) pow(2, size);
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
					float sZI1 = k < (size - 1) ? ((state & 2) / 2) - 0.5 : 0;
					c.re *= sZI1;
					H[index] = add(H[index], mult(JZ, c));
					state >>= 1;
				}
			} else {
				int xor = i ^ j; // We want the difference between i and j to be neighbours.
				int xori = (xor & i); // Each state should have one up and one down. 
				int xorj = (xor & j);
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
	// printMatrix("H exact", H, basisSize, basisSize);
	float eigenvalues[basisSize];
	fcomplex eigenVectors[(int) pow(basisSize, 2)];
	int lwork = -1;
    fcomplex wkopt;
    fcomplex* work;
    float rwork[3*basisSize - 2];
    int info;

    cheev_( "Vectors", "Lower", &basisSize, H, &basisSize, eigenvalues, &wkopt, &lwork, rwork, &info);
    lwork = (int)wkopt.re;
    work = mallocFcomplexArr(lwork);
    /* Solve eigenproblem */
    cheev_( "Vectors", "Lower", &basisSize, H, &basisSize, eigenvalues, work, &lwork, rwork, &info);
    /* Check for convergence */
    if( info > 0 ) {
       printf( "getGroundStateForBlock: cheev failed.\n" );
       exit(1);
    }
    printf("E0 = %f\n", eigenvalues[0]);
    freeFcomplexArr(work);
 }
