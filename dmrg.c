/*
* Classic DMRG implementation.
* Compile with: cc dmrg.c -L/usr/grads/lib -llapack -lblas -lm -o dmrg 
*/


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h> 
#include <unistd.h> 
#include <time.h>

#define LATTICE_SIZE 14
#define D 50
#define SITE_STATES_NUM 2

// TODO
// Smarter krons (use matrixBlock in matrices)
// Smarter matrix multiplication (for example when I know one matrix is kroned with identity)
// Pack matrices
// Upate operators smarter


// ----------------------------------------------------------------------------------------------------------------
// Auxilary functions, consts and structs
// ----------------------------------------------------------------------------------------------------------------

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
	float E0;
} MatrixBlock;

// After combining two lattice blocks, A and B, into one, we keep track of which A and B matrix blocks created each 
// block of the full lattice.
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
const fcomplex M = {.re = 0, .im = 0}, JPM = {.re = 1, .im = 0} , JZ = {.re = 0, .im = 0};
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
	if (size > 100000) {
		int i = 5;
	}
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

void printMatrix(char* desc, fcomplex* M, int rowsNum, int colsNum);
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

void printBlockStatesDoc(char* desc, BlockStatesDoc* doc) {
	printf("%s\n", desc);
	for (int i = 0; i < doc->statesDocsNum; i++) {
		printStatesDoc(&(doc->statesDocs[i]));
	}
}

// ----------------------------------------------------------------------------------------------------------------
// fconplex* matrix operations functions
// ----------------------------------------------------------------------------------------------------------------

void identity(fcomplex* result, int n);
void dagger(fcomplex* M, int rowsNum, int colsNum, fcomplex* result);
void kron(const fcomplex* restrict A, const fcomplex* restrict B, 
	int rowsNumA, int colsNumA, int rowsNumB, int colsNumB, fcomplex* result);
void multiplyMatrices(fcomplex* A, int ARows, int ACols, char* daggerA, fcomplex* B, int BRows, int BCols, char* daggerB, 
		fcomplex* result);
void addSubBlock(fcomplex* M, int MRowsNum, int MColsNum, fcomplex alpha, fcomplex* m, int mRowsNum, int mColsNum, int firstRow, int firstCol, char* dagger);
// Returns a matrix constructed of A and B as blocks, i.e.
// ( A 0 )
// ( 0 B )
void toSingleMatrix(fcomplex* restrict A, int dimA, fcomplex* restrict B, int dimB, fcomplex* result);
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

// ----------------------------------------------------------------------------------------------------------------
// MatrixBlock matrix operations functions
// ----------------------------------------------------------------------------------------------------------------

// combines a block A with block B, assuming they share there sZ value.
// Does not deal with sPlusLastSite member.
void toSingleMatrixBlock(MatrixBlock* restrict A, MatrixBlock* restrict B, MatrixBlock* result, char* combineMode);
void copyMatrixBlock(MatrixBlock* source, MatrixBlock* dest);

// ----------------------------------------------------------------------------------------------------------------
// DMRG step sub fyncs
// ----------------------------------------------------------------------------------------------------------------

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

// Creates a block of A x B states, Does not add the S+- operators to the Hamiltonian.
void combineStates(MatrixBlock* restrict A, int ABlocksNum, MatrixBlock* restrict B, int BBlocksNum, 
		MatrixBlock** result, int* resultSize, BlockStatesDoc** doc, char* combineMode) {
	MatrixBlock* blocks = mallocMatrixBlockArr(ABlocksNum * BBlocksNum);
	int blocksCounter = 0;
	*doc = mallocBlockStatesDocArr(ABlocksNum * BBlocksNum);
	for (int i = 0; i < ABlocksNum; i++) {
		for (int j = 0; j < BBlocksNum; j++) {
			
			// create the matrix block for the combined states of A and B.
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
			fcomplex* sZLastSite;
			if (*combineMode == 'f' || *combineMode == 'F') {
				sZLastSite = NULL;
			} else {
				sZLastSite = getsZLastSite(A[i].sZ, A[i].basisSize * B[j].basisSize); 
				// This matrix will be used only if A is the new site, so we assume A[i] here is a single site.
			}
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
		
			// If this is not the first block with the same sZ, unify our current block with the existing block of this sZ.
			int foundMatching = 0;
			for (int k = 0; k < blocksCounter; k++) { // TODO better way for this?
				if (block->sZ == blocks[k].sZ) {
					MatrixBlock oldBlock;
					copyMatrixBlock(blocks + k, &oldBlock);
					freeMatrixBlock(blocks + k);
					toSingleMatrixBlock(&oldBlock, block, blocks + k, combineMode);
					addSPMDonationToH(blocks + k, (*doc)[k], oldBlock.basisSize, A, i, B, j);

					// Update our book keeping structs.
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
				// Update our book keeping structs.
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

// Returns the eigenvector corresponding to the smallest eigenvalue of H, H is statesNum*statesNum.
fcomplex* getGroundStateForBlock(fcomplex* H, int statesNum, float* groundStateEnergy) {
	time_t start;
    start = time(NULL);
    time_t curr;
	curr = time(NULL);

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

void getGroundState(MatrixBlock* blocks, int blocksNum, BlockStatesDoc* docs, int* groundStateBlockIndex, fcomplex** groundState, float *E0) {
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
	*E0 = lowestEnergy;
	*groundStateBlockIndex = bestBlockIndex;
	*groundState = bestVec;
}

MatrixBlock* infiniteDmrgStep(MatrixBlock* A, int ABlocksNum, int ABasisSize, int* resultSize, int* resultBasisSize);
MatrixBlock* finiteDmrgStep(MatrixBlock* A, int ABlocksNum, MatrixBlock* B, int BBlocksNum, int* resultSize, int* resultBasisSize);
MatrixBlock* dmrgStep(MatrixBlock* A, int ABlocksNum, MatrixBlock* B, int BBlocksNum, int* resultSize, int* resultBasisSize);
void exact();

int main() {
	fcomplex sp[] = {ZERO, ZERO};
	fcomplex z[] = {ZERO};
	MatrixBlock empty = {.elementsNum = 0, .basisSize = 1, .sPlusLastSite = sp, .nextBlockBasisSize = 2,
 		.sZLastSite = z, .H = z, .sZ = 0};
	initSingleSite();
	MatrixBlock* currBlocks = mallocMatrixBlockArr(1);
	copyMatrixBlock(&empty, currBlocks);

	MatrixBlock* allBlocks[LATTICE_SIZE + 1];
	int allBlocksNums[LATTICE_SIZE + 1];
	allBlocks[0] = currBlocks;
	allBlocksNums[0] = 1;
	int nextStepBlocksNum = 1;
	int nextStepBasisSize = 1;
	int i;
	for (i = 1; i <= LATTICE_SIZE / 2; i++) {
		int thisStepBlocksNum = nextStepBlocksNum;
		int thisStepBasisSize = nextStepBasisSize;
		currBlocks = infiniteDmrgStep(currBlocks, thisStepBlocksNum, thisStepBasisSize, &nextStepBlocksNum, &nextStepBasisSize);
		allBlocks[i] = currBlocks;
		allBlocksNums[i] = nextStepBlocksNum;
	}
	// Expand A on account of B 
	for (i = LATTICE_SIZE / 2; i < LATTICE_SIZE; i++) {
		int thisStepBlocksNum = nextStepBlocksNum;
		int thisStepBasisSize = nextStepBasisSize;
		int bIndex = LATTICE_SIZE - 1 - i;
		currBlocks = finiteDmrgStep(currBlocks, thisStepBlocksNum, allBlocks[bIndex], allBlocksNums[bIndex], &nextStepBlocksNum, &nextStepBasisSize);
		allBlocks[i+1] = currBlocks;
		allBlocksNums[i+1] = nextStepBlocksNum;
	} 	
	// Expand B on account of A
	for (i = LATTICE_SIZE - 1; i > 0; i--) {
		int bIndex = LATTICE_SIZE - 1 - i;
		currBlocks = finiteDmrgStep(allBlocks[bIndex], allBlocksNums[bIndex], allBlocks[i], allBlocksNums[i], &nextStepBlocksNum, &nextStepBasisSize);
		for (int j = 0; j < allBlocksNums[bIndex + 1]; j++) freeMatrixBlock(&(allBlocks[bIndex + 1][j]));
		freeMatrixBlockArr(allBlocks[bIndex + 1]);
		allBlocks[bIndex + 1] = currBlocks;
		allBlocksNums[bIndex + 1] = nextStepBlocksNum;
	}
	// printMatrixBlock("final", allBlocks[LATTICE_SIZE - 1]);

	for (int i = 1; i < LATTICE_SIZE; i++)  {
		for (int j = 0; j < allBlocksNums[i]; j++) {
			freeMatrixBlock(allBlocks[i]+ j);
		}
	}
	// exact();
}

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

void multiplyMatrices(fcomplex* A, int ARows, int ACols, char* daggerA, fcomplex* B, int BRows, int BCols, char* daggerB, fcomplex* result) {
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
void toSingleMatrix(fcomplex* restrict A, int dimA, fcomplex* restrict B, int dimB, fcomplex* result) {
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

// combines a block A with block B, assuming they share there sZ value.
// Does not deal with sPlusLastSite member.
void toSingleMatrixBlock(MatrixBlock* restrict A, MatrixBlock* restrict B, MatrixBlock* result, char* combineMode) {
	int arraySize = getArraySize(A->basisSize + B->basisSize, A->basisSize + B->basisSize);
	fcomplex* H = mallocFcomplexArr(arraySize);
	toSingleMatrix(A->H, A->basisSize, B->H, B->basisSize, H);
	fcomplex* sZLastSite;
	if (*combineMode == 'F' || *combineMode == 'f') {
		sZLastSite = NULL;
	 } else {
	 	sZLastSite = mallocFcomplexArr(arraySize);
		toSingleMatrix(A->sZLastSite, A->basisSize, B->sZLastSite, B->basisSize, sZLastSite);
	}
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
	fcomplex* sZLastSite = NULL;
	if (source->sZLastSite != NULL) {
		sZLastSite = mallocFcomplexArr(getArraySize(source->basisSize, source->basisSize));
		memcpy(sZLastSite, source->sZLastSite, sizeof(fcomplex) * getArraySize(source->basisSize, source->basisSize));
	}
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

MatrixBlock* dmrgStep(MatrixBlock* A, int ABlocksNum, MatrixBlock* B, int BBlocksNum, int* resultSize, int* resultBasisSize) {
    
	MatrixBlock* fullLattice;
	int fullLaticeBlocksNum;
	BlockStatesDoc* fullLatticeDoc;
	combineStates(A, ABlocksNum, B, BBlocksNum,	&fullLattice, &fullLaticeBlocksNum, &fullLatticeDoc, "Full lattice");

	// Diagonalize the hamiltonian of the full lattice and get the ground state (in a (reduced) psi_ij form.)
	int groundStateBlockIndex;
	fcomplex* groundState;
	float E0;
	getGroundState(fullLattice, fullLaticeBlocksNum, fullLatticeDoc, &groundStateBlockIndex, &groundState, &E0);
	printf("elementsNum = %d, E0 = %f, sz = %f\n", fullLattice[0].elementsNum, E0, fullLattice[0].sZ);
		
	// Trace out the B sub block (using rhoA = psi*pis^dagger) and diagonalize it. 
	// This is done for each sub block separately.
	int subBlocksNum = fullLatticeDoc[groundStateBlockIndex].statesDocsNum;
	int currIndex = 0;
	float* eigenvalues[subBlocksNum];
	fcomplex* eigenvectors[subBlocksNum];
	int subBlockStatesNum[subBlocksNum];
	int fullBasisSize = 0;
	for (int i = 0; i < subBlocksNum; i++) {
		int AStatesNum = A[fullLatticeDoc[groundStateBlockIndex].statesDocs[i].ABlockIndex].basisSize;
		int BStatesNum = B[fullLatticeDoc[groundStateBlockIndex].statesDocs[i].BBlockIndex].basisSize;
		fcomplex* rhoA = mallocFcomplexArr(getArraySize(AStatesNum, AStatesNum));
		multiplyMatrices(groundState + currIndex, AStatesNum, BStatesNum, "N", groundState + currIndex, AStatesNum, BStatesNum, "D", rhoA);
		getDensityMatrixEigenVectors(rhoA, AStatesNum, &eigenvalues[i]);
		eigenvectors[i] = rhoA;
		subBlockStatesNum[i] = AStatesNum;
		currIndex += AStatesNum * BStatesNum;
		fullBasisSize += AStatesNum;
	}
	freeFcomplexArr(groundState);
	
	// We take the D states with the highest eigenvalues for rhoA.
	int newBasisSize = fullBasisSize < D ? fullBasisSize : D;
	int currEigenvector[subBlocksNum];
	for (int i = 0; i < subBlocksNum; i++) currEigenvector[i] = subBlockStatesNum[i] - 1;
	for (int i = 0; i < newBasisSize; i++) {
		float largestEigenvalue = -1;
		int largestEigenvalueBlock = 0;
		for (int j = 0; j <  subBlocksNum; j++) {
			if (currEigenvector[j] >= 0 && eigenvalues[j][currEigenvector[j]] >= largestEigenvalue) {
				largestEigenvalue = eigenvalues[j][currEigenvector[j]];
				largestEigenvalueBlock = j;
			}
		}
		currEigenvector[largestEigenvalueBlock]--;
	}
	
	// After deciding which are the RDM eigenvectors we will keep, update the blocks and return.
	MatrixBlock* result = mallocMatrixBlockArr(1);
	*resultSize = 0;
	*resultBasisSize = newBasisSize;
	for (int i = 0; i < subBlocksNum; i++) {
		int ABlockIndex = fullLatticeDoc[groundStateBlockIndex].statesDocs[i].ABlockIndex;
		if (currEigenvector[i] < subBlockStatesNum[i] - 1) { // We are using more than 0 vectors from this subblock.
			int basisSize = subBlockStatesNum[i] - 1 - currEigenvector[i];
			fcomplex* basis = eigenvectors[i] + (A[ABlockIndex].basisSize * (currEigenvector[i] + 1));
			fcomplex* H = updateOperatorForNewBasis(A[ABlockIndex].H, A[ABlockIndex].basisSize, A[ABlockIndex].basisSize, 
				basis, basisSize, basis, basisSize);
			fcomplex* sZLastSite = updateOperatorForNewBasis(A[ABlockIndex].sZLastSite, A[ABlockIndex].basisSize, A[ABlockIndex].basisSize,
				basis, basisSize, basis, basisSize);
			result[*resultSize] = (MatrixBlock) {
				.elementsNum = A[ABlockIndex].elementsNum,
				.basisSize = basisSize,
 				.sZLastSite = sZLastSite,
 				.sPlusLastSite = NULL,
 				.nextBlockBasisSize = 0,
				.H = H,
				.sZ = A[ABlockIndex].sZ
			};
			for (int j = 0; j < subBlocksNum; j++) {
				int nextBlockIndex = fullLatticeDoc[groundStateBlockIndex].statesDocs[j].ABlockIndex;
				if (A[ABlockIndex].sZ == A[nextBlockIndex].sZ - 1) {
					int nextBasisSize = subBlockStatesNum[j] - 1 - currEigenvector[j];
					fcomplex* nextBasis = eigenvectors[j] + (A[nextBlockIndex].basisSize * (currEigenvector[j] + 1));
					fcomplex* sPlusLastSite = updateOperatorForNewBasis(A[ABlockIndex].sPlusLastSite, 
						A[ABlockIndex].nextBlockBasisSize, A[ABlockIndex].basisSize,
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

	for (int i = 0; i < fullLaticeBlocksNum; i++) freeMatrixBlock(fullLattice + i);
	for (int i = 0; i < fullLaticeBlocksNum; i++) freeBlockStatesDoc(fullLatticeDoc + i);
	for (int i = 0; i < subBlocksNum; i++) freeFcomplexArr(eigenvectors[i]); 
	for (int i = 0; i < subBlocksNum; i++) freeFlaotArr(eigenvalues[i]);
	freeBlockStatesDocArr(fullLatticeDoc);
	freeMatrixBlockArr(fullLattice);

	return result;
}

void addSiteToBlock(MatrixBlock* restrict A, int ABlocksNum, 
		MatrixBlock** result, int* resultSize, BlockStatesDoc** doc) {
	combineStates(NEW_STATE_MATRIX_BLOCK, SITE_STATES_NUM, A, ABlocksNum, result, resultSize, doc, "Expansion");
	for (int i = 0; i < *resultSize; i++)  {
		//TODO to separate func.
		MatrixBlock* thisBlock = &(*result)[i];
		for (int j = 0; j < *resultSize; j++) {
			MatrixBlock* nextBlock = &(*result)[j];
			if (thisBlock->sZ == nextBlock->sZ - 1) {
				fcomplex* sPlusLastSite;
				int sPlusLastSiteRows, sPlusLastSiteCols;
				getSPlusLastSite(thisBlock, &(*doc)[i], nextBlock, &(*doc)[j], 
					NEW_STATE_MATRIX_BLOCK, &sPlusLastSite, &sPlusLastSiteRows, &sPlusLastSiteCols);
				thisBlock->sPlusLastSite = sPlusLastSite;
				thisBlock->nextBlockBasisSize = sPlusLastSiteRows;
				break;
			}
			thisBlock->nextBlockBasisSize = 0;
		}
	}

}

MatrixBlock* infiniteDmrgStep(MatrixBlock* A, int ABlocksNum, int ABasisSize, int* resultSize, int* resultBasisSize) {
	// Add a single site to A
	MatrixBlock* expandedBlockA;
	int expandedBlocksNum;
	BlockStatesDoc* expandedBlockDoc;
	addSiteToBlock(A, ABlocksNum, &expandedBlockA, &expandedBlocksNum, &expandedBlockDoc);

	// If our basis size is small enough to keep in its whole, do it.
	int currBasisSize = ABasisSize * SITE_STATES_NUM;
	if (currBasisSize <= D) {
		for (int i = 0; i < expandedBlocksNum; i++) freeBlockStatesDoc(expandedBlockDoc + i);
		freeBlockStatesDocArr(expandedBlockDoc);	
		*resultSize = expandedBlocksNum;
		*resultBasisSize = currBasisSize;
		return expandedBlockA;
	}

	// Combine our new expanded lattice block with an identical one.
	MatrixBlock expandedBlockB[expandedBlocksNum];
	for (int i = 0; i < expandedBlocksNum; i++) copyMatrixBlock(expandedBlockA + i, expandedBlockB + i);

	MatrixBlock* result = dmrgStep(expandedBlockA, expandedBlocksNum, expandedBlockB, expandedBlocksNum, resultSize, resultBasisSize);
		
	for (int i = 0; i < expandedBlocksNum; i++) freeMatrixBlock(expandedBlockA + i);
	for (int i = 0; i < expandedBlocksNum; i++) freeMatrixBlock(expandedBlockB + i);
	for (int i = 0; i < expandedBlocksNum; i++) freeBlockStatesDoc(expandedBlockDoc + i);
	freeBlockStatesDocArr(expandedBlockDoc);
	freeMatrixBlockArr(expandedBlockA);
	
	return result;
 
}

MatrixBlock* finiteDmrgStep(MatrixBlock* A, int ABlocksNum, MatrixBlock* B, int BBlocksNum, int* resultSize, int* resultBasisSize) {
	MatrixBlock* expandedBlockA;
	int expandedBlocksNum;
	BlockStatesDoc* expandedBlockDoc;
	addSiteToBlock(A, ABlocksNum, &expandedBlockA, &expandedBlocksNum, &expandedBlockDoc);
	MatrixBlock* result = dmrgStep(expandedBlockA, expandedBlocksNum, B, BBlocksNum, resultSize, resultBasisSize);
	for (int i = 0; i < expandedBlocksNum; i++) freeMatrixBlock(expandedBlockA + i);
	for (int i = 0; i < expandedBlocksNum; i++) freeBlockStatesDoc(expandedBlockDoc + i);
	freeBlockStatesDocArr(expandedBlockDoc);
	freeMatrixBlockArr(expandedBlockA);	
	return result;
}

int choose(int n, int k) {
	float result = 1;
	for (int i = 1; i <= n - k; i++) result /= i;
	for (int i = k + 1; i <= n; i++) result *= i;
	return (int) result;
}

void exact() {
	// int size = LATTICE_SIZE;
	// printf("exact solution:\n");
	// int blocksDiv[BASIS_SIZE] = {0};
	// int currPosition[size + 1] = {0};
	// int c = 0;
	// for (int i = 0; i <= size; i++) {
	// 	currPosition[i] = c;
	// 	c += choose(size, i);
	// }
	// for (int i = 0; i < BASIS_SIZE; i++) {
	// 	int counter = 0;
	// 	for (int state = i; state > 0; state >>= 1) counter += state & 1;
	// 	blocksDiv[currPosition[counter]] = i;
	// 	currPosition[counter]++;
	// }
	// fcomplex fullH[H_SIZE] = {0};
	// int curr = 0;
	// float minE = 0;
	// float minSZ = 0;
	// for (int currBlock = 0; currBlock <= size; currBlock++) {
	// 	int basisSize = choose(size, currBlock);
	// 	printf("currBlock = %d, basisSize = %d\n", currBlock, basisSize);
	// 	fcomplex* H = H + curr;
	// 	curr +=  
	// 	for (int i = 0; i < basisSize; i++) {
	// 		for (int j = 0; j < basisSize; j++) {
	// 			int index = i*basisSize + j;
	// 			H[index] = ZERO;
	// 			int iState = blocksDiv[currBlock][i];
	// 			int jState = blocksDiv[currBlock][j];
	// 			if (iState == jState) {
	// 				int state = iState;
	// 				for (int k = 0; k < size; k++) {
	// 					float sZI = (state & 1) - 0.5;
	// 					fcomplex c = {.re = sZI, .im = 0};
	// 					H[index] = add(H[index], mult(M, c));
	// 					float sZI1 = k < (size - 1) ? ((state & 2) / 2) - 0.5 : 0;
	// 					c.re *= sZI1;
	// 					H[index] = add(H[index], mult(JZ, c));
	// 					state >>= 1;
	// 				}
	// 			} else {
	// 				int xor = iState ^ jState; // We want the difference between i and j to be neighbours.
	// 				int xori = (xor & iState); // Each state should have one up and one down. 
	// 				int xorj = (xor & jState);
	// 				if (xori !=0 && xorj != 0) {
	// 					while (xor > 0) {
	// 						if (xor & 1 > 0) {
	// 							xor >>= 1;
	// 							if (xor & 1 > 0) {
	// 								xor >>= 1;
	// 								if (xor == 0) {
	// 									H[index] = add(H[index], JPM);
	// 								}
	// 								else xor = 0;
	// 							}
	// 							else xor = 0;
	// 						}
	// 						xor >>= 1;
	// 					}
	// 				}
	// 			}
	// 		}
	// 	}
	// 	// printMatrix("H exact", H, basisSize, basisSize);
	// 	float eigenvalues[basisSize];
	// 	fcomplex eigenVectors[(int) pow(basisSize, 2)];
	// 	int lwork = -1;
	//     fcomplex wkopt;
	//     fcomplex* work;
	//     float rwork[3*basisSize - 2];
	//     int info;
	//     cheev_( "Vectors", "Lower", &basisSize, H, &basisSize, eigenvalues, &wkopt, &lwork, rwork, &info);
	//     lwork = (int)wkopt.re;
	//     work = mallocFcomplexArr(lwork);
	//     /* Solve eigenproblem */
	//     cheev_( "Vectors", "Lower", &basisSize, H, &basisSize, eigenvalues, work, &lwork, rwork, &info);
	//     /* Check for convergence */
	//     if( info > 0 ) {
	//        printf( "exact: cheev failed.\n" );
	//        exit(1);
	//     }
	//     if (eigenvalues[0] < minE) {
	//     	minE = eigenvalues[0];
	//     	minSZ = (2* currBlock - size) * 0.5;
	//     }
	//     freeFcomplexArr(work);
	// 	freeFcomplexArr(H);

	// }
	// printf("E0 = %f, sZ = %f\n", minE, minSZ);
	// for (int i = 0; i <= size; i++) free(blocksDiv[i]);
 }
