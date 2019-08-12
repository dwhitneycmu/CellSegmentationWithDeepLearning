# Generate code to reduce dimensionality of a matrix
from scipy.linalg import svd
import numpy as np

class PCA:
    def __init__(self,matrix):
        self.matrix = matrix
        if matrix.size>0:
            self.matrixDecomposition()
        
    def matrixDecomposition(self):
        # Use SVD to perform PCA on matrix
        [U,s,V]=svd(self.matrix,full_matrices=False)
        n = self.matrix.shape[0]-1
        S = np.diag(s)
        self.nPCs                = len(s)
        self.weights             = U
        self.principalComponents = np.dot(S,V)
        self.explainedVariance   = (S**2)/n
        self.explainedVariance   = self.explainedVariance / np.sum(self.explainedVariance)

    def dimensionality(self,n=0):
        # Return dimensionality of the matrix decomposition
        if(n<1):
            n=self.nPCs
        relativeVarianceExplained = self.explainedVariance[0:n]
        relativeVarianceExplained = relativeVarianceExplained / np.sum(relativeVarianceExplained)
        dim = 1./np.sum(relativeVarianceExplained**2)
        #print('nPCs: {}, Dim={:4.2f}'.format(n,dim))
        return dim
    
    def filterMatrix(self,n=0,maintainVariance=True):
        # Return a filtered version of the original matrix using a number of PCs
        # specified by the variable input "n". 
        if (n<1)|(n>self.nPCs):
            n=self.nPCs
        filteredMatrix = np.matmul(self.weights[:,range(n)],self.principalComponents[range(n),:])
        if(maintainVariance):
            filteredMatrix = filteredMatrix*(np.nanvar(self.matrix.flatten())/np.nanvar(filteredMatrix.flatten()))
        return np.matmul(self.weights[:,range(n)],self.principalComponents[range(n),:])
    
    def computeTargetPCs(self,targetRatio=0.5):
        # Compute the number of PCs necessary to reduce the matrix's 
        # dimensionality by the specified target ratio
        dimensionality = targetRatio*self.dimensionality()
        for n in range(self.nPCs):
            #print('Index: {}'.format(n))
            if(self.dimensionality(n+1)>=dimensionality):
                return(n+1)

if __name__ == "__main__":
    from scipy.stats import pearsonr as corr
    import matplotlib.pyplot as plt
    
    # Test dataset of random value
    m, n = 128, 10
    weightMatrix = np.random.randn(m, n)
    pcaObj = PCA(matrix=weightMatrix)
    
    # Check that we get back the same matrix we put in when we set the number
    # of selected components equal to a number of possible values (up to n)
    for i in [n,5,1]:
        filteredMatrix = pcaObj.filterMatrix(n=i)
        pcaObjFiltered = PCA(matrix=filteredMatrix)
        isMatrixSame   = np.allclose(weightMatrix,filteredMatrix)
        r = corr(weightMatrix.flatten(),filteredMatrix.flatten())
        
        print("\nIs PCA Filtered Matrix same (n={})?: {}".format(n,isMatrixSame))
        print("Original matrix: {}".format(  weightMatrix.flatten()))
        print("Filtered matrix: {}".format(filteredMatrix.flatten()))
        print("Pearson's Correlation: r = {:4.3f}".format(r[0]))
        print('Target Dimensionality at 50%: {}'.format(pcaObj.computeTargetPCs(0.5)))
        
        (fig,ax)=plt.subplots(nrows=2,ncols=1)
        ax[0].imshow(weightMatrix.T)
        ax[0].set_title('Original dimensionality: {:4.2f}'.format(pcaObj.dimensionality()))
        ax[1].imshow(filteredMatrix.T)
        ax[1].set_title('Filtered dimensionality: {:4.2f}'.format(pcaObjFiltered.dimensionality()))