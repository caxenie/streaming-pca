package org.streamingml.experiments;

import org.ejml.data.DMatrixRMaj;
import org.ejml.ops.MatrixIO;
import org.ejml.simple.SimpleMatrix;

import java.time.Duration;
import java.time.Instant;

public class Main {

    public static void main(String[] args) throws Exception {

        ClassLoader classLoader = Main.class.getClassLoader();

//         In the testing dataset I've considered a datastream with the property that
//         the eigenvalues of the input X are close to 1, 2, ..., d and the corresponding
//         eigenvectors are close to the canonical basis of R^d, where d is the number of
//         principal components to extract

        // global parameters of the input datastream (apriori known)
        int sampleSize = 10000;
        // we trained a batch PCA on the first samples and let iteratively learn on the next
        int initialSampleSize = 5000;

        String inputDatasetFile = args[0];      // "input_data.csv";
        // precomputed from a standard PCA to avoid large convergence times
        // we have precomputed standard PCA on first half of dataset, initialSampleSize values
        // and then we iteratively calculate up to the end
        String initEigValsFile = args[1];       // "input_data_init_eigvalues.csv";
        String initEigVectFile = args[2];       // "input_data_init_eigvectors.csv";
        String initPCACenterFile = args[3];     // "input_data_pca_center.csv";

        // Execution time profiling
        Instant start;
        Instant finish;

        // load datastructs for algorithms
        try {
            DMatrixRMaj inputDataRead = MatrixIO
                    .loadCSV(classLoader.getResource(inputDatasetFile)
                            .getFile(), true);
            SimpleMatrix X = SimpleMatrix.wrap(inputDataRead);
            DMatrixRMaj initEigValRead = MatrixIO
                    .loadCSV(classLoader.getResource(initEigValsFile)
                            .getFile(), true);
            SimpleMatrix initEigVals = SimpleMatrix.wrap(initEigValRead);
            DMatrixRMaj initEigVectRead = MatrixIO
                    .loadCSV(classLoader.getResource(initEigVectFile)
                            .getFile(), true);
            SimpleMatrix initEigVecs = SimpleMatrix.wrap(initEigVectRead);
            DMatrixRMaj initPCACenterRead = MatrixIO
                    .loadCSV(classLoader.getResource(initPCACenterFile)
                            .getFile(), true);
            SimpleMatrix initPCACenter = SimpleMatrix.wrap(initPCACenterRead).transpose();

            // run algorithms
            StreamPCAModels pcaModelccPCA = new StreamPCAModels(initEigVals, initEigVecs, initPCACenter, initialSampleSize);

            // Time execution
            start = Instant.now();
            for (int dId = initialSampleSize; dId < sampleSize; dId++) {
                // update PCA
                pcaModelccPCA.setModelState(pcaModelccPCA.CovarianceFreeIncrementalPCA(pcaModelccPCA.getModelState(),
                        X.extractVector(true, dId)));
            }
            finish = Instant.now();
            long timeElapsedCCPCA = Duration.between(start, finish).toMillis();
            System.out.println("Execution time for Covariance Free algorithm for PCA on " +
                    (sampleSize - initialSampleSize) +  " values is " + timeElapsedCCPCA + " ms");
            System.out.println("Covariance Free algorithm for PCA");
            System.out.println("Eigenvalues");
            pcaModelccPCA.getModelState().getLambda().print();
            System.out.println("Eigenvectors");
            pcaModelccPCA.getModelState().getQ().print();

            // Generalized Hebbian Algorithm for PCA for samples after initSampleSize (second half of dataset)
            StreamPCAModels pcaModelghaPCA = new StreamPCAModels(initEigVals, initEigVecs, initPCACenter, initialSampleSize);

            // Time execution
            start = Instant.now();
            for (int dId = initialSampleSize; dId < sampleSize; dId++) {
                // update PCA
                pcaModelghaPCA.setModelState(pcaModelghaPCA.GeneralizedHebbianPCA(pcaModelghaPCA.getModelState(),
                        X.extractVector(true, dId)));
            }
            finish = Instant.now();
            long timeElapsedGHAPCA = Duration.between(start, finish).toMillis();
            System.out.println("Execution time for Generalized Hebbian Algorithm for PCA on " +
                    (sampleSize - initialSampleSize) +  " values is " + timeElapsedGHAPCA + " ms");
            System.out.println("Generalized Hebbian Algorithm for PCA ");
            System.out.println("Eigenvalues");
            pcaModelghaPCA.getModelState().getLambda().print();
            System.out.println("Eigenvectors");
            pcaModelghaPCA.getModelState().getQ().print();

            // Stochastic Gradient Ascent PCA  - Exact, QR decomposition based version
            // for samples after initSampleSize (second half of dataset)
            StreamPCAModels pcaModelsgaExPca = new StreamPCAModels(initEigVals, initEigVecs, initPCACenter, initialSampleSize);

            // Time execution
            start = Instant.now();
            for (int dId = initialSampleSize; dId < sampleSize; dId++) {
                // update PCA
                pcaModelsgaExPca.setModelState(pcaModelsgaExPca.StochasticGradientAscentExactPCA(pcaModelsgaExPca.getModelState(),
                        X.extractVector(true, dId)));
            }
            finish = Instant.now();
            long timeElapsedSGAEXPCA = Duration.between(start, finish).toMillis();
            System.out.println("Execution time for Generalized Hebbian Algorithm for PCA on " +
                    (sampleSize - initialSampleSize) +  " values is " + timeElapsedSGAEXPCA + " ms");
            System.out.println("Exact Stochastic Gradient Ascent PCA");
            System.out.println("Eigenvalues");
            pcaModelsgaExPca.getModelState().getLambda().print();
            System.out.println("Eigenvectors");
            pcaModelsgaExPca.getModelState().getQ().print();

            // Stochastic Gradient Ascent PCA - Fast Neural Network version
            // for samples after initSampleSize (second half of dataset)
            StreamPCAModels pcaModalsgaNnPca = new StreamPCAModels(initEigVals, initEigVecs, initPCACenter, initialSampleSize);

            // Time execution
            start = Instant.now();
            for (int dId = initialSampleSize; dId < sampleSize; dId++) {
                // update PCA
                pcaModalsgaNnPca.setModelState(pcaModalsgaNnPca.StochasticGradientAscentNeuralNetPCA(pcaModalsgaNnPca.getModelState(),
                        X.extractVector(true, dId)));
            }
            finish = Instant.now();
            long timeElapsedSGANETPCA = Duration.between(start, finish).toMillis();
            System.out.println("Execution time for Generalized Hebbian Algorithm for PCA on " +
                    (sampleSize - initialSampleSize) +  " values is " + timeElapsedSGANETPCA + " ms");
            System.out.println("Neural Net Stochastic Gradient Ascent PCA");
            System.out.println("Eigenvalues");
            pcaModalsgaNnPca.getModelState().getLambda().print();
            System.out.println("Eigenvectors");
            pcaModalsgaNnPca.getModelState().getQ().print();

            System.out.println("Execution Time Evaluation (stream) \n" + "CC PCA \t GHA PCA \t SGA EX PCA \t SGA ANN PCA \n" +
                    timeElapsedCCPCA + " ms\t " + timeElapsedGHAPCA + " ms\t \t " +
                    timeElapsedSGAEXPCA + " ms \t\t " + timeElapsedSGANETPCA + " ms");

            System.out.println("Execution Time Evaluation (single event) \n" + "CC PCA \t GHA PCA \t SGA EX PCA \t SGA ANN PCA \n" +
                    (double)timeElapsedCCPCA/(sampleSize - initialSampleSize) + " ms\t " +
                    (double)timeElapsedGHAPCA/(sampleSize - initialSampleSize) + " ms\t \t " +
                    (double)timeElapsedSGAEXPCA/(sampleSize - initialSampleSize) + " ms \t\t " +
                    (double)timeElapsedSGANETPCA/(sampleSize - initialSampleSize) + " ms");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}