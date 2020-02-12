package org.streamingml.experiments;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.interfaces.decomposition.QRDecomposition;
import org.ejml.simple.SimpleMatrix;

class StreamPCAModelsState{

    private SimpleMatrix lambda;
    private SimpleMatrix Q;
    private SimpleMatrix xbar;
    private int n;

    void setLambda(SimpleMatrix lambda) {
        this.lambda = lambda;
    }

    void setQ(SimpleMatrix q) {
        Q = q;
    }

    void setN(int n) {
        this.n = n;
    }

    void setXbar(SimpleMatrix xbar) {
        this.xbar = xbar;
    }

    SimpleMatrix getLambda(){
        return lambda;
    }

    SimpleMatrix getQ(){
        return Q;
    }

    SimpleMatrix getXbar(){
        return xbar;
    }

    int getN(){
        return n;
    }
}

 class StreamPCAModels {

    private StreamPCAModelsState modelState = new StreamPCAModelsState();

    StreamPCAModelsState getModelState() {
        return modelState;
    }

    void setModelState(StreamPCAModelsState modelState) {
        this.modelState = modelState;
    }

    StreamPCAModels(SimpleMatrix initEigVal, SimpleMatrix initEigVecs, SimpleMatrix pcaCenter, int iter){
        modelState.setLambda(initEigVal);
        modelState.setQ(initEigVecs);
        modelState.setXbar(pcaCenter);
        modelState.setN(iter);
    }

//     Covariance Free algorithm for PCA
//     Weng et al. (2003). Candid Covariance-free Incremental Principal Component Analysis.
//     IEEE Trans. Pattern Analysis and Machine Intelligence.

    StreamPCAModelsState CovarianceFreeIncrementalPCA(StreamPCAModelsState state, SimpleMatrix x){

        //        The ’amnesic’ parameter l determines the weight of past observations in the PCA update. If l=0, all
        //        observations have equal weight, which is appropriate for stationary processes. Otherwise, typical
        //        values of l range between 2 and 4. As l increases, more weight is placed on new observations and
        //        less on older ones. For meaningful results, the condition 0<=l<n should hold.
        //        The algorithm iteratively updates the PCs while deflating x. If at some point the Euclidean
        //        norm of x becomes less than tol, the algorithm stops to prevent numerical overflow.

        // recover state
        SimpleMatrix lambda = state.getLambda();
        SimpleMatrix Q = state.getQ();
        int n = state.getN();
        SimpleMatrix xbar = state.getXbar();

        // update the average
        state.setXbar(this.updateIncrementalDataMean(xbar, x, n));
        xbar = state.getXbar();

        // for the update remove the average
        x = x.minus(xbar).transpose();

        // init
        int q = lambda.numRows();
        double l = 2;
        double tol = 1e-8;
        int i, d = x.getNumElements(), k = lambda.getNumElements();
        if (q != k) {
            Q.reshape(d,q);
            lambda.reshape(q, 1);
        }
        SimpleMatrix v;
        double f = (1.0 + l)/(1.0 + n);
        double nrm;

        for (i=0; i<q; i++) {

            nrm = x.normF();
            if (nrm < tol) {
                SimpleMatrix lambdaUpdate = lambda.extractMatrix(q - i, lambda.numRows(), 0, 1);
                lambda.setColumn(0, q - i, (lambdaUpdate.scale((1.0 - f))).getDDRM().data);
                break;
            }

            if (i == n) {
                lambda.set(i, 0, nrm);
                double[] xNormedValues = (x.scale(1.0 / nrm)).getDDRM().data;
                Q.setColumn(i, 0, xNormedValues);
                break;
            }

            v = Q.extractVector(false, i).scale(((1.0 - f) * lambda.get(i,0)))
                    .plus(x.scale(f * (Q.extractVector(false, i).dot(x))));

            nrm = v.normF();
            if (nrm < tol) {
                lambda.set(i, 0, 0.0);
                break;
            }
            lambda.set(i, 0, nrm);
            double[] vNormedValues = (v.scale(1.0 / nrm)).getDDRM().data;
            Q.setColumn(i, 0, vNormedValues);
            x = x.minus(Q.extractVector(false, i)
                    .scale(Q.extractVector(false, i)
                            .dot(x)));
        }
        // populate new state
        state.setLambda(lambda);
        state.setQ(Q);
        return state;
    }

//     Generalized Hebbian Algorithm for PCA
//     Sanger (1989). Optimal unsupervised learning in a single-layer linear feedforward neural network.
//     Neural Networks Journal

    StreamPCAModelsState GeneralizedHebbianPCA(StreamPCAModelsState state, SimpleMatrix x){

        //        The vector gamma determines the weight placed on the new data in updating each eigenvector (the
        //        first coefficient of gamma corresponds to the first eigenvector, etc). It can be specified as a single
        //        positive number or as a vector of length ncol(U). Larger values of gamma place more weight on
        //        x and less on U. A common choice for (the components of) gamma is of the form c/n, with n the
        //        sample size and c a suitable positive constant.

        // recover state
        SimpleMatrix lambda = state.getLambda();
        SimpleMatrix Q = state.getQ();
        int ind = state.getN();
        SimpleMatrix gamma = new SimpleMatrix(Q.numCols(), 1);
        for (int id = 0; id < Q.numCols(); id++) {
            gamma.set(id, 1.0);
        }
        gamma = gamma.scale(1.0 / (ind * ind));
        SimpleMatrix xbar = state.getXbar();

        // update the average
        state.setXbar(this.updateIncrementalDataMean(xbar, x, ind));
        xbar = state.getXbar();

        // for the update remove the average
        x = x.minus(xbar).transpose();

        // update the predictor
        SimpleMatrix y = Q.mult(x);

        // prepare new state
        int m = Q.numRows(), n = Q.numCols();
        SimpleMatrix gamy = gamma.elementMult(y); // Schur product
        SimpleMatrix b = Q.extractVector(false, 0).scale(y.get(0));
        SimpleMatrix A = new SimpleMatrix(m,n);
        A.setColumn(0, 0,
                (Q.extractVector(false, 0)
                        .minus(b.scale(gamy.get(0))))
                        .getDDRM()
                        .data);
        for (int i=1; i<n; i++) {
            b = b.plus(Q.extractVector(false, i).scale(y.get(i)));
            A.setColumn(i, 0,
                    (Q.extractVector(false, i)
                            .minus(b.scale(gamy.get(i))))
                            .getDDRM()
                            .data);
        }
        A = A.plus(x.mult(gamy.transpose()));
        SimpleMatrix decay = ((gamma.minus(1.0)).scale(-1.0)).elementMult(lambda);
        SimpleMatrix increment = gamma.elementMult(y).elementMult(y);
        lambda = increment.plus(decay);

        // wrap return values
        state.setLambda(lambda);
        state.setQ(A);
        return state;
    }
//
//     Stochastic Gradient Ascent PCA - Exact, QR decomposition based version
//     Oja (1992). Principal components, Minor components, and linear neural networks. Neural Networks.

    StreamPCAModelsState StochasticGradientAscentExactPCA(StreamPCAModelsState state, SimpleMatrix x){

        //        The gain vector gamma determines the weight placed on the new data in updating each principal
        //        component. The first coefficient of gamma corresponds to the first principal component, etc.. It can
        //        be specified as a single positive number (which is recycled by the function) or as a vector of length
        //        ncol(U). For larger values of gamma, more weight is placed on x and less on U. A common choice
        //        for (the components of) gamma is of the form c/n, with n the sample size and c a suitable positive
        //        constant. The Stochastic Gradient Ascent PCA can be implemented exactly or through a neural network.
        //        The latter is less accurate but faster.

        // recover state
        SimpleMatrix lambda = state.getLambda();
        SimpleMatrix Q = state.getQ();

        int ind = state.getN();
        SimpleMatrix gamma = new SimpleMatrix(Q.numCols(), 1);
        for (int id = 0; id < Q.numCols(); id++) {
            gamma.set(id, 1.0);
        }
        gamma = gamma.scale(1.0 / (ind * ind));
        SimpleMatrix xbar = state.getXbar();

        // update the average
        // update the average
        state.setXbar(this.updateIncrementalDataMean(xbar, x, ind));
        xbar = state.getXbar();

        // for the update remove the average
        x = x.minus(xbar).transpose();

        // update the predictor
        SimpleMatrix y = Q.mult(x);

        SimpleMatrix W;
        SimpleMatrix Qupd = Q;
        SimpleMatrix evidence = (x.mult(y.transpose()));
        SimpleMatrix gammaDiag = gamma.diag();
        SimpleMatrix increment = evidence.mult(gammaDiag);
        Qupd = Qupd.plus(increment);

        QRDecomposition<DMatrixRMaj> qrDecomp = DecompositionFactory_DDRM.qr(Qupd.numRows(), Qupd.numCols());
        qrDecomp.decompose(Qupd.getMatrix());
        W = SimpleMatrix.wrap(qrDecomp.getQ(null, true));

        SimpleMatrix decay = ((gamma.minus(1.0)).scale(-1.0)).elementMult(lambda);
        SimpleMatrix incrementLambda = gamma.elementMult(y).elementMult(y);
        lambda = incrementLambda.plus(decay);

        // wrap return values
        state.setLambda(lambda);
        state.setQ(W);
        return state;
    }


    // Stochastic Gradient Ascent PCA - Fast Neural Network version
    // Oja (1992). Principal components, Minor components, and linear neural networks. Neural Networks.

    StreamPCAModelsState StochasticGradientAscentNeuralNetPCA(StreamPCAModelsState state, SimpleMatrix x){

        // recover state
        SimpleMatrix lambda = state.getLambda();
        SimpleMatrix Q = state.getQ();

        int ind = state.getN();
        SimpleMatrix gamma = new SimpleMatrix(Q.numCols(), 1);
        for (int id = 0; id < Q.numCols(); id++) {
            gamma.set(id, 1.0);
        }
        gamma = gamma.scale(1.0 / (ind * ind));
        SimpleMatrix xbar = state.getXbar();

        // update the average
        state.setXbar(this.updateIncrementalDataMean(xbar, x, ind));
        xbar = state.getXbar();

        // for the update remove the average
        x = x.minus(xbar).transpose();

        // update the predictor
        SimpleMatrix y = Q.mult(x);

        int m = Q.numRows(), n = Q.numCols();
        SimpleMatrix gamy = gamma.elementMult(y); // Schur product
        SimpleMatrix b = Q.extractVector(false, 0).scale(y.get(0));
        SimpleMatrix A = new SimpleMatrix(m,n);
        A.setColumn(0, 0,
                (Q.extractVector(false, 0)
                        .minus(b.scale(gamy.get(0))))
                        .getDDRM()
                        .data);
        for (int i=1; i<n; i++) {

            b = b.plus((Q.extractVector(false, i - 1).scale(y.get(i - 1)))
                    .plus(Q.extractVector(false, i).scale(y.get(i))));

            A.setColumn(i, 0,
                    (Q.extractVector(false, i)
                            .minus(b.scale(gamy.get(i))))
                            .getDDRM()
                            .data);
        }
        A = A.plus(x.mult(gamy.transpose()));
        SimpleMatrix decay = ((gamma.minus(1.0)).scale(-1.0)).elementMult(lambda);
        SimpleMatrix increment = gamma.elementMult(y).elementMult(y);
        lambda = increment.plus(decay);

        // wrap return values
        state.setLambda(lambda);
        state.setQ(A);
        return state;
    }

    // Recursive update of the sample mean vector used in all PCA algorithms.
    private SimpleMatrix updateIncrementalDataMean(SimpleMatrix xbar, SimpleMatrix x, int n) {
        //    The forgetting factor f determines the balance between past and present observations in the PCA
        //    update: the closer it is to 1 (resp. to 0), the more weight is placed on current and past observations.
        //    For a given argument n, the default value of f is 1/(n + 1).
        double f = 1.0 / (n + 1.0);
        return (xbar.scale(1.0 - f)).plus(x.scale(f));
    }
}
