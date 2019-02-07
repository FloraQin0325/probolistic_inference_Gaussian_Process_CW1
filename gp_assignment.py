import numpy as np
from scipy.optimize import minimize


# ##############################################################################
# LoadData takes the file location for the yacht_hydrodynamics.data and returns
# the data set partitioned into a training set and a test set.
# the X matrix, deal with the month and day strings.
# Do not change this function!
# ##############################################################################
def loadData(df):
    data = np.loadtxt(df)
    Xraw = data[:, :-1]
    # The regression task is to predict the residuary resistance per unit weight of displacement
    yraw = (data[:, -1])[:, None]
    X = (Xraw - Xraw.mean(axis=0)) / np.std(Xraw, axis=0)
    y = (yraw - yraw.mean(axis=0)) / np.std(yraw, axis=0)

    ind = range(X.shape[0])
    test_ind = ind[0::4]  # take every fourth observation for the test set
    train_ind = list(set(ind) - set(test_ind))
    X_test = X[test_ind]
    X_train = X[train_ind]
    y_test = y[test_ind]
    y_train = y[train_ind]

    return X_train, y_train, X_test, y_test


# ##############################################################################
# Returns a single sample from a multivariate Gaussian with mean and cov.
# ##############################################################################
def multivariateGaussianDraw(mean, cov):
    sample = np.zeros((mean.shape[0],))  # This is only a placeholder
    # Task 1:
    # TODO: Implement a draw from a multivariate Gaussian here
    sample = np.random.multivariate_normal(mean, cov)

    # Return drawn sample
    return sample


# ##############################################################################
# RadialBasisFunction for the kernel function
# k(x,x') = s2_f*exp(-norm(x,x')^2/(2l^2)). If s2_n is provided, then s2_n is
# added to the elements along the main diagonal, and the kernel function is for
# the distribution of y,y* not f, f*.
# ##############################################################################
class LinearPlusRBF():
    def __init__(self, params):
        self.ln_sigma_b = params[0]
        self.ln_sigma_v = params[1]
        self.ln_sigma_f = params[2]
        self.ln_length_scale = params[3]
        self.ln_sigma_n = params[4]

        self.sigma2_b = np.exp(2 * self.ln_sigma_b)
        self.sigma2_v = np.exp(2 * self.ln_sigma_v)
        self.sigma2_f = np.exp(2 * self.ln_sigma_f)
        self.sigma2_n = np.exp(2 * self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def setParams(self, params):
        self.ln_sigma_b = params[0]
        self.ln_sigma_v = params[1]
        self.ln_sigma_f = params[2]
        self.ln_length_scale = params[3]
        self.ln_sigma_n = params[4]

        self.sigma2_b = np.exp(2 * self.ln_sigma_b)
        self.sigma2_v = np.exp(2 * self.ln_sigma_v)
        self.sigma2_f = np.exp(2 * self.ln_sigma_f)
        self.sigma2_n = np.exp(2 * self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def getParams(self):
        return np.array([self.ln_sigma_b, self.ln_sigma_v, self.ln_sigma_f, self.ln_length_scale, self.ln_sigma_n])

    def getParamsExp(self):
        return np.array([self.sigma2_b, self.sigma2_v, self.sigma2_f, self.length_scale, self.sigma2_n])

    # ##########################################################################
    # covMatrix computes the covariance matrix for the provided matrix X using
    # the RBF. If two matrices are provided, for a training set and a test set,
    # then covMatrix computes the covariance matrix between all inputs in the
    # training and test set.
    # ##########################################################################
    def covMatrix(self, X, Xa=None):

        if Xa is not None:
            X_aug = np.zeros((X.shape[0] + Xa.shape[0], X.shape[1]))
            X_aug[:X.shape[0], :X.shape[1]] = X
            X_aug[X.shape[0]:, :X.shape[1]] = Xa
            X = X_aug

        n = X.shape[0]
        covMat1 = np.zeros((n, n))  # covMat1 means Linear
        covMat2 = np.zeros((n, n))  # covMat2 means RBF

        # Task 2:
        # TODO: Implement the covariance matrix here
        for i in range(n):
            for j in range(n):
                covMat2[i][j] = self.sigma2_f * np.exp(
                    ((-1) * (X[i] - X[j]).T.dot((X[i] - X[j]))) / (2 * self.length_scale ** 2))
                covMat1[i][j] = self.sigma2_b + self.sigma2_v * X[i].T.dot(X[j])

        covMat = covMat1 + covMat2
        print("DD", covMat.shape)
        # If additive Gaussian noise is provided, this adds the sigma2_n along
        # the main diagonal. So the covariance matrix will be for [y y*]. If
        # you want [y f*], simply subtract the noise from the lower right
        # quadrant.
        if self.sigma2_n is not None:
            covMat += self.sigma2_n * np.identity(n)

        # Return computed covariance matrix
        return covMat


class GaussianProcessRegression():
    def __init__(self, X, y, k):
        self.X = X
        self.n = X.shape[0]
        self.y = y
        self.k = k
        self.K = self.KMat(self.X)
        self.L = np.linalg.cholesky(self.K)

    # ##########################################################################
    # Recomputes the covariance matrix and the inverse covariance
    # matrix when new hyperparameters are provided.
    # ##########################################################################
    def KMat(self, X, params=None):
        if params is not None:
            self.k.setParams(params.reshape(params.size, 1))
        K = self.k.covMatrix(X)
        self.K = K
        self.L = np.linalg.cholesky(self.K)
        return K

    # ##########################################################################
    # Computes the posterior mean of the Gaussian process regression and the
    # covariance for a set of test points.
    # NOTE: This should return predictions using the 'clean' (not noisy) covariance
    # ##########################################################################

    def predict(self, Xa):
        mean_fa = np.zeros((Xa.shape[0], 1))
        cov_fa = np.zeros((Xa.shape[0], Xa.shape[0]))

        # Task 3:
        # TODO: compute the mean and covariance of the prediction
        covMatXXa = self.k.covMatrix(self.X, Xa)[self.n:, self.n:] - self.k.sigma2_n * np.identity(self.n)
        vMat_Xa = self.k.covMatrix(Xa) - self.k.sigma2_n * np.identity(self.n)
        L_inv = np.linalg.inv(self.L)
        K_inv = np.dot(L_inv.T, L_inv)
        mean_fa = covMatXXa.T.dot(K_inv).dot(self.y)
        cov_fa = vMat_Xa - covMatXXa.T.dot(K_inv).dot(covMatXXa)
        # Return the mean and covariance
        return mean_fa, cov_fa

    # ##########################################################################
    # Return negative log marginal likelihood of training set. Needs to be
    # negative since the optimiser only minimises.
    # ##########################################################################
    def logMarginalLikelihood(self, params=None):

        if params is not None:
             K = self.KMat(self.X, params)
            #self.k.setParams(params.reshape(params.size, 1))
        mll = 0

        # Task 4:
        # TODO: Calculate the log marginal likelihood ( mll ) of self.y
        #sign, logdet = np.linalg.slogdet(self.K)
        L_inv = np.linalg.inv(self.L)
        K_inv = np.dot(L_inv.T, L_inv)

        mll = 0.5 * (self.y.T).dot(K_inv).dot(self.y) + \
              0.5 * np.log(np.linalg.det(self.K)) + \
              0.5 * self.n * np.log(2 * np.pi)
        # Return mll
        return mll

    # ############################################################## ############
    # Computes the gradients of the negative log marginal likelihood wrt each
    # hyperparameter.
    # ##########################################################################
    def gradLogMarginalLikelihood(self, params=None):
        K = self.k.covMatrix(self.X)
        if params is not None:
            K = self.KMat(self.X, params)
        grad_ln_sigma_b = grad_ln_sigma_v = grad_ln_sigma_f = grad_ln_length_scale = grad_ln_sigma_n = 0
        # Task 5:
        # TODO: calculate the gradients of the negative log marginal likelihood
        # wrt. the hyperparameters

        L_inv = np.linalg.inv(self.L)
        K_inv = np.dot(L_inv.T, L_inv)

        alpha = K_inv.dot(self.y)
        tmp = (alpha.dot(alpha.T) - K_inv)
        XXT = self.X.dot(self.X.T)
        covMat22 = np.zeros((self.n, self.n))

        for i in range(self.n):
            for j in range(self.n):
                covMat22[i][j] = (self.X[i] - self.X[j]).T.dot((self.X[i] - self.X[j]))

        partRBF = self.K - self.k.sigma2_b - self.k.sigma2_v * XXT - self.k.sigma2_n * np.identity(self.n)

        grad_ln_sigma_f = -0.5 * np.trace(tmp.dot(2 * partRBF))

        grad_ln_length_scale = -0.5 * np.trace(tmp.dot(partRBF * covMat22 / (self.k.length_scale**2)))

        grad_ln_sigma_n = -0.5 * np.trace(tmp.dot(self.k.sigma2_n * 2 * np.identity(self.n)))

        grad_ln_sigma_b = -0.5 * np.trace(tmp.dot(self.k.sigma2_b * 2 * np.ones((self.n, self.n))))

        grad_ln_sigma_v = -0.5 * np.trace(tmp.dot(2 * self.k.sigma2_v * XXT))
        print(grad_ln_sigma_b, grad_ln_sigma_v, grad_ln_sigma_f, grad_ln_length_scale, grad_ln_sigma_n)
         # Combine gradients
        gradients = np.array([grad_ln_sigma_b, grad_ln_sigma_v, grad_ln_sigma_f, grad_ln_length_scale, grad_ln_sigma_n])

        # Return the gradients
        return gradients

    # ##########################################################################
    # Computes the mean squared error between two input vectors.
    # ##########################################################################
    def mse(self, ya, fbar):
        mse = 0
        # Task 7:
        # TODO: Implement the MSE between ya and fbar
        for i in range(ya.shape[0]):
            mse += (ya[i] - fbar[i]) ** 2
        mse = mse / ya.shape[0]
        # Return mse
        return mse

    # ##########################################################################
    # Computes the mean standardised log loss.
    # ##########################################################################
    def msll(self, ya, fbar, cov):
        msll = 0
        # Task 7:
        # TODO: Implement MSLL of the prediction fbar, cov given the target ya
        for i in range(ya.size):
            variance = cov[i][i] + self.k.sigma2_n
            msll += (np.log(2 * np.pi * variance)) + (ya[i] - fbar[i]) ** 2 / variance
        msll /= 2.0 * ya.size
        print(msll)
        # Return msll
        return msll

    # ##########################################################################
    # Minimises the negative log marginal likelihood on the training set to find
    # the optimal hyperparameters using BFGS.
    # ##########################################################################
    def optimize(self, params, disp=True):
        res = minimize(self.logMarginalLikelihood, params, method='BFGS', jac=self.gradLogMarginalLikelihood,
                       options={'disp': disp})
        return res.x


if __name__ == '__main__':
    np.random.seed(42)
    ##########################
    # You can put your tests here - marking
    # will be based on importing this code and calling
    # specific functions with custom input.
    ##########################
