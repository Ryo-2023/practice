import numpy as np
import matplotlib.pyplot as plt

class MLPCA: 
    def __init__(self, m):
        self.m = m
        self.d = None
        self.W = None
        self.mu = None
        self.variance = None

    def _fit_analytical(self, X):
        self.d = len(X[0])
        self.mu = np.mean(X, axis=0)
        S = np.cov(X, rowvar=False, ddof=0)
        eigenvals, eigenvecs = eigh(S)
        ind = np.argsort(eigenvals)[::-1][0:self.m] # sort the eigenvalues in decreasing order, and choose the largest m
        eigenvals_largest = eigenvals[ind]
        eigenvecs_largest = eigenvecs[:, ind]
        if self.d != self.m:
            self.variance = 1/(self.d - self.m) * np.sum(np.sort(eigenvals)[::-1][self.m:])
        else:
            self.variance = 0.0
        self.W = eigenvecs_largest @ np.diag( np.sqrt(eigenvals_largest - self.variance) )

    def _init_params(self, X, W=None, variance=None):
        '''
        Method for initializing model parameterse based on the size and variance of the input data array. 

        Parameters
        ----------
        X : 2D numpy array
            2D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
        W : 2D numpy array, default None
            2D numpy array representing the initial value of parameter W
        variance : positive float, default is None
            positive float representing the initial value of parameter variance
        '''
        self.d = len(X[0])
        self.mu = np.mean(X, axis=0)
        self.W = np.eye(self.d)[:,:self.m] if (W is None) else W # probably there are better initializations 
        self.variance = 1.0 if (variance is None) else variance

    def _estep(self, X):
        '''
        Method for performing E-step, returning 
        $\mathbb{E}_{old}\left[ z_n \right]$ and $\sum_{n=0}^{N-1} \mathbb{E}_{old}\left[ z_n z_{n}^{T} \right]$

        Parameters
        ----------
        X : 2D numpy array
            2D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.

        Returns
        ----------
        Ez : 2D numpy array
            (len(X), self.m) array, where Ez[n, i] = $\mathbb{E}_{old}[z_n]_i$
        sumEzz : 2D numpy array
            (self.m, self.m) array, where sumEzz[i,j] = $\sum_{n=0}^{N-1} \mathbb{E}_{old}\left[z_n z_{n}^{T} \right]_{i,j}$
        '''
        Minv = np.linalg.inv( (self.W).T @ self.W + self.variance*np.eye(self.m) ) # we only use inverse of the matrix M
        dX = X - self.mu
        Ez = dX @ self.W @ (Minv.T)
        sumEzz = len(X) * self.variance * Minv + (Ez.T) @ Ez
        return Ez, sumEzz


    def _mstep(self, X, Ez, sumEzz):
        '''
        Method for performing M-step, and updating model parameters

        Parameters
        ----------
        X : 2D numpy array
            2D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
        Ez : 2D numpy array
            (len(X), self.m) array, where Ez[n, i] = $\mathbb{E}_{old}[z_n]_i$
        sumEzz : 2D numpy array
            (self.m, self.m) array, where sumEzz[i,j] = $\sum_{n=0}^{N-1} \mathbb{E}_{old}\left[z_n z_{n}^{T} \right]_{i,j}$
        '''
        dX = X - self.mu
        self.W = ((dX.T) @ Ez) @  np.linalg.inv(sumEzz)
        self.variance = ( np.linalg.norm(dX)**2 - 2*np.trace(Ez @ (self.W.T) @ (dX.T) ) + np.trace( sumEzz @ self.W.T @ self.W ) )  /(len(X)*self.d)


    def _fit_em(self, X, max_iter=100, tol=1e-4, disp_message=False, W0=None, variance0=None):
        '''
        Method for performing fitting by EM algorithm

        Parameters
        ----------
        X : 2D numpy array
            2D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
        max_iter : positive int
            The maximum number of iteration allowed
        tol : positive float
            Threshold for termination of iteration. 
            Concretely, the iteration is stopped when the change of W and variance is smaller than tol.
        disp_message : Boolean, default False
            If True, the number of iteration and convergence will displayed.
        W0 : 2D numpy array, default None
            2D numpy array representing the initial value of parameter W
        variance0 : positive float, default is None
            positive float representing the initial value of parameter variance
        '''
        self._init_params(X, W=W0, variance=variance0)

        for i in range(max_iter):
            Wold = self.W
            varianceold = self.variance

            Ez, sumEzz = self._estep(X)
            self._mstep(X, Ez, sumEzz)

            err = np.sqrt(np.linalg.norm(self.W - Wold)**2 + (self.variance - varianceold)**2)
            if err < tol:
                break

        if disp_message:
            print(f"n_iter : {i}")
            print(f"converged : {i < max_iter - 1}")

    def fit(self, X, method='analytical', **kwargs):
        '''
        Method for performing fitting

        Parameters
        ----------
        X : 2D numpy array
            2D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
        method : str, either 'analytical' or 'em'
            If method == 'analytical', _fit_analytical is invoked, while if method == 'em', _fit_em is invoked.
        '''
        if method == 'analytical':
            self._fit_analytical(X)
        elif method == 'em':
            self._fit_em(X, **kwargs)
        else:
            "Method should be either `analytical` or `em`."


    def transform(self, X, return_cov=False):
        '''
        Method for performing transformation, transforming observables to latent variables

        Parameters
        ----------
        X : 2D numpy array
            (len(X), self.d) numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
        return_cov : Boolean, default False
            If True, the covariance matrix of the predictive distribution is also returned

        Returns
        ----------
        Z : 2D numpy array
            (len(X), self.m) array representing the expectation values of the latent variables 
            corresponding to the input observable X
        cov : 2D numpy array, returned only if return_cov is True
            (self.m, self.m) array representing the covariance matrix for the predictive distribution 
            Concretely, it corresponds to $\sigma^2 M^{-1}$
        '''
        Minv = np.linalg.inv( (self.W).T @ self.W + self.variance*np.eye(self.m) ) # we only use inverse of the matrix M
        dX = X - self.mu
        Z = dX @ self.W @ (Minv.T)
        if return_cov:
            cov = self.variance * Minv
            return Z, cov
        else:
            return Z

    def inverse_transform(self, Z, return_cov=False):
        '''
        Method for performing inverse transformation, transforming latent variables to observables

        Parameters
        ----------
        Z : 2D numpy array
            (len(Z), self.m) numpy array representing latent variables
        return_cov : Boolean, default False
            If True, the covariance matrix of the predictive distribution is also returned

        Returns
        ----------
        X : 2D numpy array
            (len(Z), self.d) numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.

        cov : 2D numpy array, returned only if return_cov is True
            (self.d, self.d) array representing the covariance matrix for the predictive distribution 
            Concretely, it corresponds to $\sigma^2 I_d$
        '''
        X = Z @ (self.W.T) + self.mu
        if return_cov:
            cov = self.variance * np.eye(self.d)
            return X, cov
        else:
            return X

class BPCA(MLPCA): 
    def __init__(self, m):
        super().__init__(m)
        self.alpha = None

    def _init_params(self, X, W=None, variance=None, alpha=None):
        '''
        Method for initializing model parameterse based on the size and variance of the input data array. 

        Parameters
        ----------
        X : 2D numpy array
            2D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
        W : 2D numpy array, default None
            2D numpy array representing the initial value of parameter W
        variance : positive float, default is None
            positive float representing the initial value of parameter variance
        alpha : positive float, default is None
            positive float representing the initial value of alpha
        '''
        super()._init_params(X, W, variance)
        self.alpha =  np.ones(self.m) if (alpha is None) else alpha


    def _mstep(self, X, Ez, sumEzz):
        '''
        Method for performing M-step, and updating model parameters

        Parameters
        ----------
        X : 2D numpy array
            2D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
        Ez : 2D numpy array
            (len(X), self.m) array, where Ez[n, i] = $\mathbb{E}_{old}[z_n]_i$
        sumEzz : 2D numpy array
            (self.m, self.m) array, where sumEzz[i,j] = $\sum_{n=0}^{N-1} \mathbb{E}_{old}\left[z_n z_{n}^{T} \right]_{i,j}$
        '''
        dX = X - self.mu
        self.W = ((dX.T) @ Ez) @  np.linalg.inv(sumEzz + self.variance * np.diag(self.alpha))
        self.variance = ( np.linalg.norm(dX)**2 - 2*np.trace(Ez @ (self.W.T) @ (dX.T) ) + np.trace( sumEzz @ self.W.T @ self.W ) )  /(len(X)*self.d)


    def _calc_alpha(self):
        '''
        Method for updating the value of alpha for Bayesian approach
        '''
        wnorms = np.linalg.norm(self.W, axis=0)**2
        for i in range(self.m):
            self.alpha[i] = self.d/wnorms[i] if (wnorms[i] != 0) else np.inf
    
    def _calc_cost(self, X, Ez, sumEzz):
        '''
        Method for calculating the cost function (negative log likelihood)

        Parameters
        ----------
        X : 2D numpy array
            2D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
        Ez : 2D numpy array
            (len(X), self.m) array, where Ez[n, i] = $\mathbb{E}_{old}[z_n]_i$
        sumEzz : 2D numpy array
            (self.m, self.m) array, where sumEzz[i,j] = $\sum_{n=0}^{N-1} \mathbb{E}_{old}\left[z_n z_{n}^{T} \right]_{i,j}$
        '''
        N, d = X.shape
        dX = X - self.mu
        cost = N*d/2 * np.log(2*np.pi*self.variance)
        cost += 1/(2*self.variance) * (np.linalg.norm(dX)**2 - 2*np.trace(Ez @ (self.W.T) @ (dX.T)) + np.trace(sumEzz @ self.W.T @ self.W))
        return cost / N

    def fit(self, X, max_iter=100, tol=1e-4, disp_message=False, W0=None, variance0=None, alpha0=None):
        '''
        Method for performing fitting

        Parameters
        ----------
        X : 2D numpy array
            2D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
        max_iter : positive int
            The maximum number of iteration allowed
        tol : positive float
            Threshold for termination of iteration. 
            Concretely, the iteration is stopped when the change of W and variance is smaller than tol.
        disp_message : Boolean, default False
            If True, the number of iteration and convergence will displayed.
        W0 : 2D numpy array, default None
            2D numpy array representing the initial value of parameter W
        variance0 : positive float, default is None
            positive float representing the initial value of parameter variance
        alpha0 : positive float, default is None
            positive float representing the initial value of alpha   
        '''
        self._init_params(X, W=W0, variance=variance0, alpha=alpha0)

        costs = []  # to store cost function values

        for i in range(max_iter):
            # E-step
            Ez, sumEzz = self._estep(X)

            # M-step
            self._mstep(X, Ez, sumEzz)

            # calculate and store cost function value
            cost = self._calc_cost(X, Ez, sumEzz)
            costs.append(cost)

            # check for convergence
            if i > 0 and abs(costs[i] - costs[i-1]) < tol:
                if disp_message:
                    print(f'Converged in {i} iterations.')
                break

        # plot cost function values
        plt.plot(costs)
        plt.title('Cost function values')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()
        
# データの生成
X = np.random.rand(100, 10)

# BPCAクラスのインスタンス化とフィッティング
bpca = BPCA(2)
bpca.fit(X)

# データの次元削減
X_reduced = bpca.transform(X)

# プロット
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.title('BPCA dimensionality reduction')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()



