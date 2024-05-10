import numpy as np
import matplotlib.pyplot as plt

class HMM:

    def __init__(self, K):
        self.K = K
        self.pi = None
        self.A = None
        self.phi = None


    def _init_params(self, pi=None, A=None, phi=None, seed_pi=None, seed_A=None, seed_phi=None):
        '''
        Initializes parameters pi, A, phi for EM algorithm
        '''
        self.pi =  pi if (pi is not None) else np.random.RandomState(seed=seed_pi).dirichlet(alpha=np.ones(self.K))
        self.A = A if (A is not None) else np.random.RandomState(seed=seed_A).dirichlet(alpha=np.ones(self.K), size=self.K)
        self.phi = phi if (phi is not None) else np.random.RandomState(seed=seed_phi).rand(self.K)  # emissoin probability dependent

    def _calc_pmatrix(self, X):
        '''
        Calculates a 2D array pmatrix (pmatrix[n, k] = $p (x_n | \phi_k)$) for E step. 

        Parameters
        ----------
        X : 2D numpy array
            2D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.

        Returns
        ----------
        pmatrix : 2D numpy array
            (len(X), self.K) numpy array
        '''
        pmatrix = (self.phi**X) * ((1-self.phi)**(1-X)) # emissoin probability dependent
        return pmatrix


    def _forward(self, pmatrix):
        '''
        Performs forward process and returns alpha and c (the normalization constant)

        Parameters
        ----------
        pmatrix : 2D numpy array
            (N, self.K) numpy array, where pmatrix[n, k] = $p (x_n | \phi_k)$

        Returns
        ----------
        alpha : 2D numpy array
            (N, self.K) numpy array

        c :  1D numpy array
            (N,) numpy array
        '''
        N = len(pmatrix)
        alpha = np.zeros((N, self.K))
        c = np.zeros(N)
        tmp = self.pi * pmatrix[0]
        c[0] = np.sum(tmp)
        alpha[0] = tmp / c[0]

        for n in range(1, N, 1):
            tmp = pmatrix[n] * ( (self.A).T @  alpha[n-1] )
            c[n] = np.sum(tmp)
            alpha[n] = tmp / c[n]
        return alpha, c

    def _backward(self, pmatrix, c):
        '''
        Performs backward process and returns beta

        Parameters
        ----------
        pmatrix : 2D numpy array
            (N, self.K) numpy array
        c :  1D numpy array
            (N,) numpy array

        Returns
        ----------
        beta : 2D numpy array
            (N, self.K) numpy array
        '''
        N = len(pmatrix)
        beta = np.zeros((N, self.K))
        beta[N - 1] = np.ones(self.K)
        for n in range(N-2, -1, -1):
            beta[n] = self.A @ ( beta[n+1] * pmatrix[n+1] ) / c[n+1]
        return beta

    def _estep(self, pmatrix, alpha, beta, c):
        '''
        Calculates and returns gamma and xi from the inputs including alpha, beta and c

        Parameters
        ----------
        pmatrix : 2D numpy array
            (N, self.K) numpy array
        alpha : 2D numpy array
            (N, self.K) numpy array
        beta : 2D numpy array
            (N, self.K) numpy array
        c :  1D numpy array
            (N,) numpy array

        Returns
        ----------
        gamma : 2D numpy array
            (N, self.K) array
        xi : 3D numpy array
            (N, self.K, self.K) array. Note that xi[0] is meaningless.
        '''

        gamma = alpha * beta
        xi = np.roll(alpha, shift=1, axis=0).reshape(N, self.K, 1) * np.einsum( "jk,nk->njk", self.A, pmatrix * beta) / np.reshape( c, (N, 1,1))
        return gamma, xi

    def _mstep(self, X, gamma, xi):
        '''
        Performs M step, i.e., updates parameters pi, A, phi, using the result of E step

        Parameters
        ----------
        X : 2D numpy array
            2D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
        gamma : 2D numpy array
            (N, self.K) array
        xi : 3D numpy array
            (N, self.K, self.K) array
        '''
        self.pi = gamma[0] / np.sum(gamma[0])
        xitmp = np.sum(xi[1:], axis=0)
        self.A = xitmp / np.reshape(np.sum(xitmp, axis=1) , (self.K, 1))
        self.phi = (gamma.T @ X[:,0])  / np.sum(gamma, axis=0)


    def fit(self, X, max_iter=1000, tol=1e-3, **kwargs):
        '''
        Performs fitting with EM algorithm

        Parameters
        ----------
        X : 2D numpy array
            2D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
        max_iter : positive int
            The maximum number of iteration allowed
        tol : positive float
            Threshold for termination of iteration. 
        '''
        self._init_params(**kwargs)
        log_likelihood = -np.inf
        for i in range(max_iter):
            pmatrix = self._calc_pmatrix(X)
            alpha, c = self._forward(pmatrix)
            beta = self._backward(pmatrix, c)
            gamma, xi = self._estep(pmatrix, alpha, beta, c)
            self._mstep(X, gamma, xi)

            log_likelihood_prev = log_likelihood
            log_likelihood = np.sum(np.log(c))
            if abs(log_likelihood - log_likelihood_prev) < tol:
                break
        print(f"The number of iteration : {i}")
        print(f"Converged : {i < max_iter - 1}")
        print(f"log likelihood : {log_likelihood}")

    def predict_proba(self, X):
        '''
        Calculates and returns the probability that latent variables corresponding to the input X are in each class.

        Parameters
        ----------
        X : 2D numpy array
            2D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.

        Returns
        ----------
        gamma : 2D numpy array
            (len(X), self.K) numpy array, where gamma[n, k] represents the probability that 
            the lantent variable corresponding to the n-th sample of X belongs to the k-th class.
        '''
        pmatrix = self._calc_pmatrix(X)
        alpha, c = self._forward(pmatrix)
        beta = self._backward(pmatrix, c)
        gamma = alpha * beta
        return gamma

    def predict(self, X):
        '''
        Calculates and returns which classes the latent variables corresponding to the input X are in.

        Parameters
        ----------
        X : 2D numpy array
            2D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.

        Returns
        ----------
        pred : 2D numpy array
            (len(X), self.K) numpy array, where gamma[n, k] represents the probability that 
            the lantent variable corresponding to the n-th sample of X belongs to the k-th class.
        '''
        pred = self.predict_proba(X).argmax(axis=1)
        return pred


N = 200 # the number of data

# Here we consider a heavily biased coin.
mu0 = 0.1 
mu1 = 0.8

tp = 0.03 # transition probability

rv_cointoss = np.random.RandomState(seed=0).rand(N)
rv_transition = np.random.RandomState(seed=1).rand(N)

X = np.zeros((N, 1))
states = np.zeros(N)
current_state = 0
for n in range(N):
    states[n] = current_state
    if rv_cointoss[n] < mu0*(int(not(current_state))) + mu1*current_state:
        X[n][0] = 1.0
    if rv_transition[n] < tp:
        current_state = int(not(current_state))

# Plot the true states
plt.figure(figsize=(14, 4))
plt.plot(states, label='True states')

# Fit the HMM
hmm = HMM(K=2)
hmm.fit(X)

# Predict the states
predicted_states = hmm.predict(X)

# Plot the predicted states
plt.plot(predicted_states, label='Predicted states')
plt.legend()
plt.show()


