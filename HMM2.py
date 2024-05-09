import numpy as np
import unittest

class HMM:
    def __init__(self, n_states, n_obs):
        self.n_states = n_states
        self.n_obs = n_obs

        self.trans_probs = np.random.rand(n_states, n_states)
        self.trans_probs /= np.sum(self.trans_probs, axis=1, keepdims=True)

        self.emit_probs = np.random.rand(n_states, n_obs)
        self.emit_probs /= np.sum(self.emit_probs, axis=1, keepdims=True)

        self.init_probs = np.random.rand(n_states)
        self.init_probs /= np.sum(self.init_probs)

    def forward(self, obs_seq):
        alpha = np.zeros((len(obs_seq), self.n_states))
        alpha[0] = self.init_probs * self.emit_probs[:, obs_seq[0]]

        for t in range(1, len(obs_seq)):
            alpha[t] = self.emit_probs[:, obs_seq[t]] * np.dot(alpha[t-1], self.trans_probs)

        return alpha

    def backward(self, obs_seq):
        beta = np.zeros((len(obs_seq), self.n_states))
        beta[-1] = 1

        for t in range(len(obs_seq)-2, -1, -1):
            beta[t] = np.dot(beta[t+1] * self.emit_probs[:, obs_seq[t+1]], self.trans_probs.T)

        return beta

    def train(self, obs_seq, n_iter=100):
        for _ in range(n_iter):
            alpha = self.forward(obs_seq)
            beta = self.backward(obs_seq)

            xi = np.zeros((len(obs_seq)-1, self.n_states, self.n_states))
            for t in range(len(obs_seq)-1):
                xi[t] = alpha[t].reshape(-1, 1) * self.trans_probs * self.emit_probs[:, obs_seq[t+1]] * beta[t+1]
                xi[t] /= (np.sum(xi[t]) + 1e-10)

            gamma = alpha * beta / (np.sum(alpha * beta, axis=1, keepdims=True) + 1e-10)

            self.trans_probs = np.sum(xi, axis=0) / (np.sum(gamma[:-1], axis=0).reshape(-1, 1) + 1e-10)
            self.emit_probs = np.zeros((self.n_states, self.n_obs))
            for s in range(self.n_states):
                for o in range(self.n_obs):
                    self.emit_probs[s, o] = np.sum(gamma[obs_seq == o, s]) / (np.sum(gamma[:, s]) + 1e-10)

            self.init_probs = gamma[0]

class TestHMM(unittest.TestCase):
    def setUp(self):
        n_states = 100
        n_obs = 100
        self.hmm = HMM(n_states, n_obs)  # Initialize HMM with n_states and n_obs

    def test_train(self):
        n_states = 100
        n_obs = 100
        obs_seq = np.random.randint(0, n_obs, 1000)  # Longer observation sequence with n_obs different observations

        # Set the true parameters
        true_trans_probs = np.full((n_states, n_states), 1.0/n_states)  # n_states x n_states matrix with all entries 1/n_states
        true_emit_probs = np.full((n_states, n_obs), 1.0/n_obs)  # n_states x n_obs matrix with all entries 1/n_obs
        true_init_probs = np.full(n_states, 1.0/n_states)  # n_states-dimensional vector with all entries 1/n_states

        self.hmm.train(obs_seq, n_iter=1000)  # Train the HMM with more iterations

        # Calculate the mean absolute error (MAE) between the trained parameters and the true parameters
        trans_mae = np.mean(np.abs(self.hmm.trans_probs - true_trans_probs))
        emit_mae = np.mean(np.abs(self.hmm.emit_probs - true_emit_probs))
        init_mae = np.mean(np.abs(self.hmm.init_probs - true_init_probs))

        # Calculate the accuracy in percentage
        trans_acc = (1 - trans_mae) * 100
        emit_acc = (1 - emit_mae) * 100
        init_acc = (1 - init_mae) * 100

        print(f'Transition probabilities accuracy: {trans_acc}%')
        print(f'Emission probabilities accuracy: {emit_acc}%')
        print(f'Initial probabilities accuracy: {init_acc}%')

if __name__ == '__main__':
    unittest.main()