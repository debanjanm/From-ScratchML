import numpy as np


class HiddenMarkovModel:
    """Discrete-observation HMM: a hidden Markov chain of states that emits visible symbols.

    Parameters
        pi[i]   = P(first state is i)
        A[i, j] = P(next state is j | state is i)       (rows sum to 1)
        B[i, k] = P(emit symbol k | state is i)         (rows sum to 1)

    Three classic problems, one method each:
      score(obs)    "how likely is this sequence?"      forward algorithm
      decode(obs)   "which states most likely made it?" Viterbi (best single path)
      fit(seqs)     "learn pi, A, B from sequences"      Baum-Welch (EM with forward-backward)
    plus predict_proba(obs) = posterior P(state_t | whole sequence) and sample().

    Forward/backward messages are rescaled at every step (c_t) so long sequences do not underflow;
    log-likelihood is the sum of log c_t.

    ponytail: Baum-Welch only finds a local optimum, so it restarts `n_init` times from random
    parameters and keeps the best; no priors / smoothing beyond a tiny floor."""

    def __init__(self, n_states=2, n_symbols=2, n_iter=100, tol=1e-6, n_init=5, random_state=0):
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.n_iter = n_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

    # ----- inference -----
    def _forward(self, obs):
        T = len(obs)
        alpha = np.zeros((T, self.n_states))
        c = np.zeros(T)
        alpha[0] = self.pi_ * self.B_[:, obs[0]]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A_) * self.B_[:, obs[t]]
            c[t] = alpha[t].sum()
            alpha[t] /= c[t]
        return alpha, c

    def _backward(self, obs, c):
        T = len(obs)
        beta = np.ones((T, self.n_states))
        for t in range(T - 2, -1, -1):
            beta[t] = self.A_ @ (self.B_[:, obs[t + 1]] * beta[t + 1]) / c[t + 1]
        return beta

    def score(self, obs):
        """log P(obs)."""
        return np.log(self._forward(np.asarray(obs))[1]).sum()

    def predict_proba(self, obs):
        """P(state_t = i | all observations), shape (T, n_states)."""
        obs = np.asarray(obs)
        alpha, c = self._forward(obs)
        return alpha * self._backward(obs, c)

    def decode(self, obs):
        """Viterbi: the single most likely state sequence, and its log-probability."""
        obs = np.asarray(obs)
        T = len(obs)
        logA, logB = np.log(self.A_ + 1e-300), np.log(self.B_ + 1e-300)
        delta = np.log(self.pi_ + 1e-300) + logB[:, obs[0]]
        back = np.zeros((T, self.n_states), dtype=int)
        for t in range(1, T):
            scores = delta[:, None] + logA            # [from, to]
            back[t] = scores.argmax(axis=0)
            delta = scores.max(axis=0) + logB[:, obs[t]]
        path = [int(delta.argmax())]
        for t in range(T - 1, 0, -1):
            path.append(int(back[t][path[-1]]))
        return np.array(path[::-1]), delta.max()

    def sample(self, length, random_state=None):
        rng = np.random.default_rng(random_state)
        states, obs = np.zeros(length, dtype=int), np.zeros(length, dtype=int)
        s = rng.choice(self.n_states, p=self.pi_)
        for t in range(length):
            states[t] = s
            obs[t] = rng.choice(self.n_symbols, p=self.B_[s])
            s = rng.choice(self.n_states, p=self.A_[s])
        return states, obs

    # ----- learning -----
    def fit(self, sequences):
        """Baum-Welch on one sequence or a list of sequences of integer symbols 0..n_symbols-1."""
        if np.ndim(sequences[0]) == 0:
            sequences = [sequences]
        sequences = [np.asarray(s) for s in sequences]
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            self.pi_ = rng.dirichlet(np.ones(self.n_states))
            self.A_ = rng.dirichlet(np.ones(self.n_states), size=self.n_states)
            self.B_ = rng.dirichlet(np.ones(self.n_symbols), size=self.n_states)
            history = self._baum_welch(sequences)
            if best is None or history[-1] > best[0]:
                best = (history[-1], self.pi_, self.A_, self.B_, history)
        _, self.pi_, self.A_, self.B_, self.log_likelihoods_ = best
        return self

    def _baum_welch(self, sequences):
        K, S, history = self.n_symbols, self.n_states, []
        for _ in range(self.n_iter):
            pi = np.zeros(S)
            A_num, A_den = np.zeros((S, S)), np.zeros(S)
            B_num = np.zeros((S, K))
            total = 0.0
            for obs in sequences:                                    # E step
                alpha, c = self._forward(obs)
                beta = self._backward(obs, c)
                gamma = alpha * beta
                total += np.log(c).sum()
                pi += gamma[0]
                for t in range(len(obs) - 1):
                    A_num += alpha[t][:, None] * self.A_ * (self.B_[:, obs[t + 1]] * beta[t + 1])[None, :] / c[t + 1]
                A_den += gamma[:-1].sum(axis=0)
                for k in range(K):
                    B_num[:, k] += gamma[obs == k].sum(axis=0)
            history.append(total)
            eps = 1e-10                                              # M step (tiny floor avoids 0/0)
            self.pi_ = (pi + eps) / (pi + eps).sum()
            self.A_ = (A_num + eps) / (A_num + eps).sum(axis=1, keepdims=True)
            self.B_ = (B_num + eps) / (B_num + eps).sum(axis=1, keepdims=True)
            if len(history) > 1 and abs(history[-1] - history[-2]) < self.tol:
                break
        return history
