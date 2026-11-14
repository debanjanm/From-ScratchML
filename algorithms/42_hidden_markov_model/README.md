# Hidden Markov Model

> **Notebook:** [`notebook.ipynb`](notebook.ipynb) has the explanation, the implementation, visualizations, a scikit-learn check and the tests, all in one place.

A sequence model where the thing you care about is **hidden**. A hidden state hops around a Markov chain, and at every step it emits a visible symbol. You see only the symbols (umbrella, no umbrella…) and want to reason about the states (rain, sun…). Used for speech, part-of-speech tagging, gene finding, regime detection.

## Parameters

| | meaning |
|---|---|
| `pi[i]` | probability the chain starts in state `i` |
| `A[i, j]` | probability of moving from state `i` to `j` |
| `B[i, k]` | probability that state `i` emits symbol `k` |

## The three classic problems — one method each

| Question | Method | Algorithm |
|---|---|---|
| How likely is this sequence? | `score(obs)` | **Forward**: `αₜ(j) = [Σᵢ αₜ₋₁(i) A[i,j]] · B[j, oₜ]` — sums over all `Nᵀ` paths in `O(T·N²)` |
| What states most likely produced it? | `decode(obs)` | **Viterbi**: same recursion with `max` instead of `Σ`, then trace back |
| Learn the parameters from sequences | `fit(seqs)` | **Baum-Welch** (EM): E step uses forward-backward to get `γₜ(i)` = P(state `i` at `t` \| everything) and `ξₜ(i,j)`; M step re-estimates `pi, A, B` from expected counts |

`predict_proba(obs)` gives `γ` itself — the soft version of Viterbi. `sample(length)` generates data from the model.

Messages are rescaled at every time step (`cₜ`) so a 20 000-step sequence does not underflow to 0; the log-likelihood is `Σ log cₜ`.

## Things to know

- Baum-Welch **never decreases** the likelihood (a test checks it) but only finds a *local* optimum, so `fit` restarts `n_init` times and keeps the best.
- Learned states come back in arbitrary order: state 0 is not necessarily your "sunny" — match by looking at `B`.
- The tests verify forward, Viterbi and posteriors against **brute-force enumeration of every state path** on a tiny model; the demo checks them against `hmmlearn` if it is installed (optional, `pip install hmmlearn`).

`ponytail:` discrete symbols only. Gaussian emissions (continuous observations) would swap `B[i, k]` for a density per state; the forward/Viterbi/EM structure is identical.

## Usage

```python
# (the class lives in notebook.ipynb)

hmm = HiddenMarkovModel(n_states=2, n_symbols=3).fit(sequences)   # list of int arrays, or one array
hmm.A_, hmm.B_, hmm.pi_
hmm.score(obs)           # log-likelihood
hmm.decode(obs)          # (most likely states, log-prob)
hmm.predict_proba(obs)   # P(state_t | whole sequence)
```
