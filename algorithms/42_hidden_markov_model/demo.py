import numpy as np
from model import HiddenMarkovModel

# Classic example: hidden weather (0=sunny, 1=rainy) decides what you see (0=walk, 1=shop, 2=umbrella)
truth = HiddenMarkovModel(n_states=2, n_symbols=3)
truth.pi_ = np.array([0.7, 0.3])
truth.A_ = np.array([[0.85, 0.15], [0.30, 0.70]])
truth.B_ = np.array([[0.60, 0.30, 0.10], [0.05, 0.25, 0.70]])
states, obs = truth.sample(3000, random_state=0)

# 1) Viterbi: recover the hidden weather from what was observed
path, _ = truth.decode(obs)
print("Viterbi recovers the true hidden state at %.1f%% of days (guessing the majority: %.1f%%)"
      % (100 * (path == states).mean(), 100 * max((states == 0).mean(), (states == 1).mean())))

# 2) Baum-Welch: learn the model from the observations alone (state labels are never shown)
fit = HiddenMarkovModel(n_states=2, n_symbols=3, random_state=0).fit(obs)
order = [0, 1] if fit.B_[0, 2] < fit.B_[1, 2] else [1, 0]      # state numbering is arbitrary: sort by umbrella rate
print("learned transition matrix:\n", fit.A_[np.ix_(order, order)].round(2), "\n true:\n", truth.A_)
print("learned emissions:\n", fit.B_[order].round(2), "\n true:\n", truth.B_)
print("log-likelihood: learned=%.1f  true model=%.1f" % (fit.score(obs), truth.score(obs)))

# 3) cross-check against hmmlearn when it is installed (pip install hmmlearn)
try:
    from hmmlearn.hmm import CategoricalHMM

    ref = CategoricalHMM(n_components=2)
    ref.startprob_, ref.transmat_, ref.emissionprob_ = truth.pi_, truth.A_, truth.B_
    X = obs.reshape(-1, 1)
    assert np.isclose(truth.score(obs), ref.score(X))
    assert np.allclose(truth.predict_proba(obs), ref.predict_proba(X))
    assert (truth.decode(obs)[0] == ref.predict(X)).all()
    print("score, posterior probabilities and Viterbi path identical to hmmlearn")

    # EM finds local optima, so give hmmlearn the same 5 random restarts as ours and keep its best
    runs = [CategoricalHMM(n_components=2, n_iter=100, tol=1e-6, random_state=seed, n_features=3).fit(X) for seed in range(5)]
    best = max(r.score(X) for r in runs)
    print("after fitting: ours=%.1f hmmlearn(best of 5)=%.1f" % (fit.score(obs), best))
    assert abs(fit.score(obs) - best) < 1
except ImportError:
    print("hmmlearn not installed; skipped the comparison")
