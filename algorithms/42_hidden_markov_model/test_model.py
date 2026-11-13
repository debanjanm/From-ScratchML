import itertools
import numpy as np
from model import HiddenMarkovModel


def _small_model():
    m = HiddenMarkovModel(n_states=3, n_symbols=3)
    m.pi_ = np.array([0.5, 0.3, 0.2])
    m.A_ = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    m.B_ = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    return m


def _joint(m, states, obs):
    p = m.pi_[states[0]] * m.B_[states[0], obs[0]]
    for t in range(1, len(obs)):
        p *= m.A_[states[t - 1], states[t]] * m.B_[states[t], obs[t]]
    return p


def test_forward_and_viterbi_match_brute_force_over_all_paths():
    m = _small_model()
    obs = [0, 1, 1, 2, 0, 2]
    paths = list(itertools.product(range(3), repeat=len(obs)))          # all 3^6 state sequences
    probs = np.array([_joint(m, p, obs) for p in paths])
    assert np.isclose(m.score(obs), np.log(probs.sum()))                # forward = sum over every path
    path, logp = m.decode(obs)
    assert tuple(path) == paths[probs.argmax()] and np.isclose(logp, np.log(probs.max()))  # Viterbi = best path


def test_posterior_state_probabilities_match_brute_force():
    m = _small_model()
    obs = [2, 2, 0, 1]
    paths = list(itertools.product(range(3), repeat=len(obs)))
    probs = np.array([_joint(m, p, obs) for p in paths])
    probs /= probs.sum()
    expected = np.zeros((len(obs), 3))
    for p, pr in zip(paths, probs):
        for t, s in enumerate(p):
            expected[t, s] += pr
    assert np.allclose(m.predict_proba(obs), expected)


def test_long_sequences_do_not_underflow():
    m = _small_model()
    _, obs = m.sample(20000, random_state=0)
    assert np.isfinite(m.score(obs)) and m.score(obs) < -1000          # raw probabilities would be 0.0


def test_baum_welch_never_decreases_likelihood_and_recovers_parameters():
    true = HiddenMarkovModel(n_states=2, n_symbols=3)
    true.pi_ = np.array([0.6, 0.4])
    true.A_ = np.array([[0.9, 0.1], [0.2, 0.8]])
    true.B_ = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    _, obs = true.sample(5000, random_state=1)
    fit = HiddenMarkovModel(n_states=2, n_symbols=3, random_state=0).fit(obs)
    assert (np.diff(fit.log_likelihoods_) > -1e-8).all()                # EM guarantee
    perms = [(0, 1), (1, 0)]                                            # hidden states may come back swapped
    err = min(np.abs(fit.A_[np.ix_(p, p)] - true.A_).max() + np.abs(fit.B_[list(p)] - true.B_).max() for p in perms)
    assert err < 0.1
    assert fit.score(obs) >= true.score(obs) - 5                        # fits at least as well as the truth


def test_viterbi_recovers_hidden_states():
    true = _small_model()
    states, obs = true.sample(2000, random_state=2)
    assert (true.decode(obs)[0] == states).mean() > 0.75


if __name__ == "__main__":
    test_forward_and_viterbi_match_brute_force_over_all_paths()
    test_posterior_state_probabilities_match_brute_force()
    test_long_sequences_do_not_underflow()
    test_baum_welch_never_decreases_likelihood_and_recovers_parameters()
    test_viterbi_recovers_hidden_states()
    print("all tests passed")
