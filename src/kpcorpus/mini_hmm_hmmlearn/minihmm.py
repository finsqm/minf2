from hmmlearn import base
from collections import Counter
import numpy as np

class MiniHMM(base._BaseHMM):

    def __init__(
            self, n_components=1,
            startprob_prior=1.0, transmat_prior=1.0,
            algorithm="viterbi", random_state=None,
            n_iter=10, tol=0.01,
            verbose=False, params="st",
            init_params="st", x_vals=12):

        super(self.__class__, self).init(
            n_components,
            startprob_prior, transmat_prior,
            algorithm, random_state,
            n_iter, tol,
            verbose, params,
            init_params)

        self.x_vals = x_vals


    def _init(self, X, lengths):

        super(self.__class__, self)._init(X, lengths)
        self.model = np.zeros((self.n_components, self.x_vals))


    def _check(self):

        super(self.__class__, self)._check()


    def _generate_sample_from_state(self, state):

        return np.random.choice(range(self.x_vals), p=self.model[state])


    def _compute_log_likelihood(self, X):

        return self.model[:, X].T


    def _initialize_sufficient_statistics(self):

        pass


    def _accumulate_sufficient_statistics(self, stats, X, framelogprob, posteriors, fwdlattice, bwdlattice):

        pass


    def _do_mstep(self, stats):

        pass
