import pymc3 as pm
import numpy as np
from best.plot import plot
from best.utils import get_mean_metrics

class Model:
    PRECISION_SCALING = 1e-6
    SIGMA_SCALING = 1e3
    NORMALITY_THRESHOLD = 29  # since 30 the distribution is considered normal

    def __init__(self, group_1: np.ndarray, group_2: np.ndarray):
        if not np.all(np.diff(group_1) >= 0):
            np.sort(group_1)
        if not np.all(np.diff(group_1) >= 0):
            np.sort(group_2)

        self._diff = group_1 - group_2
        self._diff_std = np.std(self._diff).item()
        self._diff_mean = np.mean(self._diff).item()
        self._mu = self._diff_mean
        self._tau = self.PRECISION_SCALING / np.power(self._diff_std, 2)
        self._sigma_low = self._diff_std / self.SIGMA_SCALING
        self._sigma_high = self._diff_std * self.SIGMA_SCALING

    def sample(self, it: int = 110000):
        with pm.Model() as posterior:
            mu = pm.Normal(name='prior_mu', mu=self._mu, tau=self._tau)
            sigma = pm.Uniform(name='prior_sigma', lower=self._sigma_low, upper=self._sigma_high)
            nu = pm.Exponential(name='Normality', lam=1 / self.NORMALITY_THRESHOLD) + 1
            lam = sigma ** -2

            r = pm.StudentT('posterior_dist', mu=mu, lam=lam, nu=nu, observed=self._diff)

            diff_of_means = pm.Deterministic('Mean', mu)
            diff_of_stds = pm.Deterministic('Std. dev', sigma)
            effect_size = pm.Deterministic('Effect size', mu / sigma)

        with posterior:
            trace = pm.sample(it)

        return posterior, trace

if __name__ == '__main__':
    drug = (101, 100, 102, 104, 102, 97, 105, 105, 98, 101, 100, 123, 105, 103, 100, 95, 102, 106,
            109, 102, 82, 102, 100, 102, 102, 101, 102, 102, 103, 103, 97, 97, 103, 101, 97, 104,
            96, 103, 124, 101, 101, 100, 101, 101, 104, 100, 101)
    placebo = (99, 101, 100, 101, 102, 100, 97, 101, 104, 101, 102, 102, 100, 105, 88, 101, 100,
               104, 100, 100, 100, 101, 102, 103, 97, 101, 101, 100, 101, 99, 101, 100, 100,
               101, 100, 99, 101, 100, 102, 99, 100, 99, 100, 100, 100, 100, 100)

    drug = np.array(drug)
    placebo = np.array(placebo)
    drug.sort()
    placebo.sort()

    model = Model(drug, placebo)
    trace = model.sample(rope_width=0.5, plot_posterior=True)

    mode, p_above_0, p_rope = get_mean_metrics(trace, rope_width=0.5)

    if p_above_0 > 91 and p_rope < 15:
        print('The drug really works, who would have thought')
