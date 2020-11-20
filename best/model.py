import pymc3 as pm
import numpy as np

import logging
logger = logging.getLogger('pymc3')
logger.setLevel(logging.ERROR)


class Model:
    PRECISION_SCALING = 1e-6
    SIGMA_SCALING = 1e3
    NORMALITY_THRESHOLD = 29  # since 30 the distribution is considered normal

    def __init__(self, model: pm.Model):
        super().__init__()

        self.model = model

    @classmethod
    def from_custom_model(cls, model: pm.Model):
        """
        As of now this method assumes that the posterior distribution has a name `posterior_dist` and
        that it contains 4 parameters: Normality, Mean, Std. dev, Effect size.

        Passing a different model is fine for sampling but the plotting will fail
        """
        return cls(model)

    @classmethod
    def from_two_groups(cls, group_1: np.ndarray, group_2: np.ndarray):
        diff = group_1 - group_2
        return Model.from_one_group(diff)

    @classmethod
    def from_one_group(cls, diff: np.ndarray):
        diff_std = np.std(diff).item()
        diff_mean = np.mean(diff).item()
        mu = diff_mean
        tau = cls.PRECISION_SCALING / np.power(diff_std, 2)
        sigma_low = diff_std / cls.SIGMA_SCALING
        sigma_high = diff_std * cls.SIGMA_SCALING

        with pm.Model() as model:
            mu = pm.Normal(name='prior_mu', mu=mu, tau=tau)
            sigma = pm.Uniform(name='prior_sigma', lower=sigma_low, upper=sigma_high)
            nu = pm.Exponential(name='Normality', lam=1 / cls.NORMALITY_THRESHOLD) + 1
            lam = sigma ** -2

            r = pm.StudentT('posterior_dist', mu=mu, lam=lam, nu=nu, observed=diff)

            mean = pm.Deterministic('Mean', mu)
            std = pm.Deterministic('Std. dev', sigma)
            effect_size = pm.Deterministic('Effect size', mu / sigma)

        return cls(model)

    def sample(self, it: int = 110000):
        with self.model:
            trace = pm.sample(it, chains=1, progressbar=False)

        return self.model, trace
