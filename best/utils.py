import numpy as np
import scipy


def hdi_of_mcmc(sample_vec, cred_mass=0.95):
    assert len(sample_vec), 'need points to find HDI'
    sorted_pts = np.sort(sample_vec)

    ci_idx_inc = int(np.floor(cred_mass * len(sorted_pts)))
    n_cis = len(sorted_pts) - ci_idx_inc
    ci_width = sorted_pts[ci_idx_inc:] - sorted_pts[:n_cis]

    min_idx = np.argmin(ci_width)
    hdi_min = sorted_pts[min_idx]
    hdi_max = sorted_pts[min_idx + ci_idx_inc]
    return hdi_min, hdi_max


def calculate_statistics(sample_vec, rope_width):
    dist = scipy.stats.gaussian_kde(sample_vec)
    bw = dist.covariance_factor()
    cut = 3 * bw
    x = np.linspace(np.min(sample_vec) - cut * bw, np.max(sample_vec) + cut * bw, 512)
    vals = dist.evaluate(x)
    max_idx = np.argmax(vals)
    mode = x[max_idx]

    p_rope = np.round(dist.integrate_box_1d(-rope_width, rope_width) * 100)

    p_above_zero = np.round(dist.integrate_box_1d(0, np.inf) * 100, 1)

    hdi_min, hdi_max = hdi_of_mcmc(sample_vec)

    return {
        'mode': mode,
        'p_rope': p_rope,
        'p_above_0': p_above_zero,
        'hdi_min': hdi_min,
        'hdi_max': hdi_max
    }


def get_statistics(sample_vec, rope_width):
    statistics = calculate_statistics(sample_vec, rope_width)
    mode = statistics['mode']
    p_rope = statistics['p_rope']
    p_above_0 = statistics['p_above_0']
    hdi_min = statistics['hdi_min']
    hdi_max = statistics['hdi_max']
    return hdi_max, hdi_min, mode, p_above_0, p_rope


def get_mean_metrics(trace, rope_width):
    means = trace['Mean']
    _, _, mode, p_above_0, p_rope = get_statistics(means, rope_width)

    return mode, p_above_0, p_rope
