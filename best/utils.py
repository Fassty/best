import numpy as np
import scipy


def calculate_hdi_range(sample_vec, cred_mass=0.95):
    assert len(sample_vec), 'No data were passed for calculation of HDI'
    sorted_pts = np.sort(sample_vec)

    ci_idx_inc = int(np.floor(cred_mass * len(sorted_pts)))
    n_cis = len(sorted_pts) - ci_idx_inc
    ci_width = sorted_pts[ci_idx_inc:] - sorted_pts[:n_cis]

    min_idx = np.argmin(ci_width)
    hdi_min = sorted_pts[min_idx]
    hdi_max = sorted_pts[min_idx + ci_idx_inc]
    return hdi_min, hdi_max


def calculate_mode(dist, sample_vec):
    bw = dist.covariance_factor()
    cut = 3 * bw
    x = np.linspace(np.min(sample_vec) - cut * bw, np.max(sample_vec) + cut * bw, 512)
    max_idx = np.argmax(dist.evaluate(x))
    mode = x[max_idx]
    return mode


def calculate_statistics(sample_vec, rope_width):
    dist = scipy.stats.gaussian_kde(sample_vec)

    hdi_min, hdi_max = calculate_hdi_range(sample_vec)
    mode = calculate_mode(dist, sample_vec)
    p_above_zero = np.round(dist.integrate_box_1d(0, np.inf) * 100, 1)
    p_rope = np.round(dist.integrate_box_1d(-rope_width, rope_width) * 100)

    return hdi_max, hdi_min, mode, p_above_zero, p_rope


def get_mean_metrics(trace, rope_width):
    means = trace['Mean']
    _, _, mode, p_above_0, p_rope = calculate_statistics(means, rope_width)

    return mode, p_above_0, p_rope
