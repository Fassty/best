import numpy as np
import random
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pymc.distributions import noncentral_t_like
from matplotlib.transforms import blended_transform_factory

from best.utils import get_statistics


def setup_plot(ax, name):
    symbol_mapping = {
        'Mean': r'$\mu$',
        'Std. dev': r'$\sigma$',
        'Effect size': r'$(\mu - 0)/\sigma$',
        'Normality': r'$\mathrm{log10}(\nu)$'
    }
    sns.despine(bottom=False, ax=ax)
    ax.spines['left'].set_visible(False)
    ax.set_title(name, fontweight='bold')
    ax.set_xlabel(symbol_mapping[name], fontsize=12)
    ax.yaxis.set_ticks([])  # don't draw y ticks


def plot_hdi(ax, hdi_max, hdi_min, trans):
    sns.lineplot([hdi_min, hdi_max], [0, 0], lw=5.0, color='black', ax=ax)
    ax.text(
        hdi_min, 0.08, f'{hdi_min:.3f}',
        transform=trans,
        horizontalalignment='center',
        verticalalignment='bottom',
    )
    ax.text(
        hdi_max, 0.08, f'{hdi_max:.3f}',
        transform=trans,
        horizontalalignment='center',
        verticalalignment='bottom',
    )
    ax.text(
        (hdi_min + hdi_max) / 2, 0.16, '95% HDI',
        transform=trans,
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=14
    )


def plot_ROPE(ax, p_rope, trans, r):
    # ROPE boundaries
    ax.axvline(-r, ymax=0.3, linestyle=':', color='darkred', lw=2.0)
    ax.axvline(r, ymax=0.3, linestyle=':', color='darkred', lw=2.0)
    # Percent in ROPE
    ax.text(
        0, 0.32, f'{p_rope:.0f}% in ROPE',
        transform=trans,
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=10,
        color='darkred'
    )


def plot_zero(ax):
    ax.axvline(0, ymax=0.5, linestyle=':')


def plot_symmetry_around_zero(ax, mode, p_above_0, trans):
    ax.text(
        mode, 0.75, f'{100 - p_above_0:.1f}% < 0 < {p_above_0:.1f}%',
        transform=trans,
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=13,
        color='green'
    )


def plot_mode(ax, mode, trans):
    ax.text(
        mode, 0.99, f'mode = {mode:.3f}',
        transform=trans,
        horizontalalignment='center',
        verticalalignment='top',
        fontsize=11
    )


def plot_statistic(ax, trace, name, rope_width, plot_0=False):
    sample_vec = trace[name]

    if name == 'Normality':
        sample_vec = np.log10(sample_vec + 1)

    hdi_max, hdi_min, mode, p_above_0, p_rope = get_statistics(sample_vec, rope_width)

    setup_plot(ax, name)

    sns.kdeplot(sample_vec, color='#89d1ea', lw=3.0, ax=ax)

    trans = blended_transform_factory(ax.transData, ax.transAxes)

    plot_hdi(ax, hdi_max, hdi_min, trans)
    plot_mode(ax, mode, trans)

    if plot_0:
        plot_symmetry_around_zero(ax, mode, p_above_0, trans)
        plot_zero(ax)

    if rope_width > 0:
        plot_ROPE(ax, p_rope, trans, rope_width)


def plot_posterior_distribution(ax, trace, model):
    observations = model['posterior_dist'].observations
    means = trace['Mean']
    stds = trace['Std. dev']
    numos = trace['Normality']

    bins = np.linspace(np.min(observations), np.max(observations), 30)

    ax.hist(observations, bins=bins, rwidth=0.5, facecolor='r', edgecolor='none', density=True)
    ax.set_xlabel('y', fontweight='bold')
    ax.set_ylabel('p(y)', fontweight='bold')

    idxs = random.sample(list(range(len(means))), 50)
    x = np.linspace(bins[0], bins[-1], 100)

    for i in idxs:
        mu = means[i]
        std = stds[i]
        lam = 1 / std ** 2
        nu = numos[i] + 1

        v = np.exp([noncentral_t_like(xi, mu, lam, nu) for xi in x])
        ax.plot(x, v, color='#89d1ea', zorder=-10)

    ax.text(
        0.8, 0.95, r'$\mathrm{N}=%d$' % len(observations),
        transform=ax.transAxes,
        horizontalalignment='left',
        verticalalignment='top'
    )

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.set_title(f'Data w. Post. Pred.', fontweight='bold')


def plot(model, trace, rope_width=0.1, figure_name='posterior.png'):
    figure = plt.figure(figsize=(8.2, 8), facecolor='white', constrained_layout=True)

    gs = figure.add_gridspec(3, 2)

    sns.set_style('ticks', {'axes.linewidth': 2, 'axes.edgecolor': 'black'})

    ax1 = figure.add_subplot(gs[0, 0])
    plot_statistic(ax1, trace, 'Mean', rope_width, plot_0=True)

    ax2 = figure.add_subplot(gs[1, 0])
    plot_statistic(ax2, trace, 'Std. dev', rope_width=0)

    ax3 = figure.add_subplot(gs[2, 0])
    plot_statistic(ax3, trace, 'Effect size', rope_width=0, plot_0=True)

    ax4 = figure.add_subplot(gs[2, 1])
    plot_statistic(ax4, trace, 'Normality', rope_width=0)

    ax5 = figure.add_subplot(gs[:-1, 1])
    plot_posterior_distribution(ax5, trace, model)

    plt.tight_layout()

    plt.savefig(figure_name)
