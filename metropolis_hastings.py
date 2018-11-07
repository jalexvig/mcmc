import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


def proposal(x, sigma=0.5):
    """
    Propose new parameters.
    """

    x_next = stats.norm(x, sigma).rvs(size=x.shape)

    return x_next


def log_likelihood(data, mu, sigma=1):
    """
    Calculate the log likelihood of some data assuming a normal distribution.
    """

    log_prob = np.log(stats.norm(mu, sigma).pdf(data)).sum()

    return log_prob


def log_prior(parameter, mu=0, sigma=1):
    """
    Calculate the log probability of some parameter assuming a normal prior distribution on that parameter.
    """

    log_probs = np.log(stats.norm(mu, sigma).pdf(parameter))

    return log_probs


def metropolis_hastings(data, n_iterations=1000):
    """
    Calculate the posterior distribution for parameters given some data.
    """

    x_current = proposal(np.ones(1))
    ll_current = log_likelihood(data, x_current)
    log_prior_current = log_prior(x_current)

    posteriors = [x_current]

    for i in range(n_iterations):

        x_proposal = proposal(x_current)
        ll_proposal = log_likelihood(data, x_proposal)
        log_prior_proposal = log_prior(x_proposal)

        accept = np.log(np.random.rand()) < ll_proposal + log_prior_proposal - ll_current - log_prior_current

        if accept:
            x_current, ll_current = x_proposal, ll_proposal

        posteriors.append(x_current)

    return posteriors


if __name__ == '__main__':

    np.random.seed(0)

    data = np.random.randn(1000) + 0.5

    print(data.mean(), data.std())

    posteriors = metropolis_hastings(data)

    plt.hist([x[0] for x in posteriors[100:]])
    plt.show()
