#  diegoandresalvarez/subset_simulation 

import numpy as np
from scipy.stats import uniform


# Modified Metropolis Algorithm
def MMA(theta0, g0, N, g_lim, b, pim, spread=1, g_failure='>=0'):
    ''' Modified Metropolis Algorithm

    Usage:
        theta, g = MMA(theta0, g0, N, g_lim, b, pim, spread=1)

    Input parameters:
        theta0: initial state of the Markov chain (seed)
        g0:     evaluation on the g_lim of theta0, i.e., g_lim(theta_0)
        N:      number of samples to be drawn
        g_lim:  limit state function g(x)
        b:      threshold (it defines the failure region F = {x : g(x) > b})
        pim:    list with the marginal PDFs of theta_1, to theta_d
        spread: spread of the proposal PDF (spread=1 by default)
        g_failure:     definition of the failure region; the possible options
                       are: >=0' (default) and '<=0'

    Output parameters:
        theta:      sampled points
        g:          g_lim(theta) for all samples
        num_eval_g: number of evaluations of the limit state function g_lim
        ar_rate:    acceptance-rejection rate
    '''
    if g_failure == '>=0':
        sign_g = 1
        if g0 < b:
            raise Exception('The initial sample does not belong to F')
    elif g_failure == '<=0':
        sign_g = -1
        if g0 < b:
            raise Exception('The initial sample does not belong to F')
    else:
        raise Exception("g_failure must be either '>=0' or '<=0'.")

    d = len(theta0)  # number of parameters (dimension of theta)

    if len(pim) != d:
        raise Exception('"pim" and "theta0" should have the same length')

    theta = np.zeros((N,d));   theta[0,:] = theta0
    g     = np.zeros(N);       g[0]       = g0
    num_eval_g = 0  # number of evaluations of g
    num_accepted = 0  # number of accepted samples

    for i in range(N-1):
        # generate a candidate state hat_xi
        xi = np.zeros(d)

        for k in range(d):
            # the proposal PDFs are defined (the must be symmetric)
            # we will use a uniform PDF in [loc, loc+scale]
            Sk = uniform(loc=theta[i,k] - spread, scale=2*spread)

            # a sample is drawn from the proposal Sk
            hat_xi_k = Sk.rvs()

            # compute the acceptance ratio
            r = pim[k].pdf(hat_xi_k)/pim[k].pdf(theta[i][k])

            # acceptance/rejection step:                        
            if np.random.rand() <= min(1, r):
                xi[k] = hat_xi_k     # accept the candidate
            else:
                xi[k] = theta[i][k]  # reject the candidate

        # check whether xi \in F by system analysis
        gg = sign_g*g_lim(xi)
        num_eval_g += 1
        if gg > b:                                             
            # xi belongs to the failure region
            num_accepted += 1
            theta[i+1, :] = xi
            g[i+1] = gg
        else:
            # xi does not belong to the failure region
            theta[i+1, :] = theta[i, :]
            g[i+1] = g[i]

    # estimation of the acceptance-rejection rate
    # ar_rate = N/(num_eval_g + 1) # N = num_eval_g + 1 always
    ar_rate = num_accepted/N

    # return theta, its corresponding g, the number of evaluations of g_lim,
    # and the acceptance-rejection rate
    return theta, sign_g*g, num_eval_g, ar_rate


# Subset simulation
def subsim(pi, pi_marginal, g_lim, p0=0.1, N=1000, spread=1, g_failure='>=0'):
    ''' Subset simulation
    Usage:
        pf = subsim(pi, pi_marginal, g_lim, N=1000, p0=0.1)

    Input parameters:
        pi:            JPDF of theta
        pi_marginal:   list with the marginal PDFs associated to the PDF pi
        g_lim:         limit state function g(x)
        N:   number of samples per conditional level, default N=1000
        p0:  conditional failure probability p0 in [0.1, 0.3], default p0=0.1
        g_failure:     definition of the failure region; the possible options
                       are: >=0' (default) and '<=0'

    Output parameters:
        theta:  list of samples for each intermediate failure level
        g:      list of the evaluations in g_lim of the samples of theta
        b:      intermediate failure thresholds
        ar:    acceptance-rejection rate for each level
        pf:     probability of failure
    '''
    if g_failure == '>=0':
        sign_g = 1
    elif g_failure == '<=0':
        sign_g = -1
    else:
        raise Exception("g_failure must be either '>=0' or '<=0'.")

    d = len(pi_marginal)  # number of dimensions of the random variable theta

    Nc = int(N*p0)  # number of Markov chains (number of seeds per level)
    Ns = int(1/p0)  # number of samples per Markov chain, including the seed

    if not (np.isclose(Nc, N*p0) and np.isclose(Ns, 1/p0)):
        raise Exception("Please choose p0 so that N*p0 and 1/p0 are natural "
                        "numbers")

    # initialization of some lists
    N_F   = []  # N_F[j] contains the number of failure samples at level j
    theta = []  # list of numpy arrays which contains the samples at each level j
    g     = []  # list of numpy arrays which contains the evaluation of each set of samples at each level j
    ar    = []  # acceptance-rejection rate at each level j
    b     = []  # intermediate threshold values

    # crude Monte Carlo in region F[0]
    j = 0       # number of conditional level

    # draw N i.i.d. samples from pi = pi(.|F0) using MCS
    theta.append(None)
    theta[0] = pi.rvs(N)
    if d == 1:
        theta[0] = theta[0][:, np.newaxis]

    # evaluate the limit state function of those N samples
    g.append(None)
    g[0] = np.empty(N)
    for i in range(N):
        g[0][i] = sign_g*g_lim(theta[0][i, :])

    # count the number of samples in level F[0]
    N_F.append(None)
    N_F[0] = np.sum(g[0] > 0)  # b = 0

    # main loop
    while N_F[j]/N < p0:  # if N_F[j] < Nc
        # sort the limit state values in ascending order
        idx = np.argsort(g[j]) # index of the sorting
        g_sorted = g[j][idx]   # np.sort(g) -> sort the points using the idx key

        # estimate the p0-percentile of g https://www.anaconda.com/distribution/
        b.append(None)
        b[j] = (g_sorted[N-Nc-1] + g_sorted[N-Nc])/2

        # select the seeds: they are the last Nc samples associated to idx
        seed  = theta[j][idx[-Nc:], :]
        gseed = g[j][idx[-Nc:]]

        # starting from seed[k,:] draw Ns-1 additional samples from pi(.|Fj)
        # using a MCMC algorithm called MMA
        theta_from_seed = Nc*[None]
        g_from_seed     = Nc*[None]
        num_eval_g      = Nc*[None]
        ar_rate         = Nc*[None]
        for k in range(Nc): # Nc = N*p0
            theta_from_seed[k], g_from_seed[k], num_eval_g[k], ar_rate[k] = \
                MMA(seed[k,:], gseed[k], Ns, g_lim, b[j], pi_marginal, spread,
                    g_failure)

        # concatenate all samples theta_from_seed[k] in a single array theta
        theta.append(None);  theta[j+1] = np.vstack(theta_from_seed)
        g.append(None);      g[j+1]     = sign_g*np.concatenate(g_from_seed)
        ar.append(ar_rate)

        # count the number of samples in level F[j+1]
        N_F.append(None)
        N_F[j+1] = np.sum(g[j+1] > 0) # b = 0

        # continue with the next intermediate failure level
        j += 1

    # estimate the probability of failure and report it
    pf = p0**j * N_F[j]/N

    # change of sign for g
    if g_failure == '<=0':  # sign_g == -1
        for i in range(len(g)):   g[i] = -g[i]
        for i in range(len(g)-1): b[i] = -b[i]

    return theta, g, b, ar, pf

def subsim_curve(g, b, p0, demand=None, g_failure='>=0'):
    ''' Estimates the curve demand vs Pf(demand)
    Usage:
        pf_demand = subsim_curve(g, b, p0, demand)

    Input parameters:
        g:      list of the evaluations in g_lim of the samples of theta
        b:      intermediate failure thresholds
        p0:     conditional failure probability p0 in [0.1, 0.3]
        demand: levels of demand. If demand is None, then this vector is
                estimated in the code, otherwise the one provided is employed

    Output parameters:
        demand:     levels of demand
        pf_demand:  probability of failure for each demand level
    '''

    # levels of demand
    if demand is None:
        num_demand = 200   # number of levels of demand
        if g_failure == '>=0':
            demand = np.linspace(min(g[0]), 0, num_demand)
        elif g_failure == '<=0':
            demand = np.linspace(0, max(g[0]), num_demand)
        else:
            raise Exception("g_failure must be either '>=0' or '<=0'.")
    else:
        num_demand = len(demand) # number of levels of demand

    m = len(b)                        # number of intermediate failure domains
    pf_demand = np.zeros(num_demand)  # pf of the levels of demand

    if g_failure == '>=0':
        k = 0 # counter for the levels of demand k = 0, 1, ..., num_demand-1

        # for each intermediate failure level j = 0, 1, ..., m-1
        for j in range(m):
           while b[j] > demand[k]:
              pf_demand[k] = p0**j * np.mean(g[j] > demand[k])
              k += 1

        # now in the last level of demand j = m
        while k < num_demand:
            pf_demand[k] = p0**m * np.mean(g[m] > demand[k])
            k += 1
    else:
        k = num_demand - 1 # counter for the levels of demand
                           # k = num_demand-1, num_demand-2, ..., 0

        # for each intermediate failure level j = 0, 1, ..., m-1
        for j in range(m):
           while b[j] < demand[k]:
              pf_demand[k] = p0**j * np.mean(g[j] < demand[k])
              k -= 1

        # now in the last level of demand j = m
        while k >= 0:
            pf_demand[k] = p0**m * np.mean(g[m] < demand[k])
            k -= 1

    return demand, pf_demand
