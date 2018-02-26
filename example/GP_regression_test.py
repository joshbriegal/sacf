import george
from george import kernels
from george.modeling import Model
from GACF import *
from GACF.datastructure import DataStructure
import numpy as np
import matplotlib.pyplot as plt
import emcee as mc
import corner
from scipy.optimize import minimize


def get_data_from_file(filename):
    ds = DataStructure(filename)
    return ds.timeseries(), ds.values()


class TwoSinModel(Model):
    parameter_names = ('A1', 'P1', 'A2', 'P2')

    def get_value(self, t):
        t = t.flatten()
        return 1. + self.A1 * np.sin((2 * np.pi / self.P1) * t) + self.A2 * np.sin((2 * np.pi / self.P2) * t)

def model_fit_method():

    noise_level = 0.01

    time_series, data = get_data_from_file(
        '/Users/joshbriegal/GitHub/GACF/example/files/NG0522-2518_025974_LC_tbin=10min.dat')
    data = np.array(data)
    time_series = np.array(time_series)

    y_err = np.linspace(noise_level, noise_level, len(time_series))

    two_sin_model = george.GP(mean=TwoSinModel(A1=1.0, P1=11.0, A2=1.0, P2=22.0))
    two_sin_model.compute(time_series, y_err)

    def lnprob(p):
        two_sin_model.set_parameter_vector(p)
        return two_sin_model.log_likelihood(data) + two_sin_model.log_prior()

    initial = two_sin_model.get_parameter_vector()
    ndim, nwalkers = len(initial), 32
    p0 = initial + 1. * (10. ** -8.) * np.random.randn(nwalkers, ndim)
    sampler = mc.EnsembleSampler(nwalkers, ndim, lnprob)

    print "Running burn in"
    p0, _, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()

    print 'Running production'
    sampler.run_mcmc(p0, 1000)

    # plt.errorbar(time_series, data, y_err, fmt='.k', capsize=0, elinewidth=0.1, markersize=0.1)
    plt.scatter(time_series, data, s=0.1, c='k')
    x = np.linspace(time_series[0], time_series[-1], 1000)

    samples = sampler.flatchain
    for s in samples[np.random.randint(len(samples), size=24)]:
        two_sin_model.set_parameter_vector(s)
        plt.plot(x, two_sin_model.mean.get_value(x), c="#4682b4", alpha=0.3)

    plt.show()

    tri_cols = ['A1', 'P1', 'A2', 'P2']
    tri_labels = ['A1', 'P1', 'A2', 'P2']

    names = two_sin_model.get_parameter_names()
    # inds = np.array([names.index("mean:"+k) for k in tri_cols])
    corner.corner(sampler.chain, labels=tri_labels)


if __name__ == '__main__':
    noise_level = 0.1
    gamma1 = 1.0
    gamma2 = 1.0
    P1 = 22.0
    P2 = 22.0

    time_series, data = get_data_from_file(
        '/Users/joshbriegal/GitHub/GACF/example/files/NG0522-2518_025974_LC_tbin=10min.dat')
    data = np.array(data)
    time_series = np.array(time_series)
    y_err = np.linspace(noise_level, noise_level, len(time_series))

    sin_1_kernel = kernels.ExpSine2Kernel(gamma=gamma1, log_period=np.log(P1))
    sin_2_kernel = kernels.ExpSine2Kernel(gamma=gamma2, log_period=np.log(P2))

    kernel = sin_2_kernel + sin_1_kernel

    gp = george.GP(kernel, mean=np.mean(data),
                   white_noise=np.log(noise_level), fit_white_noise=False, fit_mean=True)

    gp.compute(time_series, y_err)

    x_pred = x = np.linspace(time_series[0], time_series[-1], 1000)

    pred, pred_var = gp.predict(data, x_pred, return_var=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.fill_between(x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var), color='b', alpha=0.5)
    ax.plot(x_pred, pred, color='r', lw=1.5, alpha=0.5)
    ax.scatter(time_series, data, s=0.1, color='k')

    plt.show()
    fig.savefig('GP_pre.pdf')
    plt.close()

    print 'Initial ln-likelihood: {0:.2f}'.format(gp.log_likelihood(data))


    def neg_ln_like(p):
        if np.any((p[1:] < 0) + (p[1:] > 100)):
            return 1e25
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(data, quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(data, quiet=True)


    result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like, method='L-BFGS-B')
    print(result)

    gp.set_parameter_vector(result.x)
    print "\nFinal ln-likelihood: {0:.2f}".format(gp.log_likelihood(data))

    print 'GP Parameters: \n', gp.get_parameter_dict()

    pred, pred_var = gp.predict(data, x_pred, return_var=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.fill_between(x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var), color='b', alpha=0.5)
    ax.plot(x_pred, pred, color='r', lw=1.5, alpha=0.5)
    ax.scatter(time_series, data, s=0.1, color='k')

    plt.show()
    fig.savefig('GP_post.pdf')
    plt.close(fig)





