"""
Uses model from Angus et al. 2017 to generate false photometric light curves for stars dominated by rotation
(a) White noise + signal
(b) Red noise + white noise + signal
(c) Red noise + white noise + quasi periodic signal
"""
import george
from george import kernels
import numpy as np
import matplotlib.pyplot as plt
# import emcee as mc
# import corner
from scipy.optimize import minimize

if __name__ == '__main__':

    signal_amplitude = np.exp(-13)
    white_noise_amplitude = np.exp(-17)
    red_noise_amplitude = np.exp(-17)
    tot_amp = signal_amplitude + red_noise_amplitude + white_noise_amplitude
    l = np.exp(7.2)
    gamma1 = np.exp(-2.3)
    gamma2 = np.exp(-2.3)
    P1 = 22
    P2 = 0.8
    data_mean = 1.0

    # time_series = np.loadtxt('irregular_sampling.txt', dtype=np.float64, skiprows=1)
    time_series = np.linspace(0, 1000, 10000)

    periodic_kernel = kernels.CosineKernel(log_period=np.log(P1))
    red_noise_kernel = kernels.ExpSine2Kernel(gamma=gamma2, log_period=P2)
    quasi_periodic_kernel = kernels.ExpSquaredKernel(l) * kernels.ExpSine2Kernel(gamma=gamma1, log_period=np.log(P1))

    kernel1 = signal_amplitude * periodic_kernel
    kernel2 = (signal_amplitude / tot_amp) * periodic_kernel + (red_noise_amplitude / tot_amp) * red_noise_kernel
    kernel3 = (
                      signal_amplitude + red_noise_amplitude / tot_amp) * quasi_periodic_kernel  # + (red_noise_amplitude / tot_amp) * red_noise_kernel

    list_of_gps = []

    gp1 = george.GP(kernel1, mean=data_mean, white_noise=np.log(white_noise_amplitude))
    gp2 = george.GP(kernel2, mean=data_mean, white_noise=np.log(white_noise_amplitude))
    gp3 = george.GP(kernel3, mean=data_mean, white_noise=np.log(white_noise_amplitude))

    list_of_gps = [gp1, gp2, gp3]

    for gp in list_of_gps:
        gp.compute(time_series, yerr=white_noise_amplitude)

    predictions = []

    for gp in list_of_gps:
        data = gp.sample(time_series)
        norm_data = data / np.mean(data)
        data_with_noise = norm_data + np.random.randn(len(time_series)) * (white_noise_amplitude / signal_amplitude)
        predictions.append(data_with_noise)

    fig, axs = plt.subplots(3, 1, sharex=True)
    axs = list(axs)

    for p, ax in zip(predictions, axs):
        ax.scatter(time_series, p, s=0.1)

    plt.show()
