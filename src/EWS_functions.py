import numpy as np
import statsmodels.api as sm
import scipy.stats as st
from scipy.optimize import curve_fit


def runstd(x, w): # standard deviation xs[i] is for a window of width w around x[i]
    n = x.shape[0]
    xs = np.zeros_like(x)
    for i in range(w // 2): # for beginning edge bit, the window centre starts at half window size
        xw = x[: i + w // 2 + 1] # window always starts at 0 and more and more covers into range
        xw = xw - xw.mean() # make into variations around zero
        if np.std(xw) > 0:
            lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
            p0 = lg[0]
            p1 = lg[1]
            xw = xw - p0 * np.arange(xw.shape[0]) - p1 # remove linear trend

            xs[i] = np.std(xw) # calculate standard deviation
        else:
            xs[i] = np.nan
    for i in range(n - w // 2, n):
        xw = x[i - w // 2 + 1:] # for end bit window always ends at 0 and covers less and less
        xw = xw - xw.mean()
        if np.std(xw) > 0:
            lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
            p0 = lg[0]
            p1 = lg[1]

            xw = xw - p0 * np.arange(xw.shape[0]) - p1
            xs[i] = np.std(xw)
        else:
            xs[i] = np.nan

    for i in range(w // 2, n - w // 2):
        xw = x[i - w // 2 : i + w // 2 + 1] # standard window
        xw = xw - xw.mean()
        if np.std(xw) > 0:
            lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
            p0 = lg[0]
            p1 = lg[1]
            xw = xw - p0 * np.arange(xw.shape[0]) - p1

        xs[i] = np.std(xw)
    else:
        xs[i] = np.nan

    return xs

def runac(x, w):
    n = x.shape[0]
    xs = np.zeros_like(x)
    for i in range(w // 2):
        xw = x[: i + w // 2 + 1]
        xw = xw - xw.mean()
        if np.std(xw) > 0:
            lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
            p0 = lg[0]
            p1 = lg[1]
            xw = xw - p0 * np.arange(xw.shape[0]) - p1 # remove linear trend
            # within a window of data,  take from 1 to end and from 0 to -1 and find correlation
            xs[i] = np.corrcoef(xw[1:], xw[:-1])[0,1] # Pearson's R correlation coefficient
        else:
            xs[i] = np.nan

    for i in range(n - w // 2, n):
        xw = x[i - w // 2 + 1:]

        xw = xw - xw.mean()
        if np.std(xw) > 0:
            lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
            p0 = lg[0]
            p1 = lg[1]

            xw = xw - p0 * np.arange(xw.shape[0]) - p1

            xs[i] = np.corrcoef(xw[1:], xw[:-1])[0,1]
        else:
            xs[i] = np.nan

    for i in range(w // 2, n - w // 2):
        xw = x[i - w // 2 : i + w // 2 + 1]

        xw = xw - xw.mean()
        if np.std(xw) > 0:

            lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
            p0 = lg[0]
            p1 = lg[1]

            xw = xw - p0 * np.arange(xw.shape[0]) - p1

            xs[i] = np.corrcoef(xw[1:], xw[:-1])[0,1]
        else:
            xs[i] = np.nan

    return xs

def run_fit_a_ar1(x, w):
    n = x.shape[0]
    xs = np.zeros_like(x)

    for i in range(w // 2):
        xs[i] = np.nan

    for i in range(n - w // 2, n):
        xs[i] = np.nan

    for i in range(w // 2, n - w // 2):
        xw = x[i - w // 2 : i + w // 2 + 1]
        xw = xw - xw.mean() # variations in the window

        # p0, p1 = np.polyfit(np.arange(xw.shape[0]), xw, 1)
        p0, p1, r, p, se = st.linregress(np.arange(xw.shape[0]),xw)
        xw = xw - p0 * np.arange(xw.shape[0]) - p1 # remove linear trend


        dxw = xw[1:] - xw[:-1] # each element is x[i+1]-x[i] (same as np.diff(xw))

        xw = sm.add_constant(xw)
        model = sm.GLSAR(dxw, xw[:-1], rho=1)
        results = model.iterative_fit(maxiter=10)

        a = results.params[1]

        xs[i] = a
    return xs

def fourrier_surrogates(ts, ns):
    ts_fourier  = np.fft.rfft(ts)
    random_phases = np.exp(np.random.uniform(0, 2 * np.pi, (ns, ts.shape[0] // 2 + 1)) * 1.0j)
    # random_phases = np.exp(np.random.uniform(0, 2 * np.pi, (ns, ts.shape[0])) * 1.0j)
    ts_fourier_new = ts_fourier * random_phases
    new_ts = np.real(np.fft.irfft(ts_fourier_new))
    return new_ts

def kendall_tau_test(ts, ns, tau, mode1 = 'fourier', mode2 = 'linear'):
    tlen = ts.shape[0]

    if mode1 == 'fourier':
        tsf = ts - ts.mean()
        nts = fourrier_surrogates(tsf, ns) # create ns fourier surrogate ts
    elif mode1 == 'shuffle':
        nts = shuffle_surrogates(ts, ns)
    stat = np.zeros(ns)
    tlen = nts.shape[1]
    if mode2 == 'linear':
        for i in range(ns):
            stat[i] = st.linregress(np.arange(tlen), nts[i])[0]
    elif mode2 == 'kt':
        for i in range(ns):
            stat[i] = st.kendalltau(np.arange(tlen), nts[i])[0]

    p = 1 - st.percentileofscore(stat, tau) / 100.
    return p
