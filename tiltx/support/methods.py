"""
Created on Thu May 16 18:53:46 2019

@author: seslija
"""
import numpy
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import integrate



def detect_cusum(x, threshold=1, drift=0, ending=False):
    """Cumulative sum algorithm (CUSUM) to detect abrupt changes in data.

    Parameters
    ----------
    x : 1D array_like
        data.
    threshold : positive number, optional (default = 1)
        amplitude threshold for the change in the data.
    drift : positive number, optional (default = 0)
        drift term that prevents any change in the absence of change.
    ending : bool, optional (default = False)
        True (1) to estimate when the change ends; False (0) otherwise.
    
    Returns
    -------
    ta : 1D array_like [indi, indf], int
        alarm time (index of when the change was detected).
    tai : 1D array_like, int
        index of when the change started.
    taf : 1D array_like, int
        index of when the change ended (if `ending` is True).
    amp : 1D array_like, float
        amplitude of changes (if `ending` is True).

    Notes
    -----
    Tuning of the CUSUM algorithm according to Gustafsson (2000)[1]_:
    Start with a very large `threshold`.
    Choose `drift` to one half of the expected change, or adjust `drift` such
    that `g` = 0 more than 50% of the time.
    Then set the `threshold` so the required number of false alarms (this can
    be done automatically) or delay for detection is obtained.
    If faster detection is sought, try to decrease `drift`.
    If fewer false alarms are wanted, try to increase `drift`.
    If there is a subset of the change times that does not make sense,
    try to increase `drift`.

    Note that by default repeated sequential changes, i.e., changes that have
    the same beginning (`tai`) are not deleted because the changes were
    detected by the alarm (`ta`) at different instants. This is how the
    classical CUSUM algorithm operates.

    If you want to delete the repeated sequential changes and keep only the
    beginning of the first sequential change, set the parameter `ending` to
    True. In this case, the index of the ending of the change (`taf`) and the
    amplitude of the change (or of the total amplitude for a repeated
    sequential change) are calculated and only the first change of the repeated
    sequential changes is kept. In this case, it is likely that `ta`, `tai`,
    and `taf` will have less values than when `ending` was set to False.
    
    The cumsum algorithm is taken from (with minor modifications):
    https://github.com/BMClab/BMC/blob/master/functions/detect_cusum.py
    
    Duarte, M., Watanabe, R.N. (2018) Notes on Scientific Computing for 
    Biomechanics and Motor Control. GitHub repository, https://github.com/bmclab/BMC

    """
    
    x = numpy.atleast_1d(x).astype('float64')
    gp, gn = numpy.zeros(x.size), numpy.zeros(x.size)
    ta, tai, taf = numpy.array([[], [], []], dtype=int)
    tap, tan = 0, 0
    amp = numpy.array([])
    a = -0.0001
    # Find changes (online form)
    for i in range(1, x.size):
        s = x[i] - x[i-1]
        gp[i] = gp[i-1] + s - drift  # cumulative sum for + change
        gn[i] = gn[i-1] - s - drift  # cumulative sum for - change
        if gp[i] < a:
            gp[i], tap = 0, i
        if gn[i] < a:
            gn[i], tan = 0, i
        if gp[i] > threshold or gn[i] > threshold:  # change detected!
            ta = numpy.append(ta, i)    # alarm index
            tai = numpy.append(tai, tap if gp[i] > threshold else tan)  # start
            gp[i], gn[i] = 0, 0      # reset alarm
    # THE CLASSICAL CUSUM ALGORITHM ENDS HERE

    # Estimation of when the change ends (offline form)
    if tai.size and ending:
        _, tai2, _, _ = detect_cusum(x[::-1], threshold, drift)
        taf = x.size - tai2[::-1] - 1
        # Eliminate repeated changes, changes that have the same beginning
        tai, ind = numpy.unique(tai, return_index=True)
        ta = ta[ind]
        # taf = numpy.unique(taf, return_index=False)  # corect later
        if tai.size != taf.size:
            if tai.size < taf.size:
                taf = taf[[numpy.argmax(taf >= i) for i in ta]]
            else:
                ind = [numpy.argmax(i >= ta[::-1])-1 for i in taf]
                ta = ta[ind]
                tai = tai[ind]
        # Delete intercalated changes (the ending of the change is after
        # the beginning of the next change)
        ind = taf[:-1] - tai[1:] > 0
        if ind.any():
            ta = ta[~numpy.append(False, ind)]
            tai = tai[~numpy.append(False, ind)]
            taf = taf[~numpy.append(ind, False)]
        # Amplitude of changes
        amp = x[taf] - x[tai]

    return ta, tai, taf, amp



def last_stationary_point(y,t):
    """
    Finds stationary points in the signal.
    
    :param y: the array representing a time series
    :type y: list or array
    :param t: the array representing time component
    :type t: list or array
    """

    finite_difference_1 = numpy.gradient(y, t)
    
    is_peak = [finite_difference_1[i] * finite_difference_1[i + 1] 
               <= -0*0.0001 for i in range(len(finite_difference_1) - 1)]

    peak_indices = [i for i, b in enumerate(is_peak) if b]

    if len(peak_indices) == 0:
        return [],[]
        
    return peak_indices[-1]



def CUMSUM_flip(y,t):
    """ 
    Iterative cumsum algorithm for change point detection in the
    time series y. The algorithm normalizes the input gradient and feeds
    it to the standard cumsum algorithm with decreasing treshold. The 
    algorithm does not stop untill the change point is detected. If no
    change point is detected, it uses a last stationary point as the output.
    If there is no statinary point, the algorithm gives the last point
    of y as the output.
    
    :param y: the array representing a time series
    :type y: list or array
    :param t: the array representing time component
    :type t: list or array
    """
	# compute gradinet of y
    grad = numpy.gradient(y, t)
    
    # if time series shortar than 12 samples, report the last index
    if (t[-1]-t[0])< 0.120:
        return len(y)-1
    # grad_f is filtered gradient
    # l_filter = 5
    # grad_f = savgol_filter(grad, l_filter, 3)
    # no filtration applied
    grad_f = grad
    # normalize the input to repeated cumsum
    grad_norm = (grad_f-grad_f.min())/(grad_f.max()-grad_f.min())

    for k in range(0,15):
        ta_k, tai_k, taf_k, amp_k = detect_cusum(
            numpy.flip(grad_norm), 0.85-0.05*k, 0.01, True)
        if len(taf_k) > 0:
            taf = taf_k[0]
            ind = len(grad_norm) - int(taf)
            if t[taf] - t[ind] < 120:
                ta_k_aux, tai_k_aux, taf_k_aux, amp_k_aux = detect_cusum(
                    numpy.flip(grad_norm)[taf_k[0]:], 0.85-0.05*k, 0.005, True)
                if len(taf_k_aux) > 0:
                    taf = taf_k_aux[0]
                    ind = len(grad_norm) - int(taf)-taf_k[0]
            # 170 ms as an exclusion treshold
            if t[ind]-t[0] > 0.170:
                return ind
            else:
                # could not find a reaction point
                stationary_points = last_stationary_point(y,t)
                if len(stationary_points[0]) > 0:
                    stationary_index, _ = stationary_points
                    if stationary_index[-1] >= 12:
                        return stationary_index[-1]
                    else:
                        return len(y)-1
                else:
                    return len(y)-1


def number_of_flips(y,t):
    """
    Finds the number of flips in a signal y defined on t.
    That is, the function computes a number of turning points in 
    the time series y.
    
    :param y: the array representing a time series
    :type y: list or array
    :param t: the array representing time component
    :type t: list or array
    """
    finite_difference_1 = numpy.gradient(y, t)
    
    is_peak = [finite_difference_1[i] * finite_difference_1[i + 1] <= 0 
               for i in range(len(finite_difference_1) - 1)]

    peak_indices = [i for i, b in enumerate(is_peak) if b]

    if len(peak_indices) == 0:
        return 0
        
    return len(peak_indices)
                
                

def SampEn(U, m, r):
    """
    This function calculates the sample entropy of the time series U. 
    For a given embedding dimension m, tolerance r and number of data points N, 
    SampEn is the negative logarithm of the probability that if two sets of
    simultaneous data points of length m have distance smaller than r then 
    two sets of simultaneous data points of length m+1 also have smaller than r.
    
    For more information see, e.g., https://en.wikipedia.org/wiki/Sample_entropy
    
    :param U: the array representing a time series
    :type U: list or array
    :param m: embedding dimension
    :type m: integer
    :param r: tolerance
    :type r: float
    """
    def _maxdist(x_i, x_j):
        result = max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        return result

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = 1.*numpy.array([len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))])
        return sum(C)

    N = len(U)
    
    return -numpy.log(_phi(m+1) / _phi(m))



def normalize(z):
    """
    Normalize an array.
    
    :param z: The array to normalize
    :type z: array
    """
    return (z - min(z)) / (max(z) - min(z))
