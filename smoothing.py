import numpy as np

if np.__version__ != '1.26.2':
    print(f'This script was created with numpy version 1.22.1, but {np.__version__} used. Be careful.')
from numpy.polynomial.polynomial import Polynomial
from itertools import combinations
from scipy.stats import theilslopes
from collections import deque

"""q = deque([1, 2, 3, 4, 5], 5)
   q
   deque([1, 2, 3, 4, 5], maxlen=5)
   q.append(6)
   q
   deque([2, 3, 4, 5, 6], maxlen=5)
"""
from time import time


def sorted(x, y):
    """Returns the data, sorted by x-vector.
    
    Parameters
    ----------
    x, y : array-like
           Data points.
    
    Returns
    -------
    (x, y, i) : tuple
                Sorted version of data points and the indexes.
    """
    i = np.argsort(x)  # indexes of elements of `x` in the sorted order
    return x[i], y[i], i


def extended(x, y, w):
    """Returns extended (aka `tailed`) input arrays (by simple mirroring).
    
    Parameters
    ----------
    x, y : array-like
           Data points.
    w : int
        Radius of extension. The resulting arrays will contain (len(x) + 2*w) elements.
        
    Returns
    -------
    (x, y) : tuple
             Extended version of the data points.
    """
    tx = np.hstack((2 * x[0] - x[1:w + 1][::-1], x, 2 * x[-1] - x[-w - 1:-1][::-1]))
    ty = np.hstack((2 * y[0] - y[1:w + 1][::-1], y, 2 * y[-1] - y[-w - 1:-1][::-1]))
    return tx, ty


def smSG(x, y, w, n):
    """General Savitzky-Golay Filter.
    
    Parameters
    ----------
    x, y : array-like
           Coordinates of data points. Lengths of x and y must be equal.
    w : int
        Radius of the filter window. The window size is 2*w+1.
    n : int
        Order of the smoothing polynomial.
        
    Returns
    -------
    out : array-like
          Smoothed version of y coordinates.
    """
    tx, ty = extended(x, y, w)
    sy = np.zeros_like(y)  # container for the smoothed y value
    for i in range(w, len(ty) - w):  # index of the window central point
        wx = tx[i - w: i + w + 1]  # x coordinates inside the window
        wy = ty[i - w: i + w + 1]  # y coordinates inside the window
        p = Polynomial.fit(wx, wy, n)
        sy[i - w] = p(wx[w])  # smoothed value
    return sy


def smSG_bisquare(x, y, w, n, extend=True):
    """Savitzky-Golay Filter with bisquare weighting.
    
    Parameters
    ----------
    x, y : array-like
           Coordinates of data points. Lengths of x and y must be equal.
    w : int
        Radius of the filter window. The window size is 2*w+1.
    n : int
        Order of the smoothing polynomial.
        
    Returns
    -------
    out : array-like
          Smoothed version of y coordinates.
    """
    if extend:
        tx, ty = extended(x, y, w)  # extending input array
    else:
        tx, ty = x, y
    sx = np.zeros(len(tx) - 2*w)  # container for the output abscissas
    sy = np.zeros(len(ty) - 2*w)  # container for the smoothed y value
    for i in range(w, len(ty) - w):  # index of the window central point
        wx = tx[i - w: i + w + 1]  # x coordinates inside the window
        wy = ty[i - w: i + w + 1]  # y coordinates inside the window
        p = Polynomial.fit(wx, wy, n)  # unweighted fit
        for j in range(2):  # double weighting
            r = np.abs(p(wx) - wy)  # absolute residuals
            m = np.median(np.abs(r - np.median(r)))  # median absolute deviation of residuals
            W = (1 - (r / (6 * m)) ** 2) ** 2  # weights
            W[r > (6 * m)] = 0
            p = Polynomial.fit(wx, wy, n, w=W)
        sx[i - w] = wx[w]
        sy[i - w] = p(wx[w])  # smoothed value
    return sx, sy


def smSGder(x, y, w, n):
    """General Savitzky-Golay Filter with first derivative calculation.
    
    Parameters
    ----------
    x, y : array-like
           Coordinates of data points. Lengths of x and y must be equal.
    w : int
        Radius of the filter window. The window size is 2*w+1.
    n : int
        Order of the smoothing polynomial.
        
    Returns
    -------
    out : tuple(array-like, array-like)
          Smoothed version of y coordinates plus derivative.
    """
    tx, ty = extended(x, y, w)
    sy = np.zeros_like(y)  # container for the smoothed y value
    der = np.zeros_like(y)  # container for the derivalive values
    for i in range(w, len(ty) - w):  # index of the window central point
        wx = tx[i - w: i + w + 1]  # x coordinates inside the window
        wy = ty[i - w: i + w + 1]  # y coordinates inside the window
        p = np.polynomial.polynomial.Polynomial.fit(wx, wy, n)
        sy[i - w] = p(wx[w])  # smoothed value
        der[i - w] = p.deriv(1)(wx[w])  # derivative
    return sy, der


def smSGder_bisquare(x, y, w, n):
    """Savitzky-Golay Filter with bisquare weighting and first derivative calculation.
    
    Parameters
    ----------
    x, y : array-like
           Coordinates of data points. Lengths of x and y must be equal.
    w : int
        Radius of the filter window. The window size is 2*w+1.
    n : int
        Order of the smoothing polynomial.
        
    Returns
    -------
    out : tuple(array-like, array-like)
          Smoothed version of y coordinates plus derivative.
    """
    time_start = time()
    tx, ty = extended(x, y, w)  # extending input array
    sy = np.zeros_like(y)  # container for the smoothed y value
    der = np.zeros_like(y)  # container for the derivalive values
    for i in range(w, len(ty) - w):  # index of the window central point
        wx = tx[i - w: i + w + 1]  # x coordinates inside the window
        # print('{:.2f} K'.format(1/wx[-1]/8.6173303e-5 - 1/wx[0]/8.6173303e-5))
        wy = ty[i - w: i + w + 1]  # y coordinates inside the window
        p = Polynomial.fit(wx, wy, n)  # unweighted fit
        for j in range(2):  # double weighting
            r = np.abs(p(wx) - wy)  # absolute residuals
            m = np.median(np.abs(r - np.median(r)))  # median absolute deviation of residuals
            K = 4
            W = (1 - (r / (K * m)) ** 2) ** 2  # weights
            W[r > (K * m)] = 0
            try:
                p = Polynomial.fit(wx, wy, n, w=W)
            except:
                print('error')
        sy[i - w] = p(wx[w])  # smoothed value
        der[i - w] = p.deriv(1)(wx[w])  # derivative
    time_finish = time()
    print('smSGder_bisquare(x, y, w, n) took {:.3f} s to execute on the data'.format(time_finish - time_start))
    return sy, der


def smSGtan_bisquare(x, y, w, n):
    """Savitzky-Golay Filter with bisquare weighting and tangent line calculation.
    
    Parameters
    ----------
    x, y : array-like
           Coordinates of data points. Lengths of x and y must be equal.
    w : int
        Radius of the filter window. The window size is 2*w+1.
    n : int
        Order of the smoothing polynomial.
        
    Returns
    -------
    out : tuple(array-like, array-like)
          Smoothed version of y coordinates plus derivative.
    """
    time_start = time()
    tx, ty = extended(x, y, w)  # extending input array
    sy = np.zeros_like(y)  # container for the smoothed y value
    slope = np.zeros_like(y)  # container for the slope values
    inter = np.zeros_like(y)  # container for the intercept values
    for i in range(w, len(ty) - w):  # index of the window central point
        wx = tx[i - w: i + w + 1]  # x coordinates inside the window
        # print('{:.2f} K'.format(1/wx[-1]/8.6173303e-5 - 1/wx[0]/8.6173303e-5))
        wy = ty[i - w: i + w + 1]  # y coordinates inside the window
        p = Polynomial.fit(wx, wy, n)  # unweighted fit
        for j in range(2):  # double weighting
            r = np.abs(p(wx) - wy)  # absolute residuals
            m = np.median(np.abs(r - np.median(r)))  # median absolute deviation of residuals
            W = (1 - (r / (6 * m)) ** 2) ** 2  # weights
            W[r > (6 * m)] = 0
            try:
                p = Polynomial.fit(wx, wy, n, w=W)
            except:
                print('error')
        sy[i - w] = p(wx[w])  # smoothed value
        slope[i - w] = p.deriv(1)(wx[w])  # tangent slope - 1st derivative
        inter[i - w] = sy[i - w] - slope[i - w] * wx[w]
    time_finish = time()
    print('smSGder_bisquare(x, y, w, n) took {:.3f} s to execute on the data'.format(time_finish - time_start))
    return sy, slope, inter


def smSGder_bisquare_deque(x, y, w, n):
    """Savitzky-Golay Filter with bisquare weighting and first derivative calculation.
    
    Parameters
    ----------
    x, y : array-like
           Coordinates of data points. Lengths of x and y must be equal.
    w : int
        Radius of the filter window. The window size is 2*w+1.
    n : int
        Order of the smoothing polynomial.
        
    Returns
    -------
    out : tuple(array-like, array-like)
          Smoothed version of y coordinates plus derivative.
    """
    """q = deque([1, 2, 3, 4, 5], 5)
       q
       deque([1, 2, 3, 4, 5], maxlen=5)
       q.append(6)
       q
       deque([2, 3, 4, 5, 6], maxlen=5)
    """
    time_start = time()
    tx, ty = extended(x, y, w)  # extending input array
    sy = np.zeros_like(y)  # container for the smoothed y value
    der = np.zeros_like(y)  # container for the derivalive values
    # first window
    i = w
    qx = deque(tx[i - w: i + w + 1], 2 * w + 1)  # creation of queues for x
    qy = deque(ty[i - w: i + w + 1], 2 * w + 1)  # and y values
    p = Polynomial.fit(qx, qy, n)  # unweighted fit
    for j in range(2):  # double weighting
        r = np.abs(p(np.asarray(qx)) - qy)  # absolute residuals
        m = np.median(np.abs(r - np.median(r)))  # median absolute deviation of residuals
        W = (1 - (r / (6 * m)) ** 2) ** 2  # weights
        W[r > (6 * m)] = 0  # outliers
        p = Polynomial.fit(qx, qy, n, w=W)  # weighted fit
    # other windows
    for i in range(w + 1, len(ty) - w):  # index of the window central point
        qx.append(tx[i + w])  # x coordinates inside the window
        qy.append(ty[i + w])  # y coordinates inside the window
        p = Polynomial.fit(qx, qy, n)  # unweighted fit
        for j in range(2):  # double weighting
            r = np.abs(p(np.asarray(qx)) - qy)  # absolute residuals
            m = np.median(np.abs(r - np.median(r)))  # median absolute deviation of residuals
            W = (1 - (r / (6 * m)) ** 2) ** 2  # weights
            W[r > (6 * m)] = 0
            p = Polynomial.fit(qx, qy, n, w=W)
        sy[i - w] = p(qx[w])  # smoothed value
        der[i - w] = p.deriv(1)(qx[w])  # derivative
    time_finish = time()
    print('smSGder_bisquare_deque(x, y, w, n) took {:.3f} s to execute on the data'.format(time_finish - time_start))
    return sy, der


def smSGder_robust(x, y, w, n):
    """Savitzky-Golay Filter with robust regression. Based in part on
    https://stats.stackexchange.com/questions/317777/theil-sen-estimator-for-polynomial
    
    Parameters
    ----------
    x, y : array-like
           Coordinates of data points. Lengths of x and y must be equal.
    w : int
        Radius of the filter window. The window size is 2*w+1.
    n : int
        Order of the smoothing polynomial.
        
    Returns
    -------
    out : tuple(array-like, array-like)
          Smoothed version of y coordinates plus derivative.
    """
    tx, ty = extended(x, y, w)  # extending input array
    sy = np.zeros_like(y)  # container for the smoothed y value
    der = np.zeros_like(y)  # container for the derivalive values
    for i in range(w, len(ty) - w):  # index of the window central point
        print('{}/{}'.format(i - w, len(x)))
        wx = tx[i - w: i + w + 1]  # x coordinates inside the window
        wy = ty[i - w: i + w + 1]  # y coordinates inside the window
        # Theil-Sen linear approximation for outliers detection
        res = theilslopes(wy, wx)
        r = wy - (res[1] + res[0] * wx)
        sdev = np.std(r, ddof=1)
        ii = np.abs(r) < 3 * sdev
        if len(ii) < 2 * w + 1:
            wx = wx[ii]
            wy = wy[ii]

        p = Polynomial.fit(wx, wy, n)  # unweighted fit
        for j in range(2):  # double weighting
            r = np.abs(p(wx) - wy)  # absolute residuals
            m = np.median(np.abs(r - np.median(r)))  # median absolute deviation of residuals
            W = (1 - (r / (6 * m)) ** 2) ** 2  # weights
            W[r > (6 * m)] = 0
            p = Polynomial.fit(wx, wy, n, w=W)
        sy[i - w] = p(wx[w])  # smoothed value
        der[i - w] = p.deriv(1)(wx[w])  # derivative
    return sy, der


def smSG2der(x, y, w, n):
    """General Savitzky-Golay Filter with second derivative calculation.
    
    Parameters
    ----------
    x, y : array-like
           Coordinates of data points. Lengths of x and y must be equal.
    w : int
        Radius of the filter window. The window size is 2*w+1.
    n : int
        Order of the smoothing polynomial. Should be >= 2.
        
    Returns
    -------
    out : tuple(array-like, array-like)
          Smoothed version of y coordinates plus derivative.
    """
    tx, ty = extended(x, y, w)
    sy = np.zeros_like(y)  # container for the smoothed y value
    der = np.zeros_like(y)  # container for the derivalive values
    for i in range(w, len(ty) - w):  # index of the window central point
        wx = tx[i - w: i + w + 1]  # x coordinates inside the window
        wy = ty[i - w: i + w + 1]  # y coordinates inside the window
        p = np.polynomial.polynomial.Polynomial.fit(wx, wy, n)
        sy[i - w] = p(wx[w])  # smoothed value
        der[i - w] = p.deriv(2)(wx[w])  # derivative
    return sy, der


def smSGtan(x, y, w, n):
    """General Savitzky-Golay Filter with tangent calculation.
    
    Parameters
    ----------
    x, y : array-like
           Coordinates of data points. Lengths of x and y must be equal.
    w : int
        Radius of the filter window. The window size is 2*w+1.
    n : int
        Order of the smoothing polynomial.
        
    Returns
    -------
    out : tuple(array-like, array-like, array-like)
          Smoothed version of y coordinates plus tangent slope and intercept.
    """
    tx, ty = extended(x, y, w)
    sy = np.zeros_like(y)  # container for the smoothed y value
    slope = np.zeros_like(y)  # container for the slope values
    inter = np.zeros_like(y)  # container for the slope values
    for i in range(w, len(ty) - w):  # index of the window central point
        wx = tx[i - w: i + w + 1]  # x coordinates inside the window
        wy = ty[i - w: i + w + 1]  # y coordinates inside the window
        p = np.polynomial.polynomial.Polynomial.fit(wx, wy, n)
        sy[i - w] = p(wx[w])  # smoothed value
        slope[i - w] = p.deriv(1)(wx[w])  # tangent slope - 1st derivative
        inter[i - w] = sy[i - w] - slope[i - w] * wx[w]
    return sy, slope, inter


if __name__ == '__main__':
    # model data
    X = np.linspace(1, 9, 100)
    p = np.poly1d([3, 1, 4])
    Y = p(X)
    x = X + np.random.normal(0, 0.075, len(X))
    y = Y + np.random.normal(0, 5, len(X))
    y[30] = y[30] + 100
    y[40] = y[40] - 50
    y[45] = y[45] - 100

    # sorting
    x, y, dummy = sorted(x, y)

    import matplotlib.pyplot as plt

    fix, axs = plt.subplots(2, 2, constrained_layout=True)

    axs[0, 0].set_title('Original data and noise')
    axs[0, 0].plot(x, y, '.', ms=3.0)
    axs[0, 0].plot(X, Y, lw=2.0)
    axs[0, 0].twinx().bar(x, y - p(x), 0.1, color='skyblue', alpha=0.4)

    w = 9
    axs[0, 1].set_title('Extended, aka `Tailed`, data (w = {})'.format(w))
    tx, ty = extended(x, y, w)
    axs[0, 1].plot(x, y, '.', ms=3.0, zorder=2)
    axs[0, 1].plot(tx, ty, '.', ms=3.0, zorder=1)

    n = 2
    axs[1, 0].set_title('Smoothing result (n = {})'.format(n))
    axs[1, 0].plot(x, y, '.', ms=3.0)
    sy = smSG(x, y, w, n)
    axs[1, 0].plot(x, sy, 'r')
    tx = axs[1, 0].twinx()
    tx.bar(x, y - sy, 0.1, color='skyblue', alpha=0.4)
    tx.bar(x, p(x) - sy, 0.1, color='green', alpha=0.25)

    axs[1, 1].set_title('Bisquare result (n = {})'.format(n))
    axs[1, 1].plot(x, y, '.', ms=3.0)
    _, sy = smSG_bisquare(x, y, w, n)
    axs[1, 1].plot(x, sy)
    tx = axs[1, 1].twinx()
    tx.bar(x, y - sy, 0.1, color='skyblue', alpha=0.4)
    tx.bar(x, p(x) - sy, 0.1, color='green', alpha=0.25)
    plt.show()
