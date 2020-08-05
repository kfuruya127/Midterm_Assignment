# requirements
import numpy as np
import matplotlib.pyplot as plt
#import cvxpy as cv

# dataset 4
n = 200
x_d4 = 3 * (np.random.rand(n, 4) - 0.5)
y_d4 = (2 * x_d4[:, 0] - 1 * x_d4[:,1] + 0.5 + 0.5 * np.random.randn(n)) > 0
y_d4 = 2 * y_d4 -1

## Newton method
x_d4 = np.append(x_d4,np.ones((n,1)),axis=1)
it = range(1,301)
y_d4 = np.reshape(y_d4,(n,1))
w = np.random.rand(len(x_d4[0]),1)
lam = 0.01
ll_history = []
ww_history = []
alpha = 1

for t in it:
    pos = 1 / (1+np.exp(-y_d4*x_d4@w))
    dire = np.sum(((1-pos)*y_d4*x_d4),axis=0)+2*lam*w.T
    hess = np.sum((pos*(1-pos)*x_d4*x_d4),axis=0)+2*lam
    l = np.sum(np.log((1+np.exp(-y_d4*x_d4@w))))+lam*w.T@w
    ww_history.append(w)
    ll_history.append(float(l))
    w = w + (dire/hess).T * alpha / np.sqrt(t)

plt.plot(it, ll_history, 'bo-', markersize=0.5, label='Newton')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()