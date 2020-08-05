# requirements
import numpy as np
import matplotlib.pyplot as plt
import copy
#import cvxpy as cv

# dataset 4
n = 200
x_d4 = 3 * (np.random.rand(n, 4) - 0.5)
y_d4 = (2 * x_d4[:, 0] - 1 * x_d4[:,1] + 0.5 + 0.5 * np.random.randn(n)) > 0
y_d4 = 2 * y_d4 -1

x_d4 = np.append(x_d4,np.ones((n,1)),axis=1)
it = range(1,301)
y_d4 = np.reshape(y_d4,(n,1))
w_GD = np.random.rand(len(x_d4[0]),1)
w_Newton = copy.deepcopy(w_GD)
lam = 0.01
ll_history_GD = []
ll_history_Newton = []
ww_history_GD = []
ww_history_Newton = []
alpha = 1

for t in it:
    #GD
    pos = 1 / (1+np.exp(-y_d4*x_d4@w_GD))
    dire = np.sum(((1-pos)*y_d4*x_d4),axis=0)+2*lam*w_GD.T
    l = np.sum(np.log((1+np.exp(-y_d4*x_d4@w_GD))))+lam*w_GD.T@w_GD
    ww_history_GD.append(w_GD)
    ll_history_GD.append(float(l))
    w_GD = w_GD + alpha / np.sqrt(t) * dire.T

    #Newton
    pos = 1 / (1+np.exp(-y_d4*x_d4@w_Newton))
    dire = np.sum(((1-pos)*y_d4*x_d4),axis=0)+2*lam*w_Newton.T
    hess = np.sum((pos*(1-pos)*x_d4*x_d4),axis=0)+2*lam
    l = np.sum(np.log((1+np.exp(-y_d4*x_d4@w_Newton))))+lam*w_Newton.T@w_Newton
    ww_history_Newton.append(w_Newton)
    ll_history_Newton.append(float(l))
    w_Newton = w_Newton + alpha / np.sqrt(t) * (dire/hess).T

l_min = min(min(ll_history_GD),min(ll_history_Newton))
ll_history_GD = np.array(ll_history_GD)
ll_history_Newton = np.array(ll_history_Newton)

plt.plot(np.abs(ll_history_GD[:50] - l_min), 'bo-', linewidth=0.5, markersize=1, label='SteepestGD')
plt.plot(np.abs(ll_history_Newton[:50] - l_min), 'ro-', linewidth=0.5, markersize=1, label='Newton')
plt.legend()
plt.yscale('log')
plt.xlabel('iter')
plt.ylabel('diff')
plt.show()