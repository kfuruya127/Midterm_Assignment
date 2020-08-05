# requirements
import numpy as np
import matplotlib.pyplot as plt
import copy

# dataset 5
n = 200
x_d5 = 3 * (np.random.rand(n, 4) - 0.5)
W = np.array([[ 2,  -1, 0.5,],
              [-3,   2,   1,],
              [ 1,   2,   3]])
y_d5 = np.argmax(np.dot(np.hstack([x_d5[:,:2], np.ones((n, 1))]), W.T)
                        + 0.5 * np.random.randn(n, 3), axis=1)

x_d5 = np.append(x_d5,np.ones((n,1)),axis=1)
c = max(y_d5) + 1
it = range(1,301)
w_GD = np.random.rand(c,len(x_d5[0]))
w_Newton = copy.deepcopy(w_GD)
ll_history_GD = []
ll_history_Newton = []
ww_history_GD = []
ww_history_Newton = []
alpha = 1

#one-hot
y = np.zeros((n,c))
for i in range(0,n):
    y[i,y_d5[i]] = 1

for t in it:
    #GD
    p = np.dot(x_d5,w_GD.T).reshape(-1,c)
    softmax = np.exp(p) / np.sum(np.exp(p),axis=1).reshape(-1,1)
    error = y - softmax
    dire = np.dot(error.T,x_d5)
    l = - np.sum(y*np.log(softmax))
    ww_history_GD.append(w_GD)
    ll_history_GD.append(float(l))
    w_GD = w_GD +  alpha / np.sqrt(t) * dire

    #Newton
    p = np.dot(x_d5,w_Newton.T).reshape(-1,c)
    softmax = np.exp(p) / np.sum(np.exp(p),axis=1).reshape(-1,1)
    error = y - softmax
    dire = np.dot(error.T,x_d5)
    er = softmax*(1-softmax)
    hess = np.dot(er.T,x_d5**2)
    l = - np.sum(y*np.log(softmax))
    ww_history_Newton.append(w_Newton)
    ll_history_Newton.append(float(l))
    w_Newton = w_Newton + alpha / np.sqrt(t) * dire/hess

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
