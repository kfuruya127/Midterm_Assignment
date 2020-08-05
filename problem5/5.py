import numpy as np
import matplotlib.pyplot as plt
import time

def gausian_kernel(x,d,alpha):
    n = x.shape[0]
    zks = np.arange(0,n)
    zks = np.random.permutation(zks)
    mapped_x = np.zeros((n,d))
    for i,xi in enumerate(x):
        mapped = np.array([])
        for j in range(0,d):
            mapped = np.append(mapped,np.exp((-alpha)*(xi-x[zks[j%n]])@((xi-x[zks[j%n]]).T)))
            if j == n-1:
                zks = np.random.permutation(zks)
        mapped_x[i] = mapped
    return mapped_x

# dataset 1
np.random.seed(123)
n = 100
x_d1 = 3 * (np.random.rand(n, 2)-0.5)
radius = x_d1[:,0]**2 + x_d1[:,1]**2
y_d1 = (radius > 0.7 + 0.1 * np.random.randn(n)) &( radius < 2.2 + 0.1 * np.random.randn(n))
y_d1 = 2 * y_d1 -1

d_tilde = 90
alpha = 100

mapped_x = gausian_kernel(x_d1,d_tilde,alpha)

it = range(1,301)
mapped_x = np.append(mapped_x,np.ones((n,1)),axis=1)
y_d1 = np.reshape(y_d1,(n,1))
w = np.random.rand(len(mapped_x[0]),1)
lam = 0.01
ll_history = []
ww_history = []
sig = 0.1

start = time.time()
for t in it:
    pos = 1 / (1+np.exp(-y_d1*mapped_x@w))
    dire = np.sum(((1-pos)*y_d1*mapped_x),axis=0)+2*lam*w.T
    l = np.sum(np.log((1+np.exp(-y_d1*mapped_x@w))))+lam*w.T@w
    ww_history.append(w)
    ll_history.append(float(l))
    w = w +  sig / np.sqrt(t) * dire.T

elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
plt.plot(it, ll_history, 'bo-', markersize=0.5, label='steepest')
plt.legend()
plt.xlabel('itration')
plt.ylabel('loss')
plt.show()
