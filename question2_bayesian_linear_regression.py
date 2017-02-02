from __future__ import division 
import numpy as np
from scipy.optimize import curve_fit 
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# first, we set our random seed and draw the x points at random from np.random.seed(25)
# we now extract the 1D data points from the uniform distribution
sample_size=10000
data = np.random.uniform(low=-100, high=100, size=sample_size)
alpha_value=100
           # the polynomial that generates the label for each point
f = lambda x: 0.1 + 2*x + x**2 + 3*x**3
           # and finally, we populate the label array
label = np.array([]) 
for elem in data:
	label = np.append(label, np.random.normal(loc=f(elem), scale=1., size=1))


def phi(x, poly_degree):
    """ generate the polynomial vector """
    this_phi = np.zeros((poly_degree+1, x.shape[0]))
    for k in range(poly_degree+1):
        this_phi[k] = x**k
    return this_phi


def gimme_the_S(alpha, beta, poly_degree, data):
    """ generate the matrix S """
    # initialize S as alpha*I
    S = np.multiply(alpha, 
                    np.identity(poly_degree+1))
    
    # compute the kernelized data matrix Phi
    my_phi = phi(data, poly_degree)
    
    # compute the final version of S     
    S = np.add(S, np.multiply(beta, 
                              # this dot product is equivalent to 
                              # the sum over all phi from the data
                              np.dot(my_phi, 
                                     my_phi.transpose())))
    
    return np.linalg.inv(S)



def gimme_s(beta, S, new_data):
    """ compute the variance s for regressing on a new data point """
    current_degree = S.shape[0] - 1
    new_phi = phi(new_data, current_degree)
    
    # again, instead of doing the sum notation, 
    # we just do dot products
    s2 = 1./beta + np.dot(new_phi.transpose(), 
                          np.dot(S, 
                                 new_phi))
    
    return np.sqrt(s2)



def gimme_m(beta, new_data_point, old_data, old_labels, S):
    """ compute the mean m for regressing on a new data point """  
    transformed_point = phi(new_data_point, 
                            S.shape[0]-1)
    
    a = np.dot(phi(old_data, S.shape[0]-1), old_labels)

    b = np.dot(S, a)

    c = np.dot(transformed_point.transpose(), b)
    
    return beta*c


S=gimme_the_S(alpha=alpha_value, beta=1, poly_degree=3, data=data)

s = gimme_s(1, S, np.array([0]))

gimme_m(beta=1, new_data_point=np.array([0]), old_data = data, old_labels = label, S=S)


#----------plotting----------------#
plotting_data = np.linspace(-10, 10, 1000)
true_labels = f(plotting_data)

estimated_mean, estimated_variance = np.array([]), np.array([])
for point in plotting_data:
    estimated_mean = np.append(estimated_mean, 
                               gimme_m(beta=1, 
                                       new_data_point=np.array([point]), 
                                       old_data=data, 
                                       old_labels=label, 
                                       S=S))
    estimated_variance = np.append(estimated_variance, 
                                   gimme_s(beta=1, 
                                           S=S, 
                                           new_data=np.array([point])))

# and now we plot
plt.figure(25, figsize=(20,10))
plt.plot(data, label, 'o', markerfacecolor='white', 
	markersize=8, label=r'Pairs $(x,t)$ for random $x,t$')
plt.plot(plotting_data, true_labels, "g", label="True curve", 
         linewidth=3)
plt.plot(plotting_data, estimated_mean, "b", 
         label="Estimated Mean m(x)", linestyle="--", 
         linewidth=4)
plt.fill_between(plotting_data, 
                 np.subtract(estimated_mean, estimated_variance), 
                 np.add(estimated_mean, estimated_variance), 
                 color="r", alpha=0.5)
plt.title(r'Third Degree Polynomial Approximation for %d data points and $\alpha$ = %d'  %(sample_size, alpha_value))
plt.xlim(-3,3)
plt.ylim(-30,30)
plt.xlabel("x")
plt.ylabel("label")
plt.legend(loc='upper left')
plt.show()
