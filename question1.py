from __future__ import division 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
from matplotlib.patches import Rectangle

x_points=[-1,1,10,-0.5,0]
y_fixed=[0.1]*5

#########################
#Question 1b
#########################
mu_prior = 0
variance_prior = 1
sigma_prior = math.sqrt(variance_prior)

mu_post=19/12
variance_post= 1/6
sigma_post= math.sqrt(variance_prior)

plt.title('Prior and Posterior DIstributions of'+ r'$ \, \mu$' )

x = np.linspace(-4, 5, 100)
plot_prior, =plt.plot(x,mlab.normpdf(x, mu_prior, sigma_prior))
plt.text(-2.5, .25, r'$\mu=0,\ \sigma=1$')



plot_post, =plt.plot(x,mlab.normpdf(x, mu_post, sigma_post))
plt.text(2.8, .25, r'$\mu= \frac{19}{12},\ \sigma=\frac{1}{6}$')

plot_points, =plt.plot(x_points,y_fixed,'bo',color='r', markerfacecolor='red',markersize=5)
plt.legend([ plot_prior, plot_post,plot_points ], ( "Prior",'Posterior', 'Data'))

plt.xlim([-4,11])
plt.show()


#########################
#Question 1c
#########################
mu_prior = 10
variance_prior = 1
sigma_prior = math.sqrt(variance_prior)

mu_post=3.25
variance_post= 1/6
sigma_post= math.sqrt(variance_prior)

plt.title('Prior and Posterior DIstributions of'+ r'$ \, \mu$' )

x = np.linspace(0, 14, 100)
plot_prior, =plt.plot(x,mlab.normpdf(x, mu_prior, sigma_prior))
plt.text(9, .05, r'$\mu=10,\ \sigma=1$')



plot_post, =plt.plot(x,mlab.normpdf(x, mu_post, sigma_post))
plt.text(2, .05, r'$\mu=3.25,\ \sigma=\frac{1}{6}$')

plot_points, =plt.plot(x_points,y_fixed,'bo',color='r', markerfacecolor='red',markersize=5)
plt.legend([ plot_prior, plot_post,plot_points ], ( "Prior",'Posterior', 'Data'), loc='upper left')

plt.xlim([-2,14])

plt.show()



#########################
#Question 1d
#########################
mu_prior = 0
variance_prior = 1
sigma_prior = math.sqrt(variance_prior)

mu_post=12/8
variance_post= 1/8
sigma_post= math.sqrt(variance_prior)

plt.title('Prior and Posterior DIstributions of'+ r'$ \, \mu$' )

x = np.linspace(-4, 5, 100)
plot_prior, =plt.plot(x,mlab.normpdf(x, mu_prior, sigma_prior))
plt.text(-2.5, .25, r'$\mu=0,\ \sigma=1$')



plot_post, =plt.plot(x,mlab.normpdf(x, mu_post, sigma_post))
plt.text(2.8, .25, r'$\mu= \frac{12}{8},\ \sigma= \frac{1}{8}$')

plot_points, =plt.plot(x_points,y_fixed,'bo',color='r', markerfacecolor='red',markersize=5)
plt.legend([ plot_prior, plot_post,plot_points ], ( "Prior",'Posterior', 'Data'))

plt.xlim([-4,11])


plt.show()

