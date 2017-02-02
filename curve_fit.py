from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
import random #for uniform distr 
import scipy.optimize as optimize #for minimizing RMSE
from scipy.optimize import curve_fit #for minimizing RMSE
from sympy import *
from sympy import ones
from sympy.plotting import plot

x=symbols('x')
#define two arrays for x,t pairs
sample_size=1000
x_values=[]
t_values=[]
S_mat=[]
B_ML=0.98

#define f
def f(k,w0,w1,w2,w3,w4=0,w5=0):
	return w0+w1*k+ w2*pow(k,2)+ w3* pow(k,3)+w4*pow(k,4)+w5*pow(k,5)

def generate_values(n):

	for i in range(1,n):
		x= random.uniform(-100,100)
		mu=f(x,0.1,2,1,3)
		sigma=1
		t = np.random.normal(mu, sigma)
		x_values.append(x)
		t_values.append(t)

def phi_x(degree, x_values):
	f_array = Matrix(1, degree+1, lambda i,j: x_values**j )
	return f_array

def phi_x_symbolic(degree):
	f_symb= Matrix(1, degree+1, lambda i,j: x**j )
	return f_symb



def find_Smat(degree, alpha, betaML):

	alpha_I= alpha *eye(degree+1)

	temp = Matrix(degree+1, degree+1, lambda i,j: 0 )
	for element in x_values:
		temp+= phi_x(degree, element).transpose() * phi_x(degree,element)

	mult_by_beta= betaML * temp


	#add with alpha_I
	S_inverse= alpha_I + mult_by_beta

	Smat=S_inverse.inv()

	#print Smat

	return Smat

def m_x(degree,alpha,betaML):

	phi_sym_XT= phi_x_symbolic(degree)

	mult_by_beta= betaML * phi_sym_XT

	mult_by_S= mult_by_beta * find_Smat(degree, alpha, betaML)

	#find sum
	temp = Matrix(degree+1, 1, lambda i,j: 0 )
	for el_x, el_t in zip(x_values,t_values):
		temp=temp+phi_x(degree, el_x).transpose() * el_t


	final_m= mult_by_S *temp
	expr=final_m[0,0]
	fun=lambdify(x,expr)
	#print simplify( final_m[0,0] )

	k=np.linspace(-5, 5, 1000, endpoint=True)
	plot_prob, =plt.plot(x_values,t_values,'o', markerfacecolor='white', 
		markersize=8, markevery=1, label=r'Pairs $(x,t)$ for random $x,t$')

	plot_real,=plt.plot(k, fun(k) ,'r-', 
		label=r"$ m(x) $")

	plot_real,=plt.plot(k, fun(k)+B_ML ,'g--', 
		label=r"$ m(x) +\beta $")

	plot_real,=plt.plot(k, fun(k)-B_ML ,'b--', 
		label=r"$ m(x) $")

	plt.grid(True)
	plt.legend(loc='upper left')
	plt.ylim([-50,50])
	plt.xlim([-5,5])
	plt.show()

	print final_m[0,0]

	return final_m[0,0]


	#find sum term


if __name__=='__main__':


	generate_values(sample_size)

	#find_Smat(3, 1, 2)



	alpha=100
	beta_ml=1
	degree=3

	m_x(degree,alpha,B_ML)

	degree=5
	m_x(degree,alpha,B_ML)

	#x=symbols('x')
	#m=Matrix(2,2,[x,x**2,3,4])

	#print m.inv()

