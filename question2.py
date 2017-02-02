from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
import random #for uniform distr 
import scipy.optimize as optimize #for minimizing RMSE
from scipy.optimize import curve_fit #for minimizing RMSE

random.seed()

#define two arrays for x,t pairs
sample_size=3
x_values=[]
t_values=[]
S_mat=[]

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

def raise_to_power(an_array, n):
	return [x**n for x in an_array]

#which_element_x = give an element x fomr array x_values
def build_f(which_element_x, degree):
	f_array=[]
	for i in range (0,degree+1):
		f_array.append(which_element_x**i)
	return f_array

def add_arrays(one,two):
	return [x+y for x,y in zip(one,two)]

def find_S(alpha,betaML,degree):
	#first find the sum f(x_n)*f(x)
	e=alpha+ betaML*sample_size
	invert_e= 1/e 
	S_mat.append( invert_e ) #fix the first entry, since sum of (1)= sample size and invert  
	for i in range(1,degree+1):
		entry = alpha+ betaML*sum(raise_to_power(x_values,i))
		#invert entry
		invert_entry=1/entry
		#print "entry %f , invert %f" %(entry, invert_entry)
		S_mat.append(invert_entry)


def find_m_x(alpha,betaML,degree, S=S_mat):

	#first do the beta * phi(x) *S part
	#Simply multiply by beta all entries, remember that 
	#powers are now also squared

	beta_list=[x * betaML for x in S_mat]
	print "beta list: ", beta_list

	#now focus on the sum part
	observed_sum= [0] * (degree+1) #initialize as list of 0s
	for x_elem, t_elem in zip(x_values,t_values):
		f_array=build_f(x_elem, degree)
		print "f_array= " , f_array
		ft_array=[x * t_elem for x in f_array]
		observed_sum= add_arrays( observed_sum, ft_array)

	print "observed_sum= ", observed_sum


	#final list 
	m_list=[]
	for beta, element in zip(beta_list, observed_sum):
		m_list.append(beta*element)

	return m_list







		

def minimize_degree3():

	x0=np.array([0,0,0,0])


	xnp_values=np.asarray(x_values)
	tnp_values=np.asarray(t_values)
	#CODE FOR PART B
	FitParams3= np.polyfit( xnp_values, tnp_values, 3)
	print "For degree 3 (w3,w2,w1,w0)="
	print FitParams3
	p = np.poly1d(FitParams3)

	plt.figure()
	plt.title('Third Degree Polynomial Approximation for %d data points'  %(sample_size))

	plt.ylim([-3000000,3000000])
	plt.xlim([-200,200])

	plt.text(-200, -3700000, '*only 1 in 30 points appears in the graph for illustration purposes', 
		style='oblique', fontsize=10,bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})

	plot_prob, =plt.plot(x_values,t_values,'o', markerfacecolor='white', 
		markersize=8, markevery=30, label=r'Pairs $(x,t)$ for random $x,t$')
	k=np.linspace(-200, 200, 1000, endpoint=True)
	plot_real,=plt.plot(k,FitParams3[3]+FitParams3[2]*k+ FitParams3[1]* pow(k,2)+ FitParams3[0]* pow(k,3) ,'m--', 
		label=r"$f(x)=w_0 + \sum_{i = 1}^{3} w_i x^i $")
	
	plt.grid(True)
	plt.legend(loc='upper left')

	#FIND BETA
	sq_sum=0
	for i in range(0,sample_size-1):
		sq_sum+= (t_values[i]-f(x_values[i],FitParams3[3],FitParams3[2],FitParams3[1],FitParams3[0]))**2


	sq_sum= sample_size/sq_sum #invert to find beta

	print "beta = %f" %(sq_sum)


	plt.show()



def minimize_degree5():

	x0=np.array([0,0,0,0])


	xnp_values=np.asarray(x_values)
	tnp_values=np.asarray(t_values)
	#CODE FOR PART B
	FitParams5= np.polyfit( xnp_values, tnp_values, 5)
	print "For degree 5 (w5,w4,w3,w2,w1,w0)="
	print FitParams5
	# plt.figure()
	plt.title('Fifth Degree Polynomial Approximation for %d data points'  %(sample_size))

	plt.ylim([-3000000,3000000])

	plt.text(-110, -3700000, '*only 1 in 30 points appears in the graph for illustration purposes', 
		style='oblique', fontsize=10,bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})

	plot_prob, =plt.plot(x_values,t_values,'o', markerfacecolor='white', 
		markersize=8, markevery=30, label=r'Pairs $(x,t)$ for random $x,t$')
	k=np.linspace(-100, 100, 1000, endpoint=True)
	plot_real,=plt.plot(k,FitParams5[5]+FitParams5[4]*k+ FitParams5[3]* pow(k,2)+ FitParams5[2]* pow(k,3) + FitParams5[1]* pow(k,4) + FitParams5[0]* pow(k,5),'m--', 
		label=r"$f(x)=w_0 + \sum_{i = 1}^{5} w_i x^i $")
	
	plt.grid(True)
	plt.legend(loc='upper left')

		#FIND BETA
	sq_sum=0
	for i in range(0,sample_size-1):
		sq_sum+= (t_values[i]-f(x_values[i],FitParams5[5],FitParams5[4],FitParams5[3],FitParams5[2],FitParams5[1],FitParams5[0]))**2
	sq_sum= sample_size/sq_sum #invert to find beta

	print "beta = %f" %(sq_sum)
	plt.show()



if __name__=='__main__':

	#generate_values(sample_size) 

	x_values.append(1)
	x_values.append(2)
	x_values.append(3)

	t_values.append(100)
	t_values.append(200)
	t_values.append(300)


	alpha=1
	beta_ml=2
	degree=3
	find_S(alpha,beta_ml,degree)
	print "S matrix= ", S_mat
	miu= find_m_x(alpha, beta_ml, degree)
	print miu

	#minimize_degree3()
	#minimize_degree5()
	#fit_polynom(0)
	
	










	



