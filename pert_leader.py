import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error
from random import randint
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets, linear_model
from scipy.interpolate import spline
from sklearn import preprocessing
import random


mu, sigma = 0, 0.1 # mean and standard deviation

d=20
d_rep=21

def identity(x):
	return x



def representation_0(X,D):
	g= np.zeros(shape=(D,1))
	for d in range(D):
		g[d]=identity(X[d])
		
	return g

def representation_d(X,D,n):
	g = np.zeros(shape=(D,1))
	for d in range(D):
		g[d]=identity(X[d])
		#g[d]=np.dot(n[d],identity(X[d])
	if(n!=0):
		g[n-1]=0
		
	return g



Time=100
N=1000
alpha=0.5

h= np.zeros(shape=(Time,d))
loss_act_best=0

pi=np.zeros(shape=(d_rep,1))
pi_best=np.zeros(shape=(d_rep,1))
pi_ftl =np.zeros(shape=(d_rep,1))

for i in range(d_rep):
	pi[i] = 1.0/d_rep
	pi_best[i] = 1.0/d_rep
	#pi_ftl[i] = 1.0/d

loss_all=np.zeros(shape=(d,1))
loss_best_all=np.zeros(shape=(d,1))
loss_ftl_all =np.zeros(shape=(d,1))
loss_c =np.zeros(shape=(Time,d_rep))
loss_ftl_c =np.zeros(shape=(1,d_rep))
loss_best_c =np.zeros(shape=(Time,d))
loss_actual=0
loss_actual_ftl=0
loss_actual_best=np.zeros(shape=(Time,1))


l_hat=np.zeros(shape=(1,d_rep))
L_cumu=np.zeros(shape=(1,d_rep))
l_total_k=np.zeros(shape=(d_rep,1))




out =np.zeros(shape=(Time,1))
r =np.zeros(shape=(Time,1))
r_ftl =np.zeros(shape=(Time,1))







eta=0.5*math.sqrt((2*math.log(d_rep))/Time)
chi=0.25/(math.sqrt(2*N))


print eta
print chi



p=randint(0,d_rep)

p_ftl=randint(0,d_rep)
for t in range(Time):
	print t
	loss =np.zeros(shape=(d_rep,1))
	loss_ftl=np.zeros(shape=(d_rep,1))
	loss_best=0
	#loss_best=np.zeros(shape=(d,1))
	loss_best_int=5



	#loss_best=0


	pred=np.zeros(shape=(d_rep,1))
	pred_best=np.zeros(shape=(d_rep,1))
	pred_ftl=np.zeros(shape=(d_rep,1))
	
	theta=np.zeros(shape=(1,d))
	theta_best=np.zeros(shape=(1,d))
	theta_ftl =np.zeros(shape=(1,d))
	for i in range(d):
		theta[0][i]=1.0/d
		theta_best[0][i]=1.0/d
		theta_ftl[0][i]=1.0/d



	

	#loss=np.zeros(shape=(N,1))
#	print 'hereeee'
	X, y=datasets.make_regression(n_samples=N,n_features=d)

	X=preprocessing.normalize(X)
	
	y = 2*(y-min(y))/(max(y)-min(y))-1
	

	for n in range(N):
		y_best = np.dot(theta_best,representation_0(X[n],d))
		y_hat = np.dot(theta,representation_d(X[n],d,p))
		y_ftl = np.dot(theta_ftl,representation_d(X[n],d,p_ftl))
		
		
			
		for rep in range(d_rep):
			pred[rep] =  np.dot(theta,representation_d(X[n],d,rep))
			pred_best[rep] =  np.dot(theta_best,representation_0(X[n],d))
			pred_ftl[rep] = np.dot(theta_ftl,representation_d(X[n],d,rep))
		#	print y[n]-pred[rep]
			if(rep!=p):
				loss[rep] += (math.pow((y[n]-pred[rep]),2))
			#loss_best[rep] += (math.pow((y[n]-pred_best[rep]),2))
			if(rep!=p_ftl):
				loss_ftl[rep] += (math.pow((y[n]-pred_ftl[rep]),2))



		loss[p] += (math.pow((y[n]-y_hat),2))
		loss_best += (math.pow((y[n]-y_best),2))
		loss_ftl[p_ftl] += (math.pow((y[n]-y_ftl),2))
		#print (math.pow((y[n]-y_hat),2))

		if(n<N-1):
			theta_ftl = theta_ftl-chi*abs((np.dot(np.ndarray.transpose(X[n]),(math.pow((y[n]-y_ftl),2)))/d))
			
			theta_best = theta_best-chi*abs((np.dot(np.ndarray.transpose(X[n]),(math.pow((y[n]-y_best),2)))/d))
			
			theta = theta-chi*abs((np.dot(np.ndarray.transpose(X[n]),(math.pow((y[n]-y_hat),2)))/d))
		
	loss_actual += (1.0/N)*loss[p]
	loss_actual_ftl += (1.0/N)*loss_ftl[p_ftl] 
	loss_act_best += (1.0/N)*loss_best
	loss_actual_best[t] = (1.0/(t+1))*(loss_actual_best[t]+(1.0/N)*(loss_best))

	
	
	for i in range(d_rep):
		loss_c[t][i] = (1.0/N)*loss[i]
		loss_ftl_c[0][i] += (1.0/N)*loss_ftl[i]
		#loss_best_c[t][i] = (1.0/N)*loss_best[i]

	#loss_t[t]=(1/float(N))*loss_c


	expsum=0
	expsum_best=0
	expsum_ftl=0
	
	for i in range(d_rep):
		expsum = expsum+ math.exp(-eta*loss_c[t][i])*pi[i]
	#	expsum_best = expsum_best+ math.exp(-eta*loss_best_c[t][i])*pi_best[i]
	
	for i in range(d_rep):
		pi[i]=(math.exp(-eta*loss_c[t][i])*pi[i])/expsum
	
	p_ftl = np.argmin(loss_ftl_c[0])
	X_ewa = random.uniform(0,1)
	sum_p=0
	for i in range(d_rep):
		sum_p += pi[i]
		if(sum_p-X_ewa>0):
			p=i
			break

	X_ftl = random.uniform(0,1)
	sum_p=0
	for i in range(d_rep):
		sum_p += ((1-alpha)*((1.0/N)*loss_ftl[i]))
		if(sum_p-X_ftl>0):
			l_hat[0][i]=1
		else:
			l_hat[0][i]=0
		
		L_cumu[0][i] += l_hat[0][i]
		l_total_k[i]+=(1.0/N)*loss_ftl[i]
	p_ftl = np.argmin(L_cumu[0])
		






	r[t] = (1.0/(t+1))*((loss_actual) - loss_act_best)

	r_ftl[t] =(1.0/(t+1))*((loss_actual_ftl) - loss_act_best)




x=np.zeros((Time,1))
r_cum_ewa=np.zeros((Time,1))
r_cum_ftl=np.zeros((Time,1))
r_cum_ewa[0]=r[0]
r_cum_ftl[0]=r_ftl[0]


for i in range(Time):
	x[i] = i

for i in range(Time-1):
	r_cum_ewa[i+1] =r_cum_ewa[i] +r[i+1]
	r_cum_ftl[i+1] =r_cum_ftl[i] +r_ftl[i+1]





plt.plot(x,r_cum_ewa,color='g')
plt.plot(x,r_cum_ftl,color='red')

plt.title('fig2')







plt.show()







	
	

