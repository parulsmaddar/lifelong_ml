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



d=20

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
		g[n]=0
		
	return g



Time=1000
N=10000

h= np.zeros(shape=(Time,d))

pi=np.zeros(shape=(d,1))
pi_best=np.zeros(shape=(d,1))
pi_ftl =np.zeros(shape=(d,1))

for i in range(d):
	pi[i] = 1.0/d
	pi_best[i] = 1.0/d
	#pi_ftl[i] = 1.0/d

loss_all=np.zeros(shape=(d,1))
loss_best_all=np.zeros(shape=(d,1))
loss_ftl_all =np.zeros(shape=(d,1))
loss_c =np.zeros(shape=(Time,d))
loss_ftl_c =np.zeros(shape=(Time,d))
loss_best_c =np.zeros(shape=(Time,d))
loss_actual=0
loss_actual_ftl=0
loss_actual_best=0

out =np.zeros(shape=(Time,1))
r =np.zeros(shape=(Time,1))
r_ftl =np.zeros(shape=(Time,1))






eta=2*math.sqrt((2*math.log(d))/Time)
chi=0.3162/(math.sqrt(2*N))

p_ftl=0
for t in range(Time):
	loss =np.zeros(shape=(d,1))
	loss_ftl=np.zeros(shape=(d,1))
	loss_best=0

	p = np.argmax(pi)
	

	pred=np.zeros(shape=(d,1))
	pred_best=np.zeros(shape=(d,1))
	pred_ftl=np.zeros(shape=(d,1))
	
	theta=np.zeros(shape=(1,d))
	theta_best=np.zeros(shape=(1,d))
	theta_ftl =np.zeros(shape=(1,d))
	for i in range(d):
		theta[0][i]=1.0/d
		theta_best[0][i]=1.0/d
		theta_ftl[0][i]=1.0/d

	X, y=datasets.make_regression(n_samples=N,n_features=d,noise=0.01)

	X=preprocessing.normalize(X)
	y = 2*(y-min(y))/(max(y)-min(y))-1

	for n in range(N):
		y_best = np.dot(theta_best,representation_0(X[n],d))
		y_hat = np.dot(theta,representation_d(X[n],d,p))
		y_ftl = np.dot(theta_ftl,representation_d(X[n],d,p_ftl))

	
		
		for rep in range(d):
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
		

		if(n<N-1):
			theta_ftl = theta_ftl-chi*(np.dot(np.ndarray.transpose(X[n]),(math.pow((y[n]-y_ftl),2)))/d)


	loss_actual += (1.0/N)*loss[p]
	loss_actual_ftl += (1.0/N)*loss_ftl[p_ftl] 
	loss_actual_best += (1.0/N)*loss_best

	for i in range(d):
		loss_c[t][i] = (1.0/N)*loss[i]
		loss_ftl_c[t][i] = (1.0/N)*loss_ftl[i]
		


	expsum=0
	expsum_best=0
	expsum_ftl=0
	
	for i in range(d):
		expsum = expsum+ math.exp(-eta*loss_c[t][i])*pi[i]


	for i in range(d):
		pi[i]=(math.exp(-eta*loss_c[t][i])*pi[i])/expsum
	
	p_ftl = np.argmin(loss_ftl_c[t])



	r[t] = (1.0/(t+1))*(loss_actual - loss_actual_best )

	r_ftl[t] =(1.0/(t+1))*(loss_actual_ftl - loss_actual_best)



x=np.zeros((Time,1))
for i in range(Time):
	x[i] = i


plt.plot(x,r,color='g')
plt.plot(x,r_ftl,color='red')
plt.title('fig2')
plt.show()


	
	

