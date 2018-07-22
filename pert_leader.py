
import numpy as np
from sklearn import datasets
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing

Time=100
d=10
alpha=0.3
N=10

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

L_cumu=np.zeros(shape=(1,d))
l=np.zeros(shape=(1,d))
l_hat=np.zeros(shape=(1,d))
l_total_k=np.zeros(shape=(d,1))


regret=np.zeros(shape=(1,Time))
l_total=0

for t in range(Time):
	p_ftl = np.argmin(L_cumu)
	pred_ftl=np.zeros(shape=(d,1))
	theta_ftl =np.zeros(shape=(1,d))
	l=np.zeros(shape=(1,d+1))
	chi=0.3162/(math.sqrt(2*N))

    
	X, y=datasets.make_regression(n_samples=N,n_features=d)

	X=preprocessing.normalize(X)
	y = 2*(y-min(y))/(max(y)-min(y))-1

	#y = (y-min(y))/(max(y)-min(y))
	#y=y*0.01
	#print y

	for n in range(N):
		y_ftl = np.dot(theta_ftl,representation_d(X[n],d,p_ftl))
		for rep in range(d):
			pred_ftl[rep] = np.dot(theta_ftl,representation_d(X[n],d,rep))
			if(rep!=p_ftl):
				l[0][rep] += (math.pow((y[n]-pred_ftl[rep]),2))

		theta_ftl = theta_ftl-chi*(np.dot(np.ndarray.transpose(X[n]),(math.pow((y[n]-y_ftl),2)))/d)


		l[0][p_ftl] += (math.pow((y[n]-y_ftl),2))

	print l

	l_total += (1.0/N)*l[0][p_ftl]


	for i in range(d):
		if(((1-alpha)*((1.0/N)*l[0][i]))>0.5):
			l_hat[0][i]=1
		else:
			l_hat[0][i]=0
		L_cumu[0][i] = L_cumu[0][i]+l_hat[0][i]
		l_total_k[i]+=(1.0/N)*l[0][i]

	loss_min = np.amin(l_total_k)
	regret[0][t]= (1.0/(t+1))*l[0][p_ftl]-loss_min


x=np.zeros((1,Time))
for i in range(Time):
	x[0][i] = i

plt.plot(np.ndarray.transpose(x),np.ndarray.transpose(regret),color='g')
plt.title('fig2')
plt.show()

	



