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


mu, sigma = 0, 0.1 # mean and standard deviation

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
	#print p
	#print p_ftl

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



	

	#loss=np.zeros(shape=(N,1))
#	print 'hereeee'
	X, y=datasets.make_regression(n_samples=N,n_features=d,noise=0.01)

	X=preprocessing.normalize(X)
#	print X
	#y=y*0.01
	y = 2*(y-min(y))/(max(y)-min(y))-1
#	print y

	for n in range(N):
		y_best = np.dot(theta_best,representation_0(X[n],d))
		y_hat = np.dot(theta,representation_d(X[n],d,p))
		y_ftl = np.dot(theta_ftl,representation_d(X[n],d,p_ftl))

		#print y_hat
		#print y_best
		#print y_ftl

	
		
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
		#print (math.pow((y[n]-y_hat),2))

		if(n<N-1):
			theta_ftl = theta_ftl-chi*(np.dot(np.ndarray.transpose(X[n]),(math.pow((y[n]-y_ftl),2)))/d)
			#print loss.shape
			#print (theta[n].reshape(-1,1)).shape
		#	print np.divide(loss,theta[n].reshape(-1,1))
		#	print np.gradient(np.divide(loss,theta[n].reshape(-1,1)),axis=0)
			theta_best = theta_best-chi*(np.dot(np.ndarray.transpose(X[n]),(math.pow((y[n]-y_best),2)))/d)
			#print theta
			#print chi*(np.dot(np.ndarray.transpose(X[n]),(math.pow((y[n]-y_hat),2)))/d)
			theta = theta-chi*(np.dot(np.ndarray.transpose(X[n]),(math.pow((y[n]-y_hat),2)))/d)
		#print theta
		#print theta_best

			#theta_ftl[n+1] = np.argmin(loss_ftl_all,axis=1)
		#print np.linalg.norm(theta)
	#print theta

	loss_actual += (1.0/N)*loss[p]
	loss_actual_ftl += (1.0/N)*loss_ftl[p_ftl] 
	loss_actual_best += (1.0/N)*loss_best

	#print loss_actual
	#print loss_actual_best


	#print loss_ftl
	for i in range(d):
		loss_c[t][i] = (1.0/N)*loss[i]
		loss_ftl_c[t][i] = (1.0/N)*loss_ftl[i]
		#loss_best_c[t][i] = (1.0/N)*loss_best[i]

	#loss_t[t]=(1/float(N))*loss_c


	expsum=0
	expsum_best=0
	expsum_ftl=0
	#print loss
	#print loss_ftl
	#print loss_c[t]
	#print loss_ftl_c[t]
	#print loss_best_c[t]
	#if(t>0):
	for i in range(d):
		expsum = expsum+ math.exp(-eta*loss_c[t][i])*pi[i]
	#	expsum_best = expsum_best+ math.exp(-eta*loss_best_c[t][i])*pi_best[i]
	#	expsum_ftl = expsum_ftl+ math.exp(-eta*loss_ftl_c[i])*pi_ftl[i]

			#expsum_best = expsum_best+ math.exp(-eta*loss_best_all[i])*pi_best[i]


	# else:
	# 	expsum = expsum+ math.exp(-eta*loss_c[0])*pi[0]
	# 	expsum_best = expsum_best+ math.exp(-eta*loss_best_c[0])*pi_best[0]
	# 	expsum_ftl = expsum_ftl+ math.exp(-eta*loss_ftl_c[0])*pi_ftl[0]

#	print expsum
#	print expsum_best
#	print expsum_ftl

	for i in range(d):
		pi[i]=(math.exp(-eta*loss_c[t][i])*pi[i])/expsum
	
	p_ftl = np.argmin(loss_ftl_c[t])



	r[t] = (1.0/(t+1))*(loss_actual - loss_actual_best )

	r_ftl[t] =(1.0/(t+1))*(loss_actual_ftl - loss_actual_best)



x=np.zeros((Time,1))
for i in range(Time):
	x[i] = i



d_new=np.zeros((Time/2,1))
# regret=np.zeros((Time/10,1))
# regret_ftl=np.zeros((Time/10,1))

for i in range(Time/2):
	d_new[i]=out[i*2]
# 	regret[i]= (sum_c[i*10])-(sum_best_c[i*10])
# 	regret_ftl[i]= (sum_ftl_c[i*10])-(sum_best_c[i*10])


# r = loss_actual - loss_actual_best
# r_ftl =loss_actual_ftl - loss_actual_best

# d = r- r_ftl

# diff=[]
xnew = np.linspace(0,Time,1)
# diff=(regret-regret_ftl)
# print r
# print r_ftl
# diff_smooth=spline(x[:,0],r,xnew)
# diff_smooth_2=spline(x[:,0],r_ftl,xnew)
#print x
#print r

#print r_ftl

plt.plot(x,r,color='g')
plt.plot(x,r_ftl,color='red')

plt.title('fig2')
#plt.plot(xnew,logx ,color='red')






plt.show()




# y=[]
# time=100

# #declare the within task size
# d=10

# #using Stochastic Gradient Classifier as within-task algo
# clf = SGDClassifier()

# #declare representation g (for input values s)
# g=np.zeros(shape=(time,d))

# #declare labels
# y=np.zeros(shape=(time,d))
# pi=[]

# s=np.zeros(shape=(time,d))

# def fit(time,d):
# 	mu, sigma = 0, 0.1

# 	#declaring training dataset
# 	s=np.zeros(shape=(time,d))

# 	#declaring distribution g 
# 	g=np.zeros(shape=(time,d))


# 	for T in range(time):
# 		s[T] = np.random.normal(mu, sigma, d)
# 		g[T]=s[T]

# 		#setting g[0] as the best representation
# 		if(T!=0):
#  			g[T][d/2]=0

# 		y[T]=np.sign(s[T])
# 		clf.fit(s[T].reshape(-1,1), y[T])
# #	x=clf.predict(s[7].reshape(-1,1))
# #	print x
# 	return g

# def fit_within(s,y):
# 	print s
# 	print y
# 	clf.fit(s.reshape(-1,1), y)



# for i in range(time):
#     pi.append(0)


# def ewa_ll(g,x,pi,time,d):
# 	mu, sigma = 0, 0.1
# 	loss_t=[]
# 	eta=0.4
# 	for t in range(time):
# 		if(x==0):
# 			v=0
# 		else:
# 			v=t
		
# 		loss=0
# 		loss_best=0
# 		pi[0]=1.0/time
		
		

# 		#declaring test data
# 		s = np.random.normal(mu, sigma, d)

# 		for i in range(d):

# 			#true label
# 			y=np.sign(s[i])

# 			#predicted label
# 			y_hat=np.sign(np.dot(np.dot(pi[t],g[v][i]),s[i]))
# 			#clf.fit(s[i],y)
# 			#fit_within(s[i],y)
# 			#calculate squared loss
# 			loss=loss + math.pow((1-y*y_hat),2)
		

# 		loss =loss/float(d)
		
# 		if(t>0):
# 			loss_t.append(loss_t[t-1]+loss)
			
# 		else:
# 			loss_t.append(loss)
			
	
# 		expsum=0
# 		if(t>0):
# 			for i in range(t):
# 				expsum = expsum+ math.exp(-eta*loss_t[i])*pi[i]
# 		else:
# 			expsum = expsum+ math.exp(-eta*loss_t[0])*pi[0]


# 		#update the probabilities	
# 		if(t!=time-1):
# 			pi[t+1]= (math.exp(-eta*loss_t[t])*pi[t])/expsum
# 	return loss_t

# #get the set of representations (with one best representation known prior)
# dist=fit(time,d)
# loss_t=ewa_ll(dist,1,pi,time,d)
# loss_t_best=ewa_ll(dist,0,pi,time,d)

# loss_t = [x / float(time) for x in loss_t]
# loss_t_best = [x / float(time) for x in loss_t_best]
# x=[]

# #plot the results
# for i in range(time):
#    	x.append(i)
# plt.plot(x,loss_t,color='blue')
# plt.plot(x,loss_t_best,color='g')
# plt.xlabel("Time")
# plt.ylabel("Cumulative loss")
# plt.gca().legend(('cumulative loss with all the representations','cumulative loss with best representation'))
# plt.show()



	
	

