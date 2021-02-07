import numpy as np
import matplotlib.pyplot as plt
import dataset
import labels

a, b, c = dataset.a, dataset.b, dataset.c
y = labels.y

x = [np.array(a).reshape(1,30), np.array(b).reshape(1,30), np.array(c).reshape(1,30)]

y = np.array(y)

class nn(object):
	def sigmoid(self, x): 
		return 1/(1 + np.exp(-x))

	def forward(self, x, w1, w2):
		z1 = x.dot(w1)
		a1 = self.sigmoid(z1)

		z2 = a1.dot(w2)
		a2 = self.sigmoid(z2)

		return a2

	def init_wghts(self, x, y):
		l = []
		for i in range(x*y):
			l.append(np.random.randn())
		return np.array(l).reshape(x,y)

	def loss(self, out, y):
		s = (np.square(out - y))
		s = np.sum(s)/len(y)
		return s

	def backprop(self, x, y, w1, w2, alpha):
		z1 = x.dot(w1)  
		a1 = self.sigmoid(z1)
		# Output layer 
		z2 = a1.dot(w2)
		a2 = self.sigmoid(z2)

		d2 = (a2 - y)
		d1 = np.multiply((w2.dot((d2.transpose()))).transpose(),  
							   (np.multiply(a1, 1-a1))) 

		# Gradient for w1 and w2 
		w1_adj = x.transpose().dot(d1) 
		w2_adj = a1.transpose().dot(d2) 

		w1 = w1-(alpha*(w1_adj)) 
		w2 = w2-(alpha*(w2_adj)) 

		return w1, w2
  
	def train(self, x, Y, w1, w2, alpha = 0.01, epoch = 10): 
		acc =[] 
		loss =[] 
		for j in range(epoch): 
			l =[] 
			for i in range(len(x)): 
				out = self.forward(x[i], w1, w2) 
				l.append((self.loss(out, Y[i]))) 
				w1, w2 = self.backprop(x[i], y[i], w1, w2, alpha) 
			print("epochs:", j + 1, "======== acc:", (1-(sum(l)/len(x)))*100)    
			acc.append((1-(sum(l)/len(x)))*100) 
			loss.append(sum(l)/len(x)) 
		return acc, loss, w1, w2 
   
	def predict(self, x, w1, w2): 
		Out = self.forward(x, w1, w2) 
		maxm = 0
		k = 0
		for i in range(len(Out[0])): 
			if(maxm<Out[0][i]): 
				maxm = Out[0][i] 
				k = i 
		if(k == 0): 
			print("Image is of letter A.") 
		elif(k == 1): 
			print("Image is of letter B.") 
		else: 
			print("Image is of letter C.") 
		#plt.imshow(x.reshape(5, 6)) 
		#plt.show()     

nn1 = nn()
w1 = nn1.init_wghts(30, 5) 
w2 = nn1.init_wghts(5, 3) 

#print(w1, w2)

acc, losss, w1, w2 = nn1.train(x, y, w1, w2, 0.1, 100)
print(nn1.predict(x[1], w1, w2))

# plt.imshow(np.array(a).reshape(5,6))
# plt.show()