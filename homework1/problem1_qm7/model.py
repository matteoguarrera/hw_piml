import numpy as np

class Module:
	def update(self,lr): pass
	def average(self,nn,a): pass
	def backward(self,DY): pass
	def forward(self,X): pass

class Sequential(Module):

	def __init__(self,modules):
		self.modules = modules
	
	def forward(self,X):
		'''Given input X, perform a full forward pass through the MLP'''
		#TODO: fill in this function
		for module in self.modules:
			X = module.forward(X)
		return X

	def backward(self,DY):
		'''Perform a full backward pass through the MLP. 
			DY is gradient of the loss w.r.t the final output'''
		#TODO: fill in this function
		for module in self.modules[::-1]:  # reverse list
			DY = module.backward(DY)
		return DY

	def update(self,lr):
		for m in self.modules: 
			X = m.update(lr)
		
	def average(self,nn,a):
		for m,n in zip(self.modules,nn.modules): 
			m.average(n,a)

class Input(Module):
	def __init__(self, inp):
		R, Z = inp
		sample_in = np.concatenate([R, np.expand_dims(Z, -1)], axis = -1)
		self.nbout = sample_in.shape[-2] * sample_in.shape[-1]
	def forward(self,inp): 
		R, Z = inp
		rz = np.concatenate([R, np.expand_dims(Z, -1)], axis = -1)
		return rz.reshape(rz.shape[0], -1)

class Output(Module):

	def __init__(self,T):
		self.tmean = T.mean()
		self.tstd  = T.std()
		self.nbinp = 1

	def forward(self,X):
		# un-normalize the final prediction 
		self.X = X.flatten()
		return self.X*self.tstd+self.tmean

	def backward(self,DY):
		#TODO: fill in this function
		''' recall the derivative doesn't depend on the mean '''
		# Maybe add a small eps in the division for stability
		# Reverse operation of flatten in forward
		return np.ones((self.X.shape[0], 1)) * DY * self.tstd

class Linear(Module):

	def __init__(self,m,n):
		self.lr = 1 / m**.5
		self.W = np.random.normal(0,1 / m**.5,[m,n]).astype('float32')
		self.B = np.zeros([n]).astype('float32')

	def forward(self,X):
		#TODO: fill in this function

		self.X = X
		Z = X@self.W + self.B

		# print(f'Input  linear: {X.shape}')
		# print(f'Output linear: {Z.shape}')

		return Z

	def backward(self,DY):
		#TODO: fill in this function
		'''
		In the backwards pass functions, you’ll want to set the gradients of the weights of
		that layer (i.e self.DW = ...), and return the gradient of the loss with respect to that layer’s input
		'''
		# Gradient with respect to input
		self.DX = DY @ self.W.T

		# Gradient with respect to weights
		self.DW = self.X.T @ DY    # has to be same dimention of self.W  (100,1)

		# Gradient with respect to biases
		self.DB = np.sum(DY, axis=0)

		assert self.W.shape == self.DW.shape  # check dimentions
		assert self.X.shape == self.DX.shape  # check dimentions
		assert self.B.shape == self.DB.shape  # check dimentions

		return self.DX

	def update(self,lr):
		self.W -= lr*self.lr*self.DW
		self.B -= lr*self.lr*self.DB

	def average(self,nn,a):
		self.W = a*nn.W + (1-a)*self.W
		self.B = a*nn.B + (1-a)*self.B

class Tanh(Module):
	
	def forward(self,X):
		#TODO: fill in this function
		'''Compute hyperbolic tangent element-wise. '''
		return np.tanh(X)

		
	def backward(self,DY):
		#TODO: fill in this function
		return 1 - np.tanh(DY)**2
