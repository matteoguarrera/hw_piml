import os
import pickle
import sys 
import argparse
import numpy as np
import copy
import scipy
import scipy.io

def test(dataset, train_idx, test_idx, nn):
	for idx,name in zip([train_idx,test_idx],['training','test']):
		
		E_true = dataset['T'][0,idx]
		R = dataset['R'][idx]
		Z = dataset['Z'][idx]

		print(f'{name} set')
		E_pred = np.array([nn.forward((R,Z)) for _ in range(10)]).mean(axis=0)
		print('MAE:  %5.2f kcal/mol'%np.abs(E_pred-E_true).mean(axis=0))
		print('RMSE: %5.2f kcal/mol'%np.square(E_pred-E_true).mean(axis=0)**.5)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--augmentation', action='store_true', help='whether to use model trained with augmentation')
	parser.add_argument('--train_fraction', type=float, default=0.5, help='Fraction of training split to use')
	parser.add_argument('--schedule', action='store_true', help='Custom Learning Rate Schedule')

	args = parser.parse_args()
	'''Load data'''
	if not os.path.exists('qm7.mat'): os.system('wget http://www.quantum-machine.org/data/qm7.mat')
	dataset = scipy.io.loadmat('qm7.mat')

	if args.schedule: str_schedule = '_custom_schedule'
	else: str_schedule = ''

	model_path = f'nn-augment={args.augmentation}{str_schedule}.pkl'
	nn = pickle.load(open(model_path,'rb'))

	print(f'results of {model_path} after {nn.nbiter} iterations')
	
	skip_every = int(1/args.train_fraction)
	train_idx = dataset['P'][1:].flatten()[::skip_every]
	test_idx  = dataset['P'][0]
	test(dataset, train_idx, test_idx, nn)



	

