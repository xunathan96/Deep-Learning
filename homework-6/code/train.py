import numpy as np
from matplotlib import pyplot as plt

from data import IrisData
from utils import load_config
from model import GaussianMixtureModel
from model import MultivariateGaussian

def train(data, config):

	gmm = GaussianMixtureModel(config, data.x)
	gmm.initialize_EM()

	for epoch in range(4000):
		responsibilities = gmm.e_step()
		pi, mean, cov = gmm.m_step(responsibilities)

	print("class priors:", pi)
	class_assignments = gmm.class_assignments()

	return class_assignments, (pi, mean, cov)


def visualize(data, class_assignments, mixture_params):
	X = data.x
	Y = data.t
	pi, mean, cov = mixture_params

	#VISUALIZATION: a Cross Section
	plt.figure(figsize=(9,4))
	plt.subplot(121)
	for k in range(3):
		plt.scatter(X[class_assignments==k, 2], X[class_assignments==k, 1], s=2)
	plt.subplot(122)
	for k, class_name in enumerate(np.unique(Y)):
		plt.scatter(X[Y==class_name, 2], X[Y==class_name, 1], s=2)
	plt.show()

	#VISUALIZATION: PCA Projection
	evals, evecs = np.linalg.eigh(np.cov(X.T))
	to_crd = lambda x: ((x-x.mean(axis=0))@evecs)[:,-2:]
	crds = to_crd(X)

	plt.figure(figsize=(9,4))
	plt.subplot(121)
	for k in range(3):
		plt.scatter(crds[class_assignments==k, 0], crds[class_assignments==k, 1], s=2)
	plt.scatter(to_crd(mean)[:,0], to_crd(mean)[:,1], s=30, marker='+')
	plt.subplot(122)
	for k in np.unique(Y):
		plt.scatter(crds[Y==k, 0], crds[Y==k, 1], s=2)
	plt.show()



if __name__ == '__main__':
	config_path = "./config/config.yml"
	config = load_config(config_path)
	data = IrisData(config.data.path)
	class_assignments, params = train(data, config)
	visualize(data, class_assignments, params)

	