import numpy as np
import matplotlib.pyplot as plt

def calculate_accuracy(logits, labels):
	# logits has dimensions N, C
	# labels has dimensions N 
	preds = np.argmax(logits, axis=1)
	return np.average(np.equal(preds, labels))

def display_model_performance(loss, accuracy):
	plt.plot(loss, color='blue', label='training data')
	plt.legend()
	plt.title('Loss')
	plt.ylabel('Average CE Loss')
	plt.xlabel('Epoch')
	plt.show()

	plt.plot(accuracy, color='green', label='training data')
	plt.legend()
	plt.title('Accuracy')
	#plt.ylabel('')
	plt.xlabel('Epoch')
	plt.show()
