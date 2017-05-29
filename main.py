import random
import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import pylab
from PIL import Image


def veclabel(label):
    res = np.zeros((10, 1))
    res[label] = 1.0
    return res

def sigmoid(x):
    result = 1.0 / (1.0 + np.exp(-x))
    return result

data = fetch_mldata('MNIST original', data_home='C:\Anaconda3\Lib\site-packages\sklearn\datasets\data')
print(data.data.shape)
print(data.target.shape)
images, labels = data.data / 255., data.target
train_images, test_images = images[:60000], images[60000:]
train_labels, test_labels = labels[:60000], labels[60000:]

train_inputs = [np.reshape(x, (784, 1)) for x in train_images]
train_veclabels = [veclabel(x) for x in train_labels]
test_inputs = [np.reshape(x, (784, 1)) for x in test_images]


layers_count = 3
input_size = 28*28
hidden_size = 40
random.seed(1000)
hidden_bias = np.random.randn(hidden_size, 1)
output_size = 10
random.seed(10)
output_bias = np.random.randn(output_size, 1)
weights = [np.random.randn(y, x) for x, y in list(zip([input_size, hidden_size], [hidden_size, output_size]))]
biases = [hidden_bias, output_bias]
train_data = zip(train_inputs, train_veclabels)
test_data = zip(test_inputs, test_labels)

rate = 2
iterations = 50
batch_size = 4

train_data = list(train_data)
test_data = list(test_data)


for iteration in range(iterations):
    random.seed(iteration)
    random.shuffle(train_data)
    batches = [train_data[k:k+batch_size]
        for k in range(0, len(train_data), batch_size)]
    for batch in batches:

        tempbias = [np.zeros(bias.shape) for bias in biases]
        tempweight = [np.zeros(weight.shape) for weight in weights]
        for x, y in batch:

            delta_bias = [np.zeros(bias.shape) for bias in biases]
            delta_weight = [np.zeros(weight.shape) for weight in weights]

            activation = x
            activations = [x]
            z_array = []
            for bias, weight in zip(biases, weights):
                z = np.dot(weight, activation) + bias
                z_array.append(z)
                activation = sigmoid(z)
                activations.append(activation)

            delta = (activations[-1]- y) * sigmoid(z_array[-1])*(1-sigmoid(z_array[-1]))
            delta_bias[-1] = delta
            delta_weight[-1] = np.dot(delta, activations[-2].transpose())

            for layer in range(2, layers_count):
                z = z_array[-layer]
                sp = sigmoid(z)*(1-sigmoid(z))
                delta = np.dot(weights[-layer + 1].transpose(), delta) * sp
                delta_bias[-layer] = delta
                delta_weight[-layer] = np.dot(delta, activations[-layer - 1].transpose())

            new_bias = [newb + deltanb for newb, deltanb in zip(tempbias, delta_bias)]
            new_weight = [neww + deltanw for neww, deltanw in zip(tempweight, delta_weight)]

        biases = [bias - (rate / len(batch)) * newb for bias, newb in zip(biases, new_bias)]
        weights = [weight - (rate / len(batch)) * neww for weight, neww in zip(weights, new_weight)]
    print('iteration'+str(iteration)+': done!')
    test_results = []
    i=0
    # for image, label in test_data:
    #     output = list(image)
    #     for bias, weight in zip(biases, weights):
    #         output = sigmoid(np.dot(weight, output) + bias)
    #     test_results.append([np.argmax(output), label])
    #     if np.argmax(output) != label:
    #         # print(image)
    #         # print("#####")
    #         # print(image.shape)
    #         # print(image.shape )
    #         pixels = np.array(image)
    #         pixels = np.reshape(pixels, (-1, 28))
    #         # print(pixels.shape)
    #         # print(pixels)
    #         # print(np.asarray(image).ndim )
    #         plt.title('label: {0}, output: {1}'.format(label, np.argmax(output)))
    #         plt.imshow(pixels, cmap='gray')
    #
    #         import os
    #
    #         directory = os.path.dirname(os.path.realpath(__file__))
    #         title = '{0}-label-{1}-output-{2}.png'.format(i, label, np.argmax(output))
    #         # pylab.savefig(directory+'\\image\\'+title)
    #         i += 1
    #         # print (1)
    #
    # print('accuracy: {0}', 1 - i / 10000)
    # print(test_results)
test_results = []
i=0
for image, label in test_data:
    output = list(image)
    for bias, weight in zip(biases, weights):
        output = sigmoid(np.dot(weight, output) + bias)
    test_results.append([np.argmax(output), label])
    if np.argmax(output) != label:
        # print(image)
        # print("#####")
        # print(image.shape)
        #print(image.shape )
        pixels = np.array(image)
        pixels = np.reshape(pixels, (-1, 28))
        # print(pixels.shape)
        # print(pixels)
        #print(np.asarray(image).ndim )
        plt.title('label: {0}, output: {1}'.format(label, np.argmax(output)))
        plt.imshow(pixels, cmap='gray')

        import os
        directory = os.path.dirname(os.path.realpath(__file__))
        title = '{0}-label-{1}-output-{2}.png'.format(i, label, np.argmax(output))
        #pylab.savefig(directory+'\\image\\'+title)
        i+=1
        # print (1)

print('accuracy: {0}', 1-i/10000)
print(test_results)
print("######")
user_image = Image.open('user_image.png')
user_pixels = 1 - np.array(user_image.getdata())/255
user_pixels_black = []
for item in user_pixels:
    user_pixels_black.append(item[0])

user_pixels_black = np.reshape(user_pixels_black, (784, 1))
label = 2
output = list(user_pixels_black)
for bias, weight in zip(biases, weights):
    output = sigmoid(np.dot(weight, output) + bias)
print(np.argmax(output), label)
