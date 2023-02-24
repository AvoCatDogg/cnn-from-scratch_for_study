import mnist
import cifar10
import numpy as np
from conv import Conv3x3
from conv import Conv3x3xn
from maxpool import MaxPool2
from softmax import Softmax
import os
import pickle
'''
for i in range(3):
  with open(f"\cifar-10-batches-py\data_batch_{i+1}", 'rb') as fo:
      dict = pickle.load(fo, encoding='bytes')
'''

# We only use the first 1k examples of each set in the interest of time.
# Feel free to change this if you want.

with open('/cnn-from-scratch_for_study/cifar-10-batches-py/data_batch_1', 'rb') as fo:
      dict = pickle.load(fo, encoding='bytes')
      train_images = np.array(dict[b'data']).reshape(10000, 3, 32, 32).astype(np.float32)
      train_labels = np.array(dict[b'labels'])

with open('/cnn-from-scratch_for_study/cifar-10-batches-py/test_batch', 'rb') as fo:
      dict = pickle.load(fo, encoding='bytes')
      test_images = np.array(dict[b'data']).reshape(10000, 3, 32, 32).astype(np.float32)
      test_labels = np.array(dict[b'labels'])

conv1_r = Conv3x3(8)                  # 32x32x1 -> 30x30x4
conv1_g = Conv3x3(8)                  # 32x32x1 -> 30x30x4
conv1_b = Conv3x3(8)                  # 32x32x1 -> 30x30x4

pool = MaxPool2()                  # 28x28x8 -> 15x15x16

softmax = Softmax(15 * 15 * 8, 10) # 15x15x16 -> 10

def forward(image, label):
  '''
  Completes a forward pass of the CNN and calculates the accuracy and
  cross-entropy loss.
  - image is a 2d numpy array
  - label is a digit
  '''
  # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
  # to work with. This is standard practice.
  out = conv1_r.forward((image[0] / 255) - 0.5) + conv1_g.forward((image[1] / 255) - 0.5) + conv1_b.forward((image[2] / 255) - 0.5)
  out = pool.forward(out)
  out = softmax.forward(out)

  # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0

  return out, loss, acc

def train(im, label, lr):
  '''
  Completes a full training step on the given image and label.
  Returns the cross-entropy loss and accuracy.
  - image is a 2d numpy array
  - label is a digit
  - lr is the learning rate
  '''
  # Forward
  out, loss, acc = forward(im, label)

  # Calculate initial gradient
  gradient = np.zeros(10)
  gradient[label] = -1 / out[label]

  # Backprop
  gradient = softmax.backprop(gradient, lr)
  gradient = pool.backprop(gradient)
  gradient = (conv1_r.backprop(gradient, lr) + conv1_g.backprop(gradient, lr) + conv1_b.backprop(gradient, lr)) / 3

  return loss, acc

print('MNIST CNN initialized!')

# Train the CNN for 3 epochs
epoch_num = 5
lr = 0.005
for epoch in range(epoch_num):
  print('--- Epoch %d ---' % (epoch + 1))

  # Shuffle the training data
  permutation = np.random.permutation(len(train_images))
  train_images = train_images[permutation]
  train_labels = train_labels[permutation]

  # Train!
  loss = 0
  num_correct = 0
  step_arr = []
  loss_arr = []
  accuracy_arr = []

  for i, (im, label) in enumerate(zip(train_images, train_labels)):
    if i % 100 == 99:
      print(
        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
        (i + 1, loss / 100, num_correct)
      )
      loss = 0
      num_correct = 0

      step_arr.append(i+1)
      loss_arr.append(loss/100)
      accuracy_arr.append(num_correct)

    l, acc = train(im, label, lr)
    loss += l
    num_correct += acc

# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
  _, l, acc = forward(im, label)
  loss += l
  num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)

result_dict = {'step' : step_arr ,
               'loss' : loss_arr,
               'accuracy' : accuracy_arr,
               'epoch' : epoch_num,
               'learning_rate': lr}

with open(f'result_lr{lr}_epoch{epoch_num}.pkl') as f:
   pickle.dump(result_dict,f)