import tensorflow as tf 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist 
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

labels = {0 : 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 
          5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'}

def plot_label_per_class(data): 
    f, ax = plt.subplots(1,1, figsize=(12,4)) 
    g = sns.countplot(data, order = labels.values(), palette='YlGn') 
    g.set_title('Number of labels for each class') 
    plt.show()

    plot_label_per_class(y_train)

    plt.figure(figsize=(10,10)) 
    for i in range(9): 
        plt.subplot(3,3,i+1) 
        plt.imshow(x_train[i],cmap=plt.cm.binary) 
        plt.xlabel(labels[y_train[i]]) 
        plt.show()

x_train = x_train.reshape(60000,28*28) / 255.0 
x_test = x_test.reshape(10000,28*28) / 255.0 
input_size=28*28
hidden_size=128
output_size=10

model=MLP(input_size,hidden_size,output_size) 
loss_history,test_history,train_history=model.fit(x_train,y_train,num_epochs=10,batch_size=64,alpha=0.01,schedule=True)

plt.plot(loss_history,label='Loss') 
plt.plot(test_history,label='Test Accuracy')
plt.plot(train_history,label='Train Accuracy') 
plt.legend() 
plt.xlabel('Epoch') 
plt.ylabel('Value') 
plt.title('Learning Curves') 
plt.show()
