class MLP:
    
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(hidden_size,input_size) / np.sqrt(input_size)
        self.b1 = np.random.randn(hidden_size,1) / np.sqrt(hidden_size)
        self.W2 = np.random.randn(hidden_size,hidden_size) / np.sqrt(hidden_size)
        self.b2 = np.random.randn(hidden_size,1) / np.sqrt(hidden_size)
        self.W3 = np.random.randn(output_size,hidden_size) / np.sqrt(hidden_size)
        self.b3 = np.random.randn(output_size,1) / np.sqrt(output_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def ReLU(self,Z):
        return np.maximum(0,Z)

    def derivative_ReLU(self,Z):
        return (Z > 0).astype(float)

    def Softmax(self,z):
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    def cross_entropy(self,y_true,y_pred):
        return -np.log(y_pred[y_true])

    def forward_propagation(self,x,y):

        Z1 = self.W1.dot(x) + self.b1 
        A1 = self.ReLU(Z1) 

        Z2 = self.W2.dot(A1) + self.b2 
        A2 = self.ReLU(Z2) 

        Z3 = self.W3.dot(A2) + self.b3 
        A3 = self.Softmax(Z3) 

        prediction = A3[:,0]
        error = self.cross_entropy(y,prediction)

        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}

        return prediction, error, cache



    def back_propagation(self,x,y,cache):

        Z1 = cache["Z1"]
        A1 = cache["A1"]
        Z2 = cache["Z2"]
        A2 = cache["A2"]
        Z3 = cache["Z3"]
        A3 = cache["A3"]

        E = np.zeros((self.output_size,1)) 
        E[y] = 1 
        dZ3 = A3 - E 
        dW3 = dZ3.dot(A2.T) 
        db3 = dZ3 

        dA2 = self.W3.T.dot(dZ3) 
        dZ2 = dA2 * self.derivative_ReLU(Z2) 
        dW2 = dZ2.dot(A1.T) 
        db2 = dZ2 

        dA1 = self.W2.T.dot(dZ2) 
        dZ1 = dA1 * self.derivative_ReLU(Z1) 
        dW1 = dZ1.dot(x.T) 
        db1 = dZ1 

        grads = {"dW1": dW1, "db1": db1,
                "dW2": dW2, "db2": db2,
                "dW3": dW3, "db3": db3}

        return grads


    def update_parameters(self,grads,alpha):

        self.W1 -= alpha * grads["dW1"]
        self.b1 -= alpha * grads["db1"]

        self.W2 -= alpha * grads["dW2"]
        self.b2 -= alpha * grads["db2"]

        self.W3 -= alpha * grads["dW3"]
        self.b3 -= alpha * grads["db3"]

    def compute_loss(self,X,Y):

        loss = 0

        for i in range(len(X)):
            x = X[i].reshape((self.input_size, 1))
            y = Y[i]
            _, error,_=self.forward_propagation(x,y)
            loss += error

        return loss / len(X)

    def fit(self,X,Y,num_epochs,batch_size,alpha,schedule=False):

      def learning_rate_decay(base_rate,current_epoch,total_epochs,schedule=False):
          if schedule == True:
              return base_rate * (0.95 ** (current_epoch // total_epochs))
          else:
              return base_rate

      num_batches_per_epoch=int(len(X)/batch_size)
      loss_history=[]
      test_history=[]
      train_history=[]

      for epoch in range(num_epochs):

          shuffled_indices=np.random.permutation(len(X))
          X_shuffled=X[shuffled_indices]
          Y_shuffled=Y[shuffled_indices]

          for i in range(0,len(X),batch_size):

              X_batch=X_shuffled[i:i+batch_size]
              Y_batch=Y_shuffled[i:i+batch_size]

              batch_loss=0

              for j in range(len(X_batch)):

                  x=X_batch[j].reshape((self.input_size, 1))
                  y=Y_batch[j]

                  prediction,error,cache=self.forward_propagation(x,y)
                  grads=self.back_propagation(x,y,cache)
                  batch_loss+=error

                  current_learning_rate=learning_rate_decay(alpha,(epoch*num_batches_per_epoch+i+j)/num_batches_per_epoch,num_epochs,schedule)

                  self.update_parameters(grads,current_learning_rate)

              batch_loss/=len(X_batch)

              print("Epoch {}, Batch {}/{}: Loss={:.4f}".format(epoch+1,i//batch_size+1,num_batches_per_epoch,batch_loss))

          epoch_loss=self.compute_loss(X,Y)
          test_acc=self.evaluate_acc(x_test,y_test)
          train_acc=self.evaluate_acc(x_train,y_train)

          print("Epoch {}: Loss={:.4f}, Test Accuracy={:.4f}, Train Accuracy={:.4f}".format(epoch+1,epoch_loss,test_acc,train_acc))

          loss_history.append(epoch_loss)
          test_history.append(test_acc)
          train_history.append(train_acc)

      return loss_history,test_history,train_history 


    def predict(self,X):
      predictions=[]
      for i in range(len(X)):
          x=X[i].reshape((self.input_size, 1))
          prediction,_ ,_=self.forward_propagation(x,None)
          predictions.append(np.argmax(prediction))
      return predictions

    def evaluate_acc(self,X,Y):
      predictions=self.predict(X)
      return sum(predictions==Y)/len(Y)
