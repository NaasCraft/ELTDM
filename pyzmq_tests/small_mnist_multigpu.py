# Using synchronous version of EASGD

import zmq
import time
from multiprocessing import Process
import numpy as np

def init_worker(nb_epoch=8, batch_size=100):
    '''
    Define Keras model to be trained in a worker process
    Loads MNIST data
    '''
    import keras #import here to adapt env
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.utils import np_utils
    from keras import backend as K
    from keras.preprocessing.image import ImageDataGenerator
    
    nb_classes = 10

    # input image dimensions
    img_rows, img_cols = 28, 28
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    
    gen = ImageDataGenerator().flow(X_train, Y_train, 
            batch_size=batch_size,
            shuffle=True) #used to shuffle data between epochs
    
    return model, gen, X_test, Y_test

def worker_process(port=9000, w_id=0, alpha=0.1):
    print("Initializing worker on port {}".format(port))
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:{}".format(port))
    
    np.random.seed(w_id*7) 
    ## careful, seeds should be different 
    ## if different behaviors are expected between workers
    
    import os
    os.environ['THEANO_FLAGS']='mode=FAST_RUN,floatX=float32,device=cuda{}'.format(w_id)
    
    # adjust here
    nb_epoch = 12
    batch_size = 300
    model, gen, X_test, Y_test = init_worker(nb_epoch, batch_size)
    
    # lines for plotting
    times, losses = [], []
    
    for e in range(nb_epoch):
        n_batches = 60000//batch_size
        
        for i in range(n_batches):
            X_b, y_b = next(gen)

            # wait for controller to send center
            center = socket.recv_pyobj()
            # init case
            if center is None:
                center = model.get_weights()
                init_time = time.time()

            loss, acc = model.train_on_batch(X_b, y_b)
            losses.append(loss)
            times.append(time.time()-init_time)

            # update weight with center variable
            weights = model.get_weights()

            for j in range(len(weights)):
                weights[j] = weights[j] - alpha * (weights[j] - center[j])
            
            # send asap
            socket.send_pyobj(weights)
            
            model.set_weights(weights)

        loss, acc = model.evaluate(X_test,Y_test,batch_size=batch_size,verbose=0)
        print("---- (Worker {}) ----".format(w_id))
        print("Epoch {}\tLoss {:.3f}\tAcc {:.2f}%".format(e+1, loss, acc*100))
        print("---------------------")
    
    import pickle
    pickle.dump({'dt':times, 'loss':losses}, 
                open("sml_gpu_{}.pkl".format(w_id), 'wb'))
        
def controller(port=9000, n_workers=2):
    print("Initializing controller...")
    
    # parameters
    alpha = 0.1
    
    context = zmq.Context()
    
    ports = [port+k for k in range(n_workers)]
    
    # init workers and corresponding sockets (one per worker)
    workers = [context.socket(zmq.REQ) for k in range(n_workers)]
    
    for w_id in range(n_workers):
        w_port = ports[w_id]
        Process(target=worker_process, args=(w_port,w_id,alpha,)).start()
        w_sock = workers[w_id]
        w_sock.connect('tcp://localhost:{}'.format(w_port))
        
    # begin steps
    nb_epoch = 12
    batch_size = 300
    n_samples = 60000
    n_batches = n_samples//batch_size
    
    init = True
    
    for e in range(nb_epoch):
        for i in range(n_batches):
            
            for w_id in range(n_workers):
                # send request with current center to each worker
                worker = workers[w_id]

                if init:
                    worker.send_pyobj(None)
                else:
                    worker.send_pyobj(center)
                    
            weights_list = []
            for w_id in range(n_workers):
                # wait for each answer to perform update
                worker = workers[w_id]
                weights = worker.recv_pyobj()
                weights_list.append(weights)

            # compute new center value
            if init: #center is still undefined
                n_weights = len(weights)
                center = []
                for i in range(n_weights):
                    temp = weights_list[0][i]
                    for w_id in range(1,n_workers):
                        temp = temp + weights_list[w_id][i]
                    
                    temp = temp / n_workers
                    center.append(temp)
                init = False
            else:
                for i in range(n_weights):
                    temp = weights_list[0][i] - center[i]
                    
                    for w_id in range(1,n_workers):
                        temp = temp + weights_list[w_id][i] - center[i]
                        
                    center[i] = center[i] + alpha * temp
                    
if __name__ == "__main__":
    controller()