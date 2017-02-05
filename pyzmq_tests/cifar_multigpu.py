# Using synchronous version of EASGD

import zmq
import time
from multiprocessing import Process
import numpy as np

def init_worker(nb_epoch=10, batch_size=250):
    '''
    Define Keras model to be trained in a worker process
    Loads MNIST data
    '''
    import keras #import here to adapt env
    from keras.datasets import cifar10
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.utils import np_utils
    
    nb_classes = 10

    # input image dimensions
    img_rows, img_cols = 32, 32
    # The CIFAR10 images are RGB.
    img_channels = 3

    # The data, shuffled and split between train and test sets:
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    gen = ImageDataGenerator().flow(X_train, Y_train, 
            batch_size=batch_size,
            shuffle=True) #used to shuffle data between epochs
    
    return model, gen, X_test, Y_test

def worker_process(port=9000, w_id=0, alpha=0.001):
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
    nb_epoch = 10
    batch_size = 250
    model, gen, X_test, Y_test = init_worker(nb_epoch, batch_size)
    
    # lines for plotting
    times, losses = [], []
    
    for e in range(nb_epoch):
        n_batches = 50000//batch_size
        
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
                open("cifar_gpu_{}.pkl".format(w_id), 'wb'))
        
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
    nb_epoch = 10
    batch_size = 250
    n_samples = 50000
    n_batches = n_samples//batch_size
    
    init = True
    
    for e in range(nb_epoch):
        eb_time = time.time()
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
        print("epoch duration : {:.1f}".format(time.time() - eb_time))
                    
if __name__ == "__main__":
    controller()