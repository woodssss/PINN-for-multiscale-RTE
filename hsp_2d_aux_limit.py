import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import scipy.special as sc
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.models import load_model


class stdst():
    # given Ny hsp auxiliary problem solution, we now need to train a Gamma(z,y,theta)
    # since we will need the \partial_y Gamma
    # we only need to make Gamma match with the value on given grid points
    def __init__(self, layers, epsi, Nx_f, Ny_f, Nv_f, x_f, y_f, v_f, lx, ly, lv, Tset, Gamma_train, dtype, optimizer, num_ad_epochs, num_bfgs_epochs):
        self.layers = layers
        self.dtype=dtype
        self.epsi, self.Nx_f, self.Ny_f, self.Nv_f = epsi, Nx_f, Ny_f, Nv_f
        self.lx, self.ly, self.lv = lx, ly, lv
        self.xx, self.vv = np.meshgrid(x_f, v_f)
        # convert np array to tensor
        self.x_f, self.y_f, self.v_f = tf.convert_to_tensor(x_f, dtype=self.dtype), tf.convert_to_tensor(y_f, dtype=self.dtype), tf.convert_to_tensor(v_f, dtype=self.dtype)  # x_f and v_f are grid on x and v direction
        self.x_train, self.y_train, self.v_train = tf.convert_to_tensor(Tset[:,[0]], dtype=self.dtype), tf.convert_to_tensor(Tset[:,[1]], dtype=self.dtype), tf.convert_to_tensor(Tset[:,[2]], dtype=self.dtype) # x_train and v_train are input trainning set for NN

        self.Tset = tf.convert_to_tensor(Tset, dtype=self.dtype)

        self.Gamma_train = tf.convert_to_tensor(Gamma_train, dtype=self.dtype)

        self.num_ad_epochs, self.num_bfgs_epochs = num_ad_epochs, num_bfgs_epochs

        self.optimizer = optimizer

        self.max = 3*np.pi

        self.stop = 0.01





        # Initialize NN
        self.nn = tf.keras.Sequential()
        self.nn.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
        # normalise the input variable
        self.nn.add(tf.keras.layers.Lambda(
            lambda X: (X + lv) / lv - 1.0))
        for width in layers[1:-1]:
            self.nn.add(tf.keras.layers.Dense(
                width, activation=tf.nn.tanh,
                kernel_initializer='glorot_normal'))



        def my_act(x):
            return tf.nn.sigmoid(x)*self.max



        self.nn.add(tf.keras.layers.Dense(
            layers[-1], activation=my_act,
            kernel_initializer='glorot_normal'))

        # Computing the sizes of weights/biases for future decomposition
        self.sizes_w = []
        self.sizes_b = []
        for i, width in enumerate(layers):
            if i != 1:
                self.sizes_w.append(int(width * layers[1]))
                self.sizes_b.append(int(width if i != 0 else layers[1]))




    # define loss function
    def get_loss(self):
        Gamma_pred = self.nn(self.Tset)
        J = tf.reduce_mean(tf.square(Gamma_pred - self.Gamma_train))
        return J


    # define gradient of loss function for optimization step
    def get_grad(self):
        with tf.GradientTape() as tape:
            loss= self.get_loss()

        return loss, tape.gradient(loss, self.nn.trainable_variables)

    # need some functions for extracting weights from NN
    def get_weights(self):
        w = []
        for layer in self.nn.layers[1:]:
            weights_biases = layer.get_weights()
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            w.extend(weights)
            w.extend(biases)
        return tf.convert_to_tensor(w, dtype=self.dtype)

    def set_weights(self, w):
        for i, layer in enumerate(self.nn.layers[1:]):
            start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
            end_weights = sum(self.sizes_w[:i + 1]) + sum(self.sizes_b[:i])
            weights = w[start_weights:end_weights]
            w_div = int(self.sizes_w[i] / self.sizes_b[i])
            weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
            biases = w[end_weights:end_weights + self.sizes_b[i]]
            weights_biases = [weights, biases]
            layer.set_weights(weights_biases)

    def fit(self):
        start_time = time.time()
        for epoch in range(self.num_ad_epochs):
            loss, grad = self.get_grad()
            elapsed = time.time() - start_time
            if epoch % 200 == 0:
                loss = self.get_loss()
                print('Epoch: %d, Loss: %.3e, Time: %.2f' %
                      (epoch, loss, elapsed))
            if loss<self.stop:
                print('training finished')
                loss= self.get_loss()
                print('loss', loss)
                break
            self.optimizer.apply_gradients(zip(grad, self.nn.trainable_variables))

        def loss_and_flat_grad(w):
            # since we are using l-bfgs, the built-in function require
            # value_and_gradients_function
            with tf.GradientTape() as tape:
                self.set_weights(w)
                #loss = self.get_loss()
                loss= self.get_loss()

            grad = tape.gradient(loss, self.nn.trainable_variables)
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)

            return loss, grad_flat

        tfp.optimizer.lbfgs_minimize(
            loss_and_flat_grad,
            initial_position=self.get_weights(),
            max_iterations=self.num_bfgs_epochs,
            num_correction_pairs=10,
            tolerance=1e-8)
        final_loss = self.get_loss()
        print('Final loss is %.3e' % final_loss)


    def predict(self):
        Gamma = self.nn(self.Tset)
        return Gamma




    def save(self, model_name):
        self.nn.save(model_name)









if __name__ == "__main__":
    # Initialize, let x \in [0,1], v \in [-1, 1]
    lx, lv = 10, 1
    Nv_f, Nx_f = 30, 200

    ly = 1
    Ny_f = 100
    # layers for NN, take x,v as input variable, output f^* and f^n+1
    layers = [3, 40, 40, 40, 1]

    # define training set
    # the auxiliary problem is a 1-d hsp
    # [x_f, theta_f] for pde, since we need to evaluate integral, here we use quadrature rule
    x_f = np.linspace(0, lx, Nx_f).T[:, None]
    y_f = np.linspace(-ly, ly, Ny_f).T[:, None]

    # define quadrature points and weights
    points, weights = np.polynomial.legendre.leggauss(Nv_f)
    points = (points + 1) * np.pi
    weights = weights * np.pi
    v_f, weights_vf = np.float32(points[:, None]), np.float32(weights[:, None])
    v_f = np.reshape(v_f, (Nv_f, 1))
    weights_vf = np.reshape(weights_vf, (Nv_f, 1))

    # build training set for g(x,y,v_x,v_y) and give trainning sample
    def my_act(x):
        return tf.nn.sigmoid(x) * np.max(3*np.pi)


    spv = np.ones((Nv_f, 1))
    skx = np.ones((Nx_f, 1))

    x_tmp = np.kron(x_f, spv)
    v_tmp = np.kron(skx, v_f)

    Txv_set = np.concatenate([x_tmp, v_tmp], axis=1)

    sk = np.ones((Ny_f, 1))
    x_train = np.kron(sk, x_tmp)
    v_train = np.kron(sk, v_tmp)

    sp = np.ones((Nx_f*Nv_f, 1))
    y_train = np.kron(y_f, sp)

    Tset = np.concatenate([x_train, y_train, v_train], axis=1)



    Gamma_train = np.array([[]])

    for i in range(Ny_f):
        tmp_mdl_name = 'half_space_aux_with_lx10_Ny_100_i_' + str(int(i)) + '.h5'
        tmp_mdl = load_model(tmp_mdl_name, custom_objects={"my_act": my_act})
        G_tmp = tmp_mdl(Txv_set)
        #print('ss',G_tmp.shape, Gamma_train.shape)
        #Gamma_train = np.concatenate([Gamma_train, G_tmp], axis=0)
        Gamma_train = np.append(Gamma_train, G_tmp)

    Gamma_train = Gamma_train[:,None]



    # define parameter for model
    epsi = 1
    dtype = tf.float32
    num_ad_epochs = 30000
    num_bfgs_epochs = 8000
    # define adam optimizer

    optimizer = tf.keras.optimizers.Adam(
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08)



    # define model
    mdl = stdst(layers, epsi, Nx_f, Ny_f, Nv_f, x_f, y_f, v_f, lx, ly, lv, Tset, Gamma_train, dtype, optimizer, num_ad_epochs, num_bfgs_epochs)

    # train model
    mdl.fit()

    model_name = 'half_space_2d_aux_Gamma_with_lx' + str(int(lx))  + '.h5'
    mdl.save('mdls/' + model_name)











