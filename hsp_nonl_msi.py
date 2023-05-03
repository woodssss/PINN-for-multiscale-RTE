import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import sys
import scipy.special as sc



class stdst():
    def __init__(self, epsi, dt, T, Nx_f, Nv_f, x_f, v_f, lx, lv, Bd_weight, Tset, Train_BC_L, Train_BC_R, fL_train, fR_train, TL, nbc, weights, dx, f0, dtype, optimizer, num_ad_epochs, num_bfgs_epochs, file_name, nur, nug, nlr,nlg):
        self.dtype=dtype
        self.epsi, self.T, self.dt, self.Nx_f, self.Nv_f = epsi, T, dt, Nx_f, Nv_f
        self.lx, self.lv, self.dx = tf.convert_to_tensor(lx, dtype=self.dtype)*tf.ones((1,1)), lv, dx
        self.Bd_weight = Bd_weight
        self.xx, self.vv = np.meshgrid(x_f, v_f)
        self.nbc=nbc

        # number of layers for rho and g
        self.nlr, self.nlg = nlr, nlg

        # number of neuron for each layer
        self.nur, self.nug = nur, nug

        self.stop = 0.005

        self.file_name = file_name
        # convert np array to tensor
        self.x_f, self.v_f = tf.convert_to_tensor(x_f, dtype=self.dtype), tf.convert_to_tensor(v_f, dtype=self.dtype)  # x_f and v_f are grid on x and v direction
        self.x_train, self.v_train = tf.convert_to_tensor(Tset[:,[0]], dtype=self.dtype), tf.convert_to_tensor(Tset[:,[1]], dtype=self.dtype) # x_train and v_train are input trainning set for NN

        self.Tset = tf.convert_to_tensor(Tset, dtype=self.dtype)

        self.weights = tf.convert_to_tensor(weights, dtype=self.dtype)

        self.num_ad_epochs, self.num_bfgs_epochs = num_ad_epochs, num_bfgs_epochs

        self.optimizer = optimizer

        self.f0 = tf.convert_to_tensor(f0, dtype=self.dtype)

        # define BC
        self.Train_BC_L = tf.convert_to_tensor(Train_BC_L, dtype=self.dtype)
        self.Train_BC_R = tf.convert_to_tensor(Train_BC_R, dtype=self.dtype)

        self.TL = tf.convert_to_tensor(TL, dtype=self.dtype)

        self.fL_train = tf.convert_to_tensor(fL_train, dtype=self.dtype)
        self.fR_train = tf.convert_to_tensor(fR_train, dtype=self.dtype)

        self.x_Rb = self.lx * tf.ones((self.Nv_f,1), dtype=self.dtype)

        # test
        #self.limit_test()

        # Initialize NN
        self.nn = self.get_nn()
        self.nn.summary()

        # Computing the sizes of weights/biases for future decomposition
        self.sizes_w = []
        self.sizes_b = []

        for layer in self.nn.layers[2:]:
            weights_biases = layer.get_weights()
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            #print('ssssss', int(weights.shape[0]), int(biases.shape[0]))
            self.sizes_w.append(int(weights.shape[0]))
            self.sizes_b.append(int(biases.shape[0]))



    def get_nn(self):

        # define nn for rho
        input_rho = tf.keras.Input(shape=(1,))

        input_rho_mid = tf.keras.layers.Dense(units=self.nur, activation=tf.nn.tanh, kernel_initializer='glorot_normal')(
            input_rho)

        for i in range(self.nlr-1):
            input_rho_mid = tf.keras.layers.Dense(units=self.nur, activation=tf.nn.tanh, kernel_initializer='glorot_normal')(
                input_rho_mid)

        output_rho = tf.keras.layers.Dense(units=1, activation=None, kernel_initializer='glorot_normal')(
            input_rho_mid)

        # define nn for g

        input_g = tf.keras.Input(shape=(2,))

        input_g_mid = tf.keras.layers.Dense(units=self.nug, activation=tf.nn.tanh, kernel_initializer='glorot_normal')(
            input_g)

        for i in range(self.nlg-1):
            input_g_mid = tf.keras.layers.Dense(units=self.nug, activation=tf.nn.tanh, kernel_initializer='glorot_normal')(
                input_g_mid)

        output_g = tf.keras.layers.Dense(units=1, activation=None, kernel_initializer='glorot_normal')(
            input_g_mid)

        model = tf.keras.Model(inputs=[input_rho, input_g], outputs=[output_rho, output_g])

        return model


    def get_intv(self, f):

        sp = tf.ones([self.Nv_f, 1])

        sk = tf.linspace(np.float32(0), self.Nx_f - 1, self.Nx_f, name="linspace")

        sk = tf.reshape(sk, (self.Nx_f, 1))

        id = self.Kron_TF(sk, sp)

        id = tf.cast(id, tf.int32)

        id = tf.reshape(id, [self.Nx_f * self.Nv_f])

        dup_p = tf.constant([self.Nx_f, 1], tf.int32)

        weights_rep = tf.tile(self.weights, dup_p)

        res = tf.math.segment_sum(weights_rep * f, id)

        #print('ssp', weights_rep.shape, res.shape)

        return res


    def get_pde1(self):
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and t
            tape.watch(self.x_train)
            tape.watch(self.v_train)
            tape.watch(self.x_f)
            # Packing together the inputs
            Train = tf.stack([self.x_train[:, 0], self.v_train[:, 0]], axis=1)

            # Getting the prediction
            F_T, F_I = self.nn([self.x_f, Train])

            F_I_x = tape.gradient(F_I, self.x_train)

        F_T_p4 = tf.pow(F_T, 4)

        F_T_p4_vec = self.get_rho_vec(F_T_p4)

        pde1 = self.v_train*F_I_x - F_T_p4_vec + F_I

        # print('sp', pde1.shape, int_vg.shape, T_xx.shape)

        return pde1


    def get_pde2(self):
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and t
            tape.watch(self.x_train)
            tape.watch(self.v_train)
            tape.watch(self.x_f)
            # Packing together the inputs
            Train = tf.stack([self.x_train[:, 0], self.v_train[:, 0]], axis=1)

            # Getting the prediction
            F_T, F_I = self.nn([self.x_f, Train])

            F_T_x = tape.gradient(F_T, self.x_f)

        F_T_xx = tape.gradient(F_T_x, self.x_f)

        F_T_p4 = tf.pow(F_T, 4)

        avg_F_I = self.get_intv(F_I)*0.5

        pde2 = F_T_xx + avg_F_I - F_T_p4

        # print('sp', pde2.shape, rho_x_vec.shape, int_vg_vec.shape)

        return pde2


    def get_pde3(self):

        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and t
            tape.watch(self.x_train)
            tape.watch(self.v_train)
            tape.watch(self.x_f)
            # Packing together the inputs
            Train = tf.stack([self.x_train[:, 0], self.v_train[:, 0]], axis=1)

            # Getting the prediction
            F_T, F_I = self.nn([self.x_f, Train])

            F_T_x = tape.gradient(F_T, self.x_f)

        int_v_FI = self.get_intv(self.v_train*F_I)*0.5

        pde3 = int_v_FI - F_T_x

        # print('sp', pde3.shape, T.shape, rho.shape)

        return pde3



    def get_rho_vec(self, rho):

        sp=tf.ones((self.Nv_f,1))

        rho_vec = self.Kron_TF(rho, sp)

        rho_vec = tf.reshape(rho_vec, [self.Nx_f * self.Nv_f, 1])

        return rho_vec



    def get_f_bc_loss(self):
        F_T_L, F_I_L = self.nn([tf.zeros((1,1)), self.Train_BC_L])

        BC1 = tf.reduce_mean(tf.square(F_T_L - self.TL))

        BC2 = tf.reduce_mean(tf.square(F_I_L - self.fL_train))

        BC= BC1 +BC2

        # print('bcsp',  fL.shape, BC.shape, BC1.shape, rhoL.shape, rhoL_vec.shape)
        # fsdfsdfsd
        return BC

    def get_bc_d_loss(self):

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.x_Rb)
            tape.watch(self.v_f)
            tape.watch(self.lx)
            Train_Rb = tf.concat([self.x_Rb, self.v_f], axis=1)

            F_T_R, F_I_R = self.nn([self.lx, Train_Rb])

            F_T_R_x = tape.gradient(F_T_R, self.lx)

            F_I_R_x = tape.gradient(F_I_R, self.x_Rb)

            F_I_R_v = tape.gradient(F_I_R, self.v_f)

        BC1 = tf.reduce_mean(tf.square(F_T_R_x))

        BC2 = tf.reduce_mean(tf.square(F_I_R_x))

        BC3 = tf.reduce_mean(tf.square(F_I_R_v))

        BC = BC1 + BC2 +BC3

        return BC



    # define loss function
    def get_loss(self):
        # loss function contains 3 parts: PDE ( converted to IC), BC and Mass conservation
        # pde
        pde1 = self.get_pde1()

        pde2 = self.get_pde2()

        pde3 = self.get_pde3()

        J1 = tf.reduce_sum(self.get_intv(tf.square(pde1)))*self.dx

        J2 = tf.reduce_sum(tf.square(pde2))*self.dx

        J3 = tf.reduce_sum(tf.square(pde3))*self.dx

        # BC rho
        J4 = self.get_f_bc_loss()

        loss = J1 + J2 + J3 + self.Bd_weight*J4
        return loss, J1, J2, J3, J4



    def Kron_TF(self, A, B):
        A_shape = A.get_shape()
        B_shape = B.get_shape()

        for i in range(A_shape[0]):
            for j in range(A_shape[1]):
                if j == 0:
                    temp = tf.squeeze(A[i, j] * B)

                else:
                    temp = tf.concat([temp, tf.squeeze(A[i, j] * B)], 1)
            if i == 0:
                result = temp
            else:
                result = tf.concat([result, temp], 0)
        return result


    # define gradient of loss function for optimization step
    def get_grad(self):
        with tf.GradientTape() as tape:
            loss, J1, J2, J3, J4 = self.get_loss()

        return loss, tape.gradient(loss, self.nn.trainable_variables)

    # Extracting weights from NN, and use as initial weights for L-BFGS
    def get_weights(self):
        w = []
        #print('wsp',len(self.nn.trainable_variables), tf.shape_n(self.nn.trainable_variables))
        self.nn.summary()
        for layer in self.nn.layers[2:]:
            weights_biases = layer.get_weights()
            #print('wbsp', len(weights_biases), weights_biases)
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            w.extend(weights)
            w.extend(biases)
        return tf.convert_to_tensor(w, dtype=self.dtype)

    # Update weights every step in L-BFGS
    def set_weights(self, w):
        for i, layer in enumerate(self.nn.layers[2:]):
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
                print('Epoch: %d, Loss: %.3e, Time: %.2f' %
                      (epoch, loss, elapsed))
                loss, J1, J2, J3, J4 = self.get_loss()
                print('loss 1-5', J1, J2, J3, J4)
                with open(self.file_name, 'a') as fw:
                    print('This is epsi=', self.epsi, ' Nx=', self.Nx_f, file=fw)
                    print('Epoch: %d, Loss: %.3e, Time: %.2f' %
                          (epoch, loss, elapsed), file=fw)
                    print('loss=', loss, 'J1-J4 are ', J1, J2, J3, J4, file=fw)

                F_T, F_I = self.predict()
                #print('ss', rho_pred.shape, rho_vec_pred.shape, g_pred.shape)
                plt.figure(1)
                plt.plot(self.x_f, F_T, 'r-o')
                plt.title('F_T')
                fig = plt.figure(2)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(self.xx, self.vv, F_I.numpy().reshape(self.Nx_f, self.Nv_f).T)
                plt.title('F_I')
                plt.show(block=False)
                loss, J1, J2, J3, J4 = self.get_loss()
                print('loss 1-5', J1, J2, J3, J4)

            if loss<self.stop:
                print('training finished')
                loss, J1, J2, J3, J4 = self.get_loss()
                print('loss 1-5', loss, J1, J2, J3, J4)
                F_T, F_I = self.predict()
                # print('ss', rho_pred.shape, rho_vec_pred.shape, g_pred.shape)
                plt.figure(1)
                plt.plot(self.x_f, F_T, 'r-o')
                plt.title('F_T')
                fig = plt.figure(2)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(self.xx, self.vv, F_I.numpy().reshape(self.Nx_f, self.Nv_f).T)
                plt.title('F_I')
                plt.show(block=False)
                break

            self.optimizer.apply_gradients(zip(grad, self.nn.trainable_variables))

        def loss_and_flat_grad(w):
            # since we are using l-bfgs, the built-in function require
            # value_and_gradients_function
            with tf.GradientTape() as tape:
                self.set_weights(w)
                loss , J1, J2, J3, J4= self.get_loss()

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

        F_T, F_I = self.predict()
        # print('ss', rho_pred.shape, rho_vec_pred.shape, g_pred.shape)
        plt.figure(1)
        plt.plot(self.x_f, F_T, 'r-o')
        plt.title('F_T')
        fig = plt.figure(2)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.xx, self.vv, F_I.numpy().reshape(self.Nx_f, self.Nv_f).T)
        plt.title('F_I')

        plt.show(block=False)


        final_loss , J1, J2, J3, J4= self.get_loss()
        print('Final loss is %.3e' % final_loss)

        with open(self.file_name, 'a') as fw:
            print('This is epsi=', self.epsi, ' Nx=', self.Nx_f, file=fw)
            print('Final loss=', final_loss, 'J1-J3 are ', J1, J2, J3, J4, file=fw)


    def predict(self):
        F_T, F_I = self.nn([self.x_f, self.Tset])
        return F_T, F_I

    def save(self, model_name):
        self.nn.save(model_name)









if __name__ == "__main__":
    # input parameters
    Bd_weight = 2
    Nx_f = 800
    nlr = 4
    nlg = 4
    nur = 50
    nug = 50

    # Initialize, let x \in [0,1], v \in [-1, 1]
    lx, lv, T = 10, 1, 1

    # define training set
    # [x_f, v_f] for pde, since we need to evaluate integral, here we use quadrature rule
    Nv_f= 60

    x_f = np.linspace(0 , lx , Nx_f).T[:, None]
    
    dx = 1/(Nx_f-1)

    points, weights = np.polynomial.legendre.leggauss(Nv_f)
    points = lv * points
    weights = lv * weights
    v_f, weights_vf = np.float32(points[:, None]), np.float32(weights[:, None])

    Tset = np.ones((Nx_f * Nv_f, 2))  # Training set, the first column is x and the second column is v

    for i in range(Nx_f):
        Tset[i * Nv_f:(i + 1) * Nv_f, [0]] = x_f[i][0] * np.ones_like(v_f)
        Tset[i * Nv_f:(i + 1) * Nv_f, [1]] = v_f

    # [x_0, v_0] for IC
    x_0, v_0 = x_f, v_f
    rho_0 = 1+0.5*np.sin(2*np.pi*x_0)
    temp_M0 = np.exp(-15*(v_0-0.5)**2) + np.exp(-15*(v_0+0.5)**2)
    #temp_M0 = np.exp(-15 * (v_0) ** 2)
    f_0_vec=np.zeros((Nx_f*Nv_f,1))
    for i in range(Nx_f*Nv_f):
        f_0_vec[i] = rho_0[math.floor(i/Nv_f)]*temp_M0[i%Nv_f]


    # For BC, there are two BCs, for v>0 and for v<0
    nbc=60
    v_bc_pos, v_bc_neg = np.random.rand(1, nbc).T, -np.random.rand(1, nbc).T
    #v_bc_pos, v_bc_neg = np.linspace(0, lv, nbc).T[:, None], np.linspace(-lv, 0, nbc).T[:, None]
    x_bc_pos, x_bc_zeros = np.ones((nbc,1)), np.zeros((nbc,1))

    Train_BC_L = np.float32(np.concatenate((x_bc_zeros, v_bc_pos), axis=1))
    Train_BC_R = np.float32(np.concatenate((x_bc_pos, v_bc_neg), axis=1))

    fL_train = np.float32(5*np.sin(v_bc_pos))
    fR_train = 0

    TL = 1

    plt.plot(v_bc_pos, fL_train, 'ro')
    plt.show()

    #print('bc', Train_BC_L, Train_BC_R)

    # define parameter for model
    dt=0.1
    f0=f_0_vec
    dtype = tf.float32
    num_ad_epochs = 10001
    num_bfgs_epochs = 10000
    # define adam optimizer
    train_steps = 5
    lr_fn = tf.optimizers.schedules.PolynomialDecay(1e-3, train_steps, 1e-5, 2)
    optimizer = tf.keras.optimizers.Adam(
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08)



    # save model
    def num2str_deciaml(x):
        s=str(x)
        c=''
        for i in range(len(s)):
            if s[i]=='0':
                c = c + 'z'
            elif s[i]=='.':
                c = c + 'p'
            elif s[i]=='-':
                c = c + 'n'
            else:
                c = c + s[i]

        return c


    epsi = 1


    file_name = 'NLRTE_hsp_epsi_' + num2str_deciaml(epsi) + '_jw_' + str(Bd_weight) + '_Nx_' + str(Nx_f) + '_nlr_' + str(nlr) + '_nlg_' + str(nlg)+ '_nur_' + str(nur)+ '_nug_' + str(nug) + '.' + 'txt'

    # define model
    mdl = stdst(epsi, dt, T, Nx_f, Nv_f, x_f, v_f, lx, lv, Bd_weight, Tset, Train_BC_L, Train_BC_R, fL_train,
                fR_train, TL, nbc, weights_vf, dx, f0, dtype, optimizer, num_ad_epochs, num_bfgs_epochs, file_name, nur, nug, nlr,nlg)

    # train model
    mdl.fit()



    model_name = 'NLRTE_hsp_epsi_' + num2str_deciaml(epsi) + '_jw_' + str(Bd_weight) + '_Nx_' + str(Nx_f) + '_nlr_' + str(nlr) + '_nlg_' + str(nlg)+ '_nur_' + str(nur)+ '_nug_' + str(nug) + '.' +  'h5'
    mdl.save('mdls/' + model_name)

    F_T, F_I =mdl.predict()

    F_I_pred = F_I.numpy().reshape(Nx_f,Nv_f)


    #test_f0 = f_0_vec.reshape(Nx_f,Nv_f)

    xx,vv=np.meshgrid(x_f,v_f)

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, vv, F_I_pred.T)
    plt.title('f')

    plt.figure(2)
    plt.plot(x_f, F_I, 'r-o')









