import tensorflow as tf
import matplotlib.pyplot as pltf
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

folder_path = '/mnt/6a35bb5c-3c75-400e-a2fa-146fee20967e/home/fcssaleh/Weighted_Belief_Propagation/BCH_63_51'
H = np.load(folder_path+'/H_BCH_63_51.npy')

H = np.array(H,dtype=np.float32)
_H = np.ones(H.shape,dtype=np.float32) - H
#zero_x_v = np.load(folder_path+'/test_246000.npy')

r_code, n_code = np.shape(H)
k_code = n_code - r_code
code_rate = 1.0*k_code/n_code

degree_check = np.sum(H,axis=1)
max_degree_check = np.max(degree_check)
max_degree_check  = max_degree_check.astype('int32')
extended_H = np.tile(H,(1,max_degree_check))
extended_H = np.reshape(extended_H,[r_code,max_degree_check,n_code])
mask1 = np.zeros([r_code,max_degree_check,n_code],dtype=np.float32)
for i in range(r_code):
    k = 0
    for j in range(n_code):
        if H[i,j] == 1:
            mask1[i,k,j] = 1.
            k +=1
    extended_H[i,k:max_degree_check,:] = 0.     
mask2 = extended_H - mask1
mask3 = np.ones(extended_H.shape,dtype=np.float32) - mask2  

num_iteration = 6
batch_size = 200
noise_seed = 345000 
random = np.random.RandomState(noise_seed) 

snr_dBrange = np.array([4,4.5,5,5.5,6,6.5,7,7.5,8,8.5])
snr_print = np.array([1,2,3,4,5,6])
snr_lin = 10.0**(snr_dBrange/10.0)
sigma = np.sqrt(1.0/(2.0*snr_lin*code_rate))
sigma = np.array([sigma for _ in range(20)])
sigma = np.reshape(sigma,[-1])
sigma_power2 = sigma**2
num_of_batch = 10000000
batch_in_epoch = 2000
batches_for_val = np.array([5,20,30,50,100,200,400,5000])
batches_for_val = batches_for_val.astype('int32')

var_dict = {}
check_dict = {}
weight_var_dict = {}
weight_check_dict = {}
bias_var_dict = {}

x = tf.placeholder(tf.float32, shape=[batch_size, n_code])
y = tf.placeholder(tf.float32, shape=[batch_size, n_code])
y_test = tf.placeholder(tf.float32, shape=[batch_size, n_code])
threshold = 0.99999997

# l = 0 
x_tile = tf.tile(x,multiples=[1,r_code])
x_tile_reshape = tf.reshape(x_tile,[batch_size,r_code,n_code])
x_tile_reshape = tf.multiply(H,x_tile_reshape)
v_l0 = tf.tanh(0.5*x_tile_reshape)

i = 0
#weight_check_dict["wc_l{0}".format(i)] = tf.Variable(mask1)
#weight_check_dict["wc_l{0}".format(i)] = tf.multiply(weight_check_dict["wc_l{0}".format(i)],mask1)
check_dict["c_l{0}".format(i)] = tf.tile(v_l0,multiples=[1,1,max_degree_check])
check_dict["c_l{0}".format(i)] = tf.reshape(check_dict["c_l{0}".format(i)],[batch_size,r_code,max_degree_check,n_code])
check_dict["c_l{0}".format(i)] = tf.multiply(mask2,check_dict["c_l{0}".format(i)]) + mask3 #+ weight_check_dict["wc_l{0}".format(i)]
check_dict["c_l{0}".format(i)] = tf.reduce_prod(check_dict["c_l{0}".format(i)],axis=3)
check_dict["c_l{0}".format(i)] = tf.tile(check_dict["c_l{0}".format(i)],multiples=[1,1,n_code])
check_dict["c_l{0}".format(i)] = tf.reshape(check_dict["c_l{0}".format(i)],[batch_size,r_code,n_code,max_degree_check])
check_dict["c_l{0}".format(i)] = tf.transpose(check_dict["c_l{0}".format(i)],perm=[0,1,3,2])
check_dict["c_l{0}".format(i)] = tf.multiply(mask1,check_dict["c_l{0}".format(i)])
check_dict["c_l{0}".format(i)] = tf.reduce_sum(check_dict["c_l{0}".format(i)],axis=2)
check_dict["c_l{0}".format(i)] = tf.clip_by_value(check_dict["c_l{0}".format(i)],-threshold ,threshold)
check_dict["c_l{0}".format(i)] = 2 * tf.atanh(check_dict["c_l{0}".format(i)])
for i in range(1,num_iteration):
    
    #bias_var_dict["bv_vl{0}".format(i)] = tf.Variable(np.ones([n_code],dtype=np.float32))
    bias_var_dict["bv_vl{0}".format(i)] = tf.Variable(H)
    weight_var_dict["wv_l{0}".format(i)] = tf.Variable(H)
    weight_var_dict["wv_l{0}".format(i)] = tf.multiply(weight_var_dict["wv_l{0}".format(i)],H)
    bias_var_dict["bv_vl{0}".format(i)] = tf.multiply(bias_var_dict["bv_vl{0}".format(i)],H)
    t = tf.multiply(weight_var_dict["wv_l{0}".format(i)],check_dict["c_l{0}".format(i-1)])
    var_dict["v_l{0}".format(i)] = tf.reduce_sum(t,axis=1) #+ tf.multiply(bias_var_dict["bv_vl{0}".format(i)],x)
    var_dict["v_l{0}".format(i)] = tf.tile(var_dict["v_l{0}".format(i)],multiples=[1,r_code])
    var_dict["v_l{0}".format(i)] = tf.reshape(var_dict["v_l{0}".format(i)],[batch_size,r_code,n_code])    
    var_dict["v_l{0}".format(i)] = var_dict["v_l{0}".format(i)] - check_dict["c_l{0}".format(i-1)]
    var_dict["v_l{0}".format(i)] = var_dict["v_l{0}".format(i)] + tf.multiply(bias_var_dict["bv_vl{0}".format(i)],x_tile_reshape)
    var_dict["v_l{0}".format(i)] = tf.multiply(H,var_dict["v_l{0}".format(i)]) 
    var_dict["v_l{0}".format(i)] = tf.tanh(0.5*var_dict["v_l{0}".format(i)])
    
    #weight_check_dict["wc_l{0}".format(i)] = tf.Variable(mask1)
    #weight_check_dict["wc_l{0}".format(i)] = tf.multiply(weight_check_dict["wc_l{0}".format(i)],mask1)
    check_dict["c_l{0}".format(i)] = tf.tile(var_dict["v_l{0}".format(i)],multiples=[1,1,max_degree_check])
    check_dict["c_l{0}".format(i)] = tf.reshape(check_dict["c_l{0}".format(i)],[batch_size,r_code,max_degree_check,n_code])
    check_dict["c_l{0}".format(i)] = tf.multiply(mask2,check_dict["c_l{0}".format(i)]) + mask3 #+weight_check_dict["wc_l{0}".format(i)]
    check_dict["c_l{0}".format(i)] = tf.reduce_prod(check_dict["c_l{0}".format(i)],axis=3)
    check_dict["c_l{0}".format(i)] = tf.tile(check_dict["c_l{0}".format(i)],multiples=[1,1,n_code])
    check_dict["c_l{0}".format(i)] = tf.reshape(check_dict["c_l{0}".format(i)],[batch_size,r_code,n_code,max_degree_check])
    check_dict["c_l{0}".format(i)] = tf.transpose(check_dict["c_l{0}".format(i)],perm=[0,1,3,2])
    check_dict["c_l{0}".format(i)] = tf.multiply(mask1,check_dict["c_l{0}".format(i)])
    check_dict["c_l{0}".format(i)] = tf.reduce_sum(check_dict["c_l{0}".format(i)],axis=2)
    check_dict["c_l{0}".format(i)] = tf.clip_by_value(check_dict["c_l{0}".format(i)],-threshold ,threshold)
    check_dict["c_l{0}".format(i)] = 2 * tf.atanh(check_dict["c_l{0}".format(i)])
    
b_out = tf.Variable(np.ones([n_code],dtype=np.float32))
w_out = tf.Variable(H)
w_out = tf.multiply(H,w_out)
out = tf.multiply(w_out,check_dict["c_l5"])
out = tf.reduce_sum(out,axis=1) + tf.multiply(b_out,x)    
#sig_out = tf.nn.sigmoid(out["l5"])
error = tf.cast(out > 0,'float')
error = tf.math.abs(error - y_test)
BER_one_batch = tf.reduce_mean(error)

cost = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=out,labels=y))
#train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
saver = tf.train.Saver()

loss_v = np.zeros([1, 1], dtype=np.float32)
ber_val = np.zeros([8], dtype=np.float32)
BER = np.zeros([1,8], dtype=np.float32)

with tf.Session() as sess:    
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, folder_path+'/Saved_Model/my_model')
    X_test = np.load(folder_path+'/Saved_Model/TestData/x.npy')
    Y_test = np.zeros([batch_size,n_code], dtype=np.float32)
    batches_for_val = np.array([5,7,7,10,15,500,800,7500])
    vector_number = 0
    for i in range(8):
        BER_sum_batchs = 0
        for j in range(batches_for_val[i]): 
            start = batch_size*j + vector_number
            stop = batch_size*(j+1) + vector_number
            x_v_j = X_test[start:stop,:]
            BER_j = sess.run(fetches = BER_one_batch ,feed_dict={x:x_v_j,y_test:Y_test})
            BER_sum_batchs = BER_j +BER_sum_batchs
        ber_val[i] = BER_sum_batchs / batches_for_val[i]
        vector_number = vector_number+batches_for_val[i]*batch_size
    print('SNR[dB] validation - [1,2,3,4,5,6,7,8]')
    print('BER validation - ', ber_val) 


