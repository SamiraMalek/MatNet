import numpy as np

folder_path = '/mnt/6a35bb5c-3c75-400e-a2fa-146fee20967e/home/fcssaleh/Weighted_Belief_Propagation/BCH_127_64'
H = np.load(folder_path+'/H_BCH_127_64.npy')
G = np.load(folder_path+'/G_BCH_127_64.npy')

r_code, n_code = np.shape(H)
k_code = n_code - r_code
batch_size = 200

code_rate = 1.0*k_code/n_code

# init the AWGN
start_snr = 1
stop_snr = 8
word_seed = 1234
noise_seed = 1234
batches_for_val = np.array([5,5,10,20,20,30,50,300]) 
snr_db = np.arange(start_snr,stop_snr+1,1,dtype=np.float32)
snr_lin = 10.0**(snr_db/10.0)
scaling_factor = np.sqrt(1.0/(2.0*snr_lin*code_rate)) 
random = np.random.RandomState(noise_seed) 
wordRandom = np.random.RandomState(word_seed)
   
x_v = np.zeros([1,n_code], dtype=np.float32)
nun0_x_v = np.zeros([1,n_code], dtype=np.float32)
nun0_y_v = np.zeros([1,n_code], dtype=np.float32)

n_snr = -1
for k_sf in scaling_factor:
    n_snr += 1
    for j in range(batches_for_val[n_snr]):
  
        random_code = wordRandom.randint(0, 2, size=(batch_size, k_code))  
        nun0_y = np.dot(random_code, G) % 2
        nun0_x = random.normal(0.0,1.0,[batch_size,n_code])*k_sf + (-1)**(1-nun0_y)
        nun0_x_llr = 2*nun0_x/(k_sf**2)
        
        nun0_x_v = np.vstack((nun0_x_v,nun0_x_llr)) 
        nun0_y_v = np.vstack((nun0_y_v,nun0_y))
        np.save('x.npy',nun0_x_v[1:,:]) 
        np.save('y.npy',nun0_y_v[1:,:]) 
    print('SNR_',n_snr+1,'Completed')



















