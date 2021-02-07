import numpy as np
H = np.load('/mnt/6a35bb5c-3c75-400e-a2fa-146fee20967e/home/fcssaleh/Weighted_Belief_Propagation/BCH_63_51/H_BCH_63_51.npy')

r_code, n_code = np.shape(H)
k_code = n_code - r_code
batch_size = 200

code_rate = 1.0*k_code/n_code

# init the AWGN
start_snr = 1
stop_snr = 8
noise_seed = 1234  
batches_for_val = np.array([5,7,7,10,15,500,800,7500])  
snr_db = np.arange(start_snr,stop_snr+1,1,dtype=np.float32)
snr_lin = 10.0**(snr_db/10.0)
scaling_factor = np.sqrt(1.0/(2.0*snr_lin*code_rate)) 
random = np.random.RandomState(noise_seed) 
   
x_v = np.zeros([1,n_code], dtype=np.float32)


n_snr = -1
for k_sf in scaling_factor:
    n_snr += 1
    for j in range(batches_for_val[n_snr]):
        x = random.normal(0.0,1.0,[batch_size,n_code])*k_sf + 1.0
        x_llr = 2*x/(k_sf**2)
        x_v = np.vstack((x_v,x_llr))   
        np.save('x.npy',x_v[1:,:]) 
    print('SNR_',n_snr+1,'completed.')

