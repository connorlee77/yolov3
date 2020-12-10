from scipy.io import loadmat

x = loadmat('bdd_out_cam/person/b1c66a42-6f7d68ca.mat')
print(x['cam'].shape)