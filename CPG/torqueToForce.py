import numpy as np
def torqueToForce(cpg, torques):
#UNTITLED3 Summary of this function goes here
#   Detailed explanation goes here
	forces = np.zeros([3,6])

	J = cpg['xmk'].getLegJacobians(cpg['legs'])
	torques = np.reshape(torques, [3,6]);
	for leg in range(6): 
		Jl = J[0:3,:,leg]
		# print("!!!!!!",np.shape(Jl))
		# print("######",np.shape(torques[:,leg]))
		# print("11111111111111",np.shape(np.linalg(Jl.T)@torques[:,leg]))

		# forces[:,leg] = (np.linalg.inv(Jl.T)@torques[:,leg]);  
		forces[:,leg] = np.dot(Jl,torques[:,leg])

	return forces    
