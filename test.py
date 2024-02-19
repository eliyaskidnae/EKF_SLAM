import numpy as np

x = [   np.array([ [ -40.00287756],
            [  4.97197045]]),   
        np.array([[-4.97486593],
                    [  40.00389625]]),  
        np.array([[-4.96150404],
                    [  25.00786367]]),  
        np.array([[-19.9995227 ],
                    [  3.00883563]])]

zk = np.array([[9],[9]])

y = np.array([[0], [1], [2], [3]]) 
z = np.array([[0], [1], [2], [3]])


m = np.block([[y],[z]])

print(m)

for i in range(len(x)):

    zk = np.block([[zk],[x[i]]])

print(zk)