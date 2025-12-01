import numpy as np



fx = 1200.0   
fy = 1200.0   
cx = 640.0    
cy = 360.0    
# Baseline between the two camera positions (meters)
BASELINE = 0.40   

K_left = np.array([[fx,   0, cx],
                   [ 0, fy, cy],
                   [ 0,   0,  1]], dtype=np.float32)

K_right = K_left.copy()
