from numpy import cross, eye, dot
from scipy.linalg import expm, norm
import math
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def M(axis, theta):
    return expm(cross(eye(3), axis/norm(axis)*theta))

# v, axis, theta = [1,0,0], [0,0,1], math.pi
# M0 = M(axis, theta)
# print(dot(M0,v))

if __name__ == "__main__":
    scale = 25
    input_path = f'processored/Scannetpp_crop_scaled/0a7cc12c0e/pred_normal'
    output_path = f'processored/Scannetpp_crop_scaled/0a7cc12c0e/pred_normal_{scale}'
    npz_list = sorted(Path(input_path).glob('*.npz'))
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for npz in npz_list:
        print(npz.name)
        npz_np = np.load(npz)['arr_0']
        shape = npz_np.shape
        random_axis = np.random.rand(3)
        random_axis /= np.linalg.norm(random_axis)
        theta = math.radians(np.random.uniform(-scale, scale))
        M0 = M(random_axis, theta)
        npz_np = npz_np.reshape(-1, 3)
        npz_np = npz_np.transpose()
        npz_np = M0 @ npz_np
        npz_np = npz_np.transpose()
        npz_np = npz_np.reshape(*shape)
        output_file_path = output_path / npz.name
        np.savez(output_file_path, npz_np)
        
        pred_norm_rgb = ((npz_np + 1) * 0.5) * 255
        pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
        pred_norm_rgb = pred_norm_rgb.astype(np.uint8)                  # (B, H, W, 3)

        target_path = '%s/%s.png' % (output_path, npz.name[:-4])
        plt.imsave(target_path, pred_norm_rgb[0, :, :, :])


    pass