import trimesh
import numpy as np
def produce(input, npz, output):
    pass
    scale_mat = np.load(npz)['scale_mat_000000']
    print(scale_mat)
    mesh = trimesh.load(input)
    mesh.apply_transform(np.linalg.inv(scale_mat))
    mesh.export(output)
if __name__ == "__main__":
    pass
    case_list = ['24', '37', '40']
    for case in case_list:
        produce(f'result/{case}/monosdf.ply', f'../../../dataset/processored/DTU/scan{case}/cameras_sphere.npz', f'result/{case}/monosdf_align.ply')
        produce(f'result/{case}/monosdf+.ply', f'../../../dataset/processored/DTU/scan{case}/cameras_sphere.npz', f'result/{case}/monosdf+_align.ply')