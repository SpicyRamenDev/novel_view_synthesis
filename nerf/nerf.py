import json
import numpy as np

def split_transforms(path, dest):
    transforms_path = path / 'transforms.json.bak'
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
    transforms['aabb_scale'] = 0.5
    frames = transforms['frames']
    p = np.array([
        [1,  0,  0,  0],
        [0,  0, -1,  0],
        [0,  1,  0,  0],
        [0,  0,  0,  1]
    ])
    for frame in frames:
        m = frame['transform_matrix']
        m = np.matmul(p, m).tolist()
        frame['transform_matrix'] = m
        frame['image_path'] = str(path / frame['image_path'])
    with open(dest / 'transforms_train.json', 'w') as f:
        transforms['frames'] = frames[:4]
        json.dump(transforms, f, indent=4)
    with open(dest / 'transforms_val.json', 'w') as f:
        transforms['frames'] = frames[:16]
        json.dump(transforms, f, indent=4)
    with open(dest / 'transforms_test.json', 'w') as f:
        transforms['frames'] = frames[:16]
        json.dump(transforms, f, indent=4)
