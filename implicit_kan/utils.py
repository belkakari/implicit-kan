import random

import numpy as np
import torch


def get_grid(h, w, b=0, norm=True, device="cpu"):
    if norm:
        xgrid = np.linspace(0, w, num=w) / w
        ygrid = np.linspace(0, h, num=h) / h
    else:
        xgrid = np.linspace(0, w, num=w)
        ygrid = np.linspace(0, h, num=h)
    xv, yv = np.meshgrid(xgrid, ygrid, indexing="xy")
    grid = np.stack([xv, yv], axis=-1)[None]

    grid = torch.from_numpy(grid).float().to(device)
    if b > 0:
        grid = grid.expand(b, -1, -1, -1)  # [Batch, H, W, UV]
        return grid.permute(0, 3, 1, 2)  # [Batch, UV, H, W]
    else:
        return grid[0].permute(2, 0, 1)  # [UV, H, W]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
