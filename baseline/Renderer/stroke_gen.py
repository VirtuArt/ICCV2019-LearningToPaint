from typing import Tuple, List, Union, Any

import cv2
import numpy as np

def normal(x, width):
    return (int)(x * (width - 1) + 0.5)

def draw(f, width=128):
    x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = f
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1
    x0 = normal(x0, width * 2)
    x1 = normal(x1, width * 2)
    x2 = normal(x2, width * 2)
    y0 = normal(y0, width * 2)
    y1 = normal(y1, width * 2)
    y2 = normal(y2, width * 2)
    z0 = (int)(1 + z0 * width // 2)
    z2 = (int)(1 + z2 * width // 2)
    canvas = np.zeros([width * 2, width * 2]).astype('float32')
    tmp = 1. / 100
    for i in range(100):
        t = i * tmp
        x = (int)((1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2)
        y = (int)((1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2)
        z = (int)((1-t) * z0 + t * z2)
        w = (1-t) * w0 + t * w2
        cv2.circle(canvas, (y, x), z, w, -1)
    return 1 - cv2.resize(canvas, dsize=(width, width))


def draw_graphite(f_: Union[List[Any], Tuple[Any]],
                  width: int = 512) -> np.core.ndarray:
    """
    Draw a Graphite stroke on top of a clean canvas

    :param f_: (x0, y0, x1, y1, x2, y2, r0, r2, c0, c2, smoothness). s.t r_i are the radii at the edges, c_i represent the colors
              All values are normalized by the canvas size
    :param width: the canvas width (pixels)
    :return:
    """
    x0, y0, x1, y1, x2, y2, r0, r2, c0, c2, smoothness = f_
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1
    x0 = normal(x0, width * 2)
    x1 = normal(x1, width * 2)
    x2 = normal(x2, width * 2)
    y0 = normal(y0, width * 2)
    y1 = normal(y1, width * 2)
    y2 = normal(y2, width * 2)
    r0 = (int)(1 + r0 * width // 2)
    r2 = (int)(1 + r2 * width // 2)
    canvas = np.zeros([width * 2, width * 2]).astype('float32')
    tmp = 1. / 400
    for i in range(400):
        t = i * tmp
        x = (int)((1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2)
        y = (int)((1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2)
        radius = (int)((1-t) * r0 + t * r2)
        color = (1-t) * c0 + t * c2
        cv2.circle(img=canvas,
                   center=(y, x),
                   radius=radius,
                   color=color,
                   thickness=-1,  # fill the circle with the specified color
                   lineType=cv2.LINE_AA,  # antialiased line
                   )
        # smoothness_mask = (2 * (1-smoothness) * np.random.rand(canvas.shape) + smoothness)
    return (2 * (1-smoothness) * np.random.rand(width, width) + smoothness) * (1 - cv2.resize(canvas, dsize=(width, width)))


import matplotlib.pyplot as plt


def blend_strokes(canvas1: np.core.ndarray, canvas2: np.core.ndarray, mode: str = "darken") -> np.core.ndarray:
    assert canvas1.shape == canvas2.shape, f"both canvases must share the same size, {canvas1.shape} != {canvas2.shape}"
    if mode == "darken":
        return np.minimum(canvas1, canvas2)
    else:
        raise NotImplementedError(f"{mode} blending mode is not supported.")


if __name__ == '__main__':
    # vec = np.random.uniform(0,1.,10)
    # res = draw(vec)

    p0 = (0.2, 0.8)
    p1 = (0, 0.35)
    p2 = (0.5, 0.3)

    p00 = (0.2, 0.5)
    p11 = (0, 0.35)
    p22 = (0.5, 0.8)

    radius = (5/512, 20 / 512)  # (2/512, 4 / 512)  # start, end
    colors = (50, 150)  # start, end
    smoothness = 0.7

    f = (*p0, *p1, *p2, *radius, *colors, smoothness)

    res = draw_graphite(f)

    radius2 = [17/512, 17/512]
    colors2 = [160, 160]
    f2 = (*p00, *p11, *p22, *radius2, *colors2, smoothness)
    res2 = draw_graphite(f2)

    merged_res = blend_strokes(res, res2, mode="darken")
    fig, axes = plt.subplots(1,3, figsize=(45, 15))
    # axes[0].imshow(np.random.rand(512,512) * res, cmap="gray")
    # axes[1].imshow(np.random.rand(512, 512) * res2, cmap="gray")
    axes[0].imshow(res, cmap="gray")
    axes[1].imshow(res2, cmap="gray")
    axes[2].imshow(merged_res, cmap="gray")
    plt.show()
    plt.imsave("/Users/eyalziv/my_projects/VirtuArt/ICCV2019-LearningToPaint/baseline/Renderer/graphite_blending.png",
               np.hstack([res, res2, merged_res]), cmap="gray")
