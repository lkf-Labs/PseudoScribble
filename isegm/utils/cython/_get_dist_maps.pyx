import numpy as np
cimport cython
cimport numpy as np
from libc.stdlib cimport malloc, free

ctypedef struct qnode:
    int row
    int col
    int layer
    int orig_row
    int orig_col

@cython.infer_types(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def get_dist_maps(np.ndarray[np.float32_t, ndim=2, mode="c"] points,
                  int height, int width, float norm_delimeter):
    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] dist_maps = \
        np.full((2, height, width), 1e6, dtype=np.float32, order="C")
    #x y不断变
    cdef int *dxy = [-1, 0, 0, -1, 0, 1, 1, 0]
    cdef int i, j, x, y, dx, dy
    cdef qnode v
    cdef qnode *q = <qnode *> malloc((4 * height * width + 1) * sizeof(qnode))
    cdef int qhead = 0, qtail = -1
    cdef float ndist
    #正、负点击分开来求
    #points = 2n×3
    for i in range(points.shape[0]):
        x, y = round(points[i, 0]), round(points[i, 1])
        if x >= 0:
            qtail += 1
            q[qtail].row = x
            q[qtail].col = y
            q[qtail].orig_row = x
            q[qtail].orig_col = y
            #正点击为偶数 负点击为奇数 2n×3
            if i >= points.shape[0] / 2:
                q[qtail].layer = 1
            else:
                q[qtail].layer = 0
            dist_maps[q[qtail].layer, x, y] = 0
    #遍历每个圆心
    while qtail - qhead + 1 > 0:
        #圆心
        v = q[qhead]
        qhead += 1

        for k in range(4):
            x = v.row + dxy[2 * k]
            y = v.col + dxy[2 * k + 1]
            #计算每个像素点与圆心的距离
            #只有当像素点与本次圆心的距离更小时才需要更新
            ndist = ((x - v.orig_row)/norm_delimeter) ** 2 + ((y - v.orig_col)/norm_delimeter) ** 2
            if (x >= 0 and y >= 0 and x < height and y < width and
                dist_maps[v.layer, x, y] > ndist):
                qtail += 1
                q[qtail].orig_col = v.orig_col
                q[qtail].orig_row = v.orig_row
                q[qtail].layer = v.layer
                q[qtail].row = x
                q[qtail].col = y
                dist_maps[v.layer, x, y] = ndist

    free(q)
    # 2 ×350 ×740
    return dist_maps
