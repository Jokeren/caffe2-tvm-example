from __future__ import division
from topi.util import get_const_int, get_const_tuple, const_matrix
import numpy as np
from topi.nn.util import get_const_int, get_pad_tuple
from topi.nn import pad
import tvm
from tvm import autotvm
import collections


MNTiles = [
    (6, 8)
]
KTile = 256
ARCH = "neon"
BITCODE_PATHS = [
    "/Users/kerenzhou/Codes/tvm-samples/autotune/gemm__neon.bc"
]

@tvm.register_func("tvm_callback_llvm_bitcode_path")
def bitcode_paths():
    global BITCODE_PATHS
    return BITCODE_PATHS

# Tensorized
def intrin_gemm(M, N, K):
    assert (M, N) in MNTiles, (M, N)
    dtype = 'float32'
    A = tvm.placeholder((K, M), dtype=dtype, name='A')
    B = tvm.placeholder((K, N), dtype=dtype, name='B')
    k = tvm.reduce_axis((0, K), name='k')
    C = tvm.compute((M, N), lambda m, n:
                    tvm.sum(A[k, m] * B[k, n], axis=[k]), name='C')

    Ab = tvm.decl_buffer(A.shape, A.dtype,
                        name="A",
                        offset_factor=M,
                        strides=[M, 1])
    Bb = tvm.decl_buffer(B.shape, B.dtype,
                        name="B",
                        offset_factor=N,
                        strides=[N, 1])
    Cb = tvm.decl_buffer(C.shape, C.dtype,
                        name="C",
                        offset_factor=1,
                        strides=[tvm.var('ldc'), 1])

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]

        def body():
            irb = tvm.ir_builder.create()
            extern_call = tvm.call_extern(
                "int32",
                "sgemm_compute_{M}x{N}__{ARCH}".format(M=M, N=N, ARCH=ARCH),
                K,
                irb.buffer_ptr(aa),
                aa.elem_offset,
                irb.buffer_ptr(bb),
                bb.elem_offset,
                irb.buffer_ptr(cc),
                cc.elem_offset,
                cc.strides[0])
            irb.emit(extern_call)
            return irb.get()

        def reset():
            irb = tvm.ir_builder.create()
            extern_call = tvm.call_extern(
                "int32",
                "sgemm_reset_{M}x{N}__{ARCH}".format(M=M, N=N, ARCH=ARCH),
                irb.buffer_ptr(cc),
                cc.elem_offset,
                cc.strides[0])
            irb.emit(extern_call)
            return irb.get()

        def update():
            irb = tvm.ir_builder.create()
            extern_call = tvm.call_extern(
                "int32",
                "sgemm_update_{M}x{N}__{ARCH}".format(M=M, N=N, ARCH=ARCH),
                K,
                irb.buffer_ptr(aa),
                aa.elem_offset,
                irb.buffer_ptr(bb),
                bb.elem_offset,
                irb.buffer_ptr(cc),
                cc.elem_offset,
                cc.strides[0])
            irb.emit(extern_call)
            return irb.get()
        return body(), reset(), update()

    with tvm.build_config():
        return tvm.decl_tensor_intrin(C.op,
                                      intrin_func,
                                      binds={A: Ab, B: Bb, C: Cb})

def decl_winograd(cfg, data, kernel, strides, padding, layout, out_dtype, VK=6, VP=8, packed_output=False):
    # return _baseline_winograd(cfg, data, kernel, strides, padding, layout, out_dtype)
    N, CI, IH, IW = get_const_tuple(data.shape)
    CO, _, KH, KW = get_const_tuple(kernel.shape)
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)

    assert layout == 'NCHW'
    assert KH == 3 and KW == 3 and HPAD == 1 and WPAD == 1 and HSTR == 1 and WSTR == 1
    data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")


    A_data = np.array(
        [[1,  1,  1,   1,    1,    32,      32,    0],
         [0,  1,  -1,  2,   -2,  16,   -16,   0],
         [0,  1,  1,   4,    4,   8,    8,   0],
         [0,  1,  -1,  8,   -8,   4,   -4,   0],
         [0,  1,  1,   16,  16,   2,  2,   0],
         [0,  1,  -1,  32,  -32,  1,  -1,  1]],
        dtype=np.float32).T
    G_data = np.array(
        [[1,      0,     0],
         [-2/9,  -2/9,   -2/9],
         [-2/9,   2/9,   -2/9],
         [1/90,  1/45,   2/45],
         [1/90,  -1/45,  2/45],
         [1/45,    1/90, 1/180],
         [1/45,   -1/90, 1/180],
         [0,      0,     1]],
        dtype=np.float32)
    B_data = np.array(
        [[1,   0,    -21/4,    0,    21/4,     0,    -1,  0],
         [0,   1,      1,    -17/4,  -17/4,    1,    1,   0],
         [0,   -1,     1,    17/4,   -17/4,   -1,    1,   0],
         [0,  1/2,    1/4,   -5/2,   -5/4,     2,    1,   0],
         [0,  -1/2,   1/4,    5/2,   -5/4,    -2,    1,   0],
         [0,   2,      4,    -5/2,    -5,     1/2,   1,   0],
         [0,   -2,     4,     5/2,    -5,    -1/2,   1,   0],
         [0,   -1,     0,    21/4,     0,    -21/4,  0,   1]],
        dtype=np.float32).T

    m = A_data.shape[1]
    r = 3
    alpha = m + r - 1

    C = CI

    H = (IH + 2 * HPAD - 3) // HSTR + 1
    W = (IW + 2 * WPAD - 3) // WSTR + 1
    nH, nW = (H + m-1) // m, (W + m-1) // m

    def round_up(a, b): return ((a + b - 1) // b) * b
    K = round_up(CO, VK)
    P = round_up(N * nH * nW, VP)

    assert K % VK == 0
    assert P % VP == 0

    G = const_matrix(G_data, 'G')
    r_kh = tvm.reduce_axis((0, KH), 'r_kh')
    r_kw = tvm.reduce_axis((0, KW), 'r_kw')
    assert K >= CO
    if K > CO:
        kernel_pad = pad(kernel, (0, 0, 0, 0), (K - CO, 0, 0, 0), name="kernel_pad")
    else:
        kernel_pad = kernel
    input_tile = tvm.placeholder(shape=(P // VP, C, alpha, alpha, VP), dtype='float32', name="input_tile")
    U = tvm.placeholder(shape=(K // VK, alpha, alpha, C, VK), dtype='float32', name="U")
    #U = tvm.compute(
    #    (K // VK, alpha, alpha, C, VK), lambda k, eps, nu, c, kk:
    #    tvm.sum(kernel_pad[k * VK + kk][c][r_kh][r_kw].astype(out_dtype) *
    #            G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]), name='U')


    ## pack input tile
    #input_tile = tvm.compute((P // VP, C, alpha, alpha, VP),
    #                         lambda b, c, eps, nu, bb:
    #                         data_pad[(b*VP+bb) // (nH*nW)][c][(b*VP+bb) // nW % nH * m + eps]
    #                         [(b*VP+bb) % nW * m + nu],
    #                         name='d')

    def compute_B_T_dot_X(b, c, eps, nu, bb):
        temp_expr = {}
        for j in range(alpha):
            wd0 = input_tile[b][c][0][j][bb] - input_tile[b][c][6][j][bb]
            d4_sub_d2 = input_tile[b][c][4][j][bb] - input_tile[b][c][2][j][bb]
            wd7 = input_tile[b][c][7][j][bb] - input_tile[b][c][1][j][bb]
            d3_sub_d5 = input_tile[b][c][3][j][bb] - input_tile[b][c][5][j][bb]
            wd1 = input_tile[b][c][2][j][bb] + input_tile[b][c][6][j][bb]
            wd2 = input_tile[b][c][1][j][bb] + input_tile[b][c][5][j][bb]
            wd4 = input_tile[b][c][5][j][bb] + input_tile[b][c][1][j][bb] * 0.25
            wd5 = input_tile[b][c][6][j][bb] - input_tile[b][c][4][j][bb] * 5
            wd3 = input_tile[b][c][6][j][bb] + input_tile[b][c][2][j][bb] * 0.25
            wd6 = input_tile[b][c][1][j][bb] + input_tile[b][c][5][j][bb] * 0.25

            wd0 = wd0 + d4_sub_d2 * 5.25
            wd7 = wd7 + d3_sub_d5 * 5.25

            wd1 = wd1 - input_tile[b][c][4][j][bb] * 4.25
            wd2 = wd2 - input_tile[b][c][3][j][bb] * 4.25

            wd3 = wd3 - input_tile[b][c][4][j][bb] * 1.25
            wd5 = wd5 + input_tile[b][c][2][j][bb] * 4
            wd4 = wd4 - input_tile[b][c][3][j][bb] * 1.25
            wd6 = wd6 - input_tile[b][c][3][j][bb] * 1.25

            temp_expr[(0, j)] = wd0
            temp_expr[(1, j)] = wd1 + wd2
            temp_expr[(2, j)] = wd1 - wd2
            temp_expr[(3, j)] = wd3 + wd4 * 2
            temp_expr[(4, j)] = wd3 - wd4 * 2
            temp_expr[(5, j)] = wd5 + wd6 * 2
            temp_expr[(6, j)] = wd5 - wd6 * 2
            temp_expr[(7, j)] = wd7

        now = tvm.const(0.0, "float32")
        for ii in range(alpha):
            for jj in range(alpha):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now

    B_T_dot_X = tvm.compute((P // VP, C, alpha, alpha, VP), compute_B_T_dot_X, name="B_T_dot_X")

    def compute_X_dot_B(b, eps, nu, c, bb):
        temp_expr = {}

        for i in range(alpha):
            wd0 = B_T_dot_X[b][c][i][0][bb] - B_T_dot_X[b][c][i][6][bb]
            d4_sub_d2 = B_T_dot_X[b][c][i][4][bb] - B_T_dot_X[b][c][i][2][bb]
            wd7 = B_T_dot_X[b][c][i][7][bb] - B_T_dot_X[b][c][i][1][bb]
            d3_sub_d5 = B_T_dot_X[b][c][i][3][bb] - B_T_dot_X[b][c][i][5][bb]
            wd1 = B_T_dot_X[b][c][i][2][bb] + B_T_dot_X[b][c][i][6][bb]
            wd2 = B_T_dot_X[b][c][i][1][bb] + B_T_dot_X[b][c][i][5][bb]
            wd4 = B_T_dot_X[b][c][i][5][bb] + B_T_dot_X[b][c][i][1][bb] * 0.25
            wd5 = B_T_dot_X[b][c][i][6][bb] - B_T_dot_X[b][c][i][4][bb] * 5
            wd3 = B_T_dot_X[b][c][i][6][bb] + B_T_dot_X[b][c][i][2][bb] * 0.25
            wd6 = B_T_dot_X[b][c][i][1][bb] + B_T_dot_X[b][c][i][5][bb] * 0.25

            wd0 = wd0 + d4_sub_d2 * 5.25
            wd7 = wd7 + d3_sub_d5 * 5.25

            wd1 = wd1 - B_T_dot_X[b][c][i][4][bb] * 4.25
            wd2 = wd2 - B_T_dot_X[b][c][i][3][bb] * 4.25

            wd3 = wd3 - B_T_dot_X[b][c][i][4][bb] * 1.25
            wd5 = wd5 + B_T_dot_X[b][c][i][2][bb] * 4
            wd4 = wd4 - B_T_dot_X[b][c][i][3][bb] * 1.25
            wd6 = wd6 - B_T_dot_X[b][c][i][3][bb] * 1.25

            temp_expr[(i, 0)] = wd0
            temp_expr[(i, 1)] = wd1 + wd2
            temp_expr[(i, 2)] = wd1 - wd2
            temp_expr[(i, 3)] = wd3 + wd4 * 2
            temp_expr[(i, 4)] = wd3 - wd4 * 2
            temp_expr[(i, 5)] = wd5 + wd6 * 2
            temp_expr[(i, 6)] = wd5 - wd6 * 2
            temp_expr[(i, 7)] = wd7

        now = tvm.const(0.0, "float32")
        for ii in range(alpha):
            for jj in range(alpha):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now
    V = tvm.compute((P // VP, alpha, alpha, C, VP), compute_X_dot_B, name="V")

    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute(
        (K // VK, P // VP, alpha, alpha, VK, VP),
        lambda k, b, eps, nu, kk, bb: tvm.sum(U[k][eps][nu][c][kk] * V[b][eps][nu][c][bb], axis=c),
        name='M')

    def compute_A_T_dot_M(k, b, eps, nu, kk, bb):
        temp_expr = {}

        for j in range(alpha):
            m1_add_m2 = M[k][b][1][j][kk][bb] + M[k][b][2][j][kk][bb]
            m1_sub_m2 = M[k][b][1][j][kk][bb] - M[k][b][2][j][kk][bb]
            m3_add_m4 = M[k][b][3][j][kk][bb] + M[k][b][4][j][kk][bb]
            m3_sub_m4 = M[k][b][3][j][kk][bb] - M[k][b][4][j][kk][bb]
            m5_add_m6 = M[k][b][5][j][kk][bb] + M[k][b][6][j][kk][bb]
            m5_sub_m6 = M[k][b][5][j][kk][bb] - M[k][b][6][j][kk][bb]
            s0 = M[k][b][0][j][kk][bb] + m1_add_m2
            s5 = M[k][b][7][j][kk][bb] + m1_sub_m2
            s1 = m1_sub_m2 + m5_sub_m6 * 16
            s4 = m1_add_m2 + m3_add_m4 * 16
            s2 = m1_add_m2 + 8 * m5_add_m6
            s3 = m1_sub_m2 + 8 * m3_sub_m4
            s0 = s0 + m5_add_m6 * 32
            s5 = s5 + m3_sub_m4 * 32
            s1 = s1 + m3_sub_m4 * 2
            s4 = s4 + m5_add_m6 * 2
            s0 = s0 + m3_add_m4
            s5 = s5 + m5_sub_m6
            s2 = s2 + m3_add_m4 * 4
            s3 = s3 + m5_sub_m6 * 4
            temp_expr[(0, j)] = s0
            temp_expr[(1, j)] = s1
            temp_expr[(2, j)] = s2
            temp_expr[(3, j)] = s3
            temp_expr[(4, j)] = s4
            temp_expr[(5, j)] = s5
        now = tvm.const(0.0, "float32")
        for ii in range(m):
            for jj in range(alpha):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now

    A_T_dot_M = tvm.compute((K // VK, P // VP, m, alpha, VK, VP), compute_A_T_dot_M, name="A_T_dot_M")

    def compute_X_dot_A(k, b, eps, nu, kk, bb):
        temp_expr = {}

        for i in range(m):
            m1_add_m2 = A_T_dot_M[k][b][i][1][kk][bb] + A_T_dot_M[k][b][i][2][kk][bb]
            m1_sub_m2 = A_T_dot_M[k][b][i][1][kk][bb] - A_T_dot_M[k][b][i][2][kk][bb]
            m3_add_m4 = A_T_dot_M[k][b][i][3][kk][bb] + A_T_dot_M[k][b][i][4][kk][bb]
            m3_sub_m4 = A_T_dot_M[k][b][i][3][kk][bb] - A_T_dot_M[k][b][i][4][kk][bb]
            m5_add_m6 = A_T_dot_M[k][b][i][5][kk][bb] + A_T_dot_M[k][b][i][6][kk][bb]
            m5_sub_m6 = A_T_dot_M[k][b][i][5][kk][bb] - A_T_dot_M[k][b][i][6][kk][bb]
            s0 = A_T_dot_M[k][b][i][0][kk][bb] + m1_add_m2
            s5 = A_T_dot_M[k][b][i][7][kk][bb] + m1_sub_m2
            s1 = m1_sub_m2 + m5_sub_m6 * 16
            s4 = m1_add_m2 + m3_add_m4 * 16
            s2 = m1_add_m2 + 8 * m5_add_m6
            s3 = m1_sub_m2 + 8 * m3_sub_m4
            s0 = s0 + m5_add_m6 * 32
            s5 = s5 + m3_sub_m4 * 32
            s1 = s1 + m3_sub_m4 * 2
            s4 = s4 + m5_add_m6 * 2
            s0 = s0 + m3_add_m4
            s5 = s5 + m5_sub_m6
            s2 = s2 + m3_add_m4 * 4
            s3 = s3 + m5_sub_m6 * 4
            temp_expr[(i, 0)] = s0
            temp_expr[(i, 1)] = s1
            temp_expr[(i, 2)] = s2
            temp_expr[(i, 3)] = s3
            temp_expr[(i, 4)] = s4
            temp_expr[(i, 5)] = s5
        now = tvm.const(0.0, "float32")
        for ii in range(m):
            for jj in range(m):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now

    Y = tvm.compute((K // VK, P // VP, m, m, VK, VP), compute_X_dot_A, name="Y")

    # unpack output
    def _output(n, k_, h, w):
        b_idx = n * nH * nW + (h//m) * nW + w//m
        b = b_idx // VP
        bb = b_idx % VP
        k = k_ // VK
        kk = k_ % VK
        return Y[k][b][h % m][w % m][kk][bb]

    output = tvm.compute((N, CO, H, W), _output,
                         name='output', tag='winograd_conv_output')

    if cfg:
        cfg.add_flop(2 * N * K * H * W * KH * KW * C)

    return Y, input_tile, U, output

def schedule_winograd(cfg, output, VK=6, VP=8):
    s = tvm.create_schedule(output.op)
    if not cfg:
        print("herehere111")
        return s
    if output.name == "Y":
        Y = output
    else:
        Y = output.op.input_tensors[0]
    A_T_dot_M = Y.op.input_tensors[0]
    M = A_T_dot_M.op.input_tensors[0]
    U, V = M.op.input_tensors
    B_T_dot_X = V.op.input_tensors[0]
    #input_tile = B_T_dot_X.op.input_tensors[0]
    #data_pad = input_tile.op.input_tensors[0]
    # padding

    UNROLL = cfg['unroll'].val
    VECTORIZE = cfg['vectorize'].val
    TENSORIZE = cfg['tensorize'].val

    #if cfg['data_pad_inline'].val:
    #    s[data_pad].compute_inline()

    ## pack input tiles
    #(b, c, eps, nu, bb) = input_tile.op.axis
    #if cfg['input_tile_REORDER_C'].val:
    #    s[input_tile].reorder(b, eps, nu, c, bb)
    #if UNROLL:
    #    [s[input_tile].unroll(ax) for ax in [eps, nu]]
    #if VECTORIZE:
    #    s[input_tile].vectorize(bb)
    #if autotvm.GLOBAL_SCOPE.in_tuning:
    #    s[input_tile].pragma(b, 'debug_skip_region')
    #    s[data_pad].pragma(data_pad.op.axis[0], 'debug_skip_region')
    # s[input_tile].compute_inline()

    # transform kernel
    #if isinstance(U.op, tvm.tensor.ComputeOp):
    #    kernel, G = U.op.input_tensors
    #    if isinstance(kernel.op, tvm.tensor.ComputeOp):
    #        s[kernel].compute_inline()

    #    s[G].compute_inline()
    #    k, eps, nu, c, kk, = s[U].op.axis
    #    # r_kh, r_kw = s[U].op.reduce_axis
    #    # s[U].reorder(k, c, eps, nu, r_kh, r_kw, kk)
    #    # s[U].unroll(eps)
    #    # s[U].unroll(nu)
    #    # s[U].unroll(r_kh)
    #    # s[U].unroll(r_kw)
    #    # s[U].vectorize(kk)
    #    if autotvm.GLOBAL_SCOPE.in_tuning:
    #        # kernel transformation will be pre-computed during compilation, so we skip
    #        # this part to make tuning records correct
    #        s[U].pragma(k, 'debug_skip_region')

    # if autotvm.GLOBAL_SCOPE.in_tuning:
    #     # kernel transformation will be pre-computed during compilation, so we skip
    #     # this part to make tuning records correct
    #     s[output].pragma(s[output].axis[0], 'debug_skip_region')

    (k, b, eps, nu, kk, bb) = A_T_dot_M.op.axis
    s[A_T_dot_M].reorder(b, k, eps, nu, kk, bb)
    if UNROLL:
        [s[A_T_dot_M].unroll(ax) for ax in [eps, nu, kk]]
    if VECTORIZE:
        s[A_T_dot_M].vectorize(bb)

    if cfg['M_COMPUTE_AT'].val:
        s[M].compute_at(s[A_T_dot_M], b)
    (k, b, eps, nu, kk, bb) = Y.op.axis
    s[Y].reorder(b, k, eps, nu, kk, bb)
    if UNROLL:
        [s[Y].unroll(ax) for ax in [eps, nu, kk]]

    if VECTORIZE:
        s[Y].vectorize(bb)

    if cfg['A_T_dot_M_COMPUTE_AT'].val:
        s[A_T_dot_M].compute_at(s[Y], b)

    # Schedule V
    (b, c, eps, nu, bb) = B_T_dot_X.op.axis
    if UNROLL:
        [s[B_T_dot_X].unroll(ax) for ax in [eps, nu]]
    if VECTORIZE:
        s[B_T_dot_X].vectorize(bb)

    # if cfg['B_T_dot_X_REORDER_C'].val:
    #     s[B_T_dot_X].reorder(b, eps, nu, c, bb)

    #if cfg['input_tile_COMPUTE_AT'].val:
    #    s[input_tile].compute_at(s[B_T_dot_X], b)

    (b, eps, nu, c, bb) = V.op.axis
    if UNROLL:
        [s[V].unroll(ax) for ax in [eps, nu]]
    if VECTORIZE:
        s[V].vectorize(bb)
    if cfg['V_REORDER_C'].val:
        s[V].reorder(b, eps, nu, c, bb)
    print('reorder {}'.format(cfg['V_REORDER_C'].val))

    if cfg['B_T_dot_X_COMPUTE_AT'].val:
        s[B_T_dot_X].compute_at(s[V], b)

    (k, b, eps, nu, kk, bb) = M.op.axis
    if cfg['V_COMPUTE_AT'].val:
        s[V].compute_at(s[M], b)
    s[M].reorder(b, k, eps, nu, kk, bb)
    K = get_const_int(M.op.reduce_axis[0].dom.extent)
    s[M].tensorize(kk, intrin_gemm(M=VK, N=VP, K=K))
    return s

A_data = np.array([[1,  1,  1,   1,    1,    32,      32,    0],
                   [0,  1,  -1,  2,   -2,   16,   -16,   0],
                   [0,  1,  1,   4,    4,   8,    8,   0],
                   [0,  1,  -1,  8,   -8,   4,   -4,   0],
                   [0,  1,  1,   16,  16,   2,  2,   0],
                   [0,  1,  -1,  32,  -32,  1,  -1,  1]],
                  dtype=np.float32).T
m = A_data.shape[1]
r = 3
alpha = m + r - 1

HSTR = 1
WSTR = 1
HPAD = 1
WPAD = 1


# s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
# s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
# s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
# s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
# s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
# s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m

def A_T_dot_X(X):
    m1_add_m2 = X[1] + X[2]
    m1_sub_m2 = X[1] - X[2]
    m3_add_m4 = X[3] + X[4]
    m3_sub_m4 = X[3] - X[4]
    m5_add_m6 = X[5] + X[6]
    m5_sub_m6 = X[5] - X[6]
    s0 = X[0] + m1_add_m2
    s5 = X[7] + m1_sub_m2
    s1 = m1_sub_m2 + m5_sub_m6 * 16
    s4 = m1_add_m2 + m3_add_m4 * 16
    s2 = m1_add_m2 + 8 * m5_add_m6
    s3 = m1_sub_m2 + 8 * m3_sub_m4
    s0 = s0 + m5_add_m6 * 32
    s5 = s5 + m3_sub_m4 * 32
    s1 = s1 + m3_sub_m4 * 2
    s4 = s4 + m5_add_m6 * 2
    s0 = s0 + m3_add_m4
    s5 = s5 + m5_sub_m6
    s2 = s2 + m3_add_m4 * 4
    s3 = s3 + m5_sub_m6 * 4

    result = np.zeros((6, s0.size))
    result[0] = s0
    result[1] = s1
    result[2] = s2
    result[3] = s3
    result[4] = s4
    result[5] = s5
    return result

M = None

def decl_output_transform_minimal(cfg, X, M, VK, VP):

    def compute_A_T_dot_M(k, b, eps, nu, kk, bb):
        temp_expr = {}

        for j in range(alpha):
            m1_add_m2 = M[k][b][1][j][kk][bb] + M[k][b][2][j][kk][bb]
            m1_sub_m2 = M[k][b][1][j][kk][bb] - M[k][b][2][j][kk][bb]
            m3_add_m4 = M[k][b][3][j][kk][bb] + M[k][b][4][j][kk][bb]
            m3_sub_m4 = M[k][b][3][j][kk][bb] - M[k][b][4][j][kk][bb]
            m5_add_m6 = M[k][b][5][j][kk][bb] + M[k][b][6][j][kk][bb]
            m5_sub_m6 = M[k][b][5][j][kk][bb] - M[k][b][6][j][kk][bb]
            s0 = M[k][b][0][j][kk][bb] + m1_add_m2
            s5 = M[k][b][7][j][kk][bb] + m1_sub_m2
            s1 = m1_sub_m2 + m5_sub_m6 * 16
            s4 = m1_add_m2 + m3_add_m4 * 16
            s2 = m1_add_m2 + 8 * m5_add_m6
            s3 = m1_sub_m2 + 8 * m3_sub_m4
            s0 = s0 + m5_add_m6 * 32
            s5 = s5 + m3_sub_m4 * 32
            s1 = s1 + m3_sub_m4 * 2
            s4 = s4 + m5_add_m6 * 2
            s0 = s0 + m3_add_m4
            s5 = s5 + m5_sub_m6
            s2 = s2 + m3_add_m4 * 4
            s3 = s3 + m5_sub_m6 * 4
            temp_expr[(0, j)] = s0
            temp_expr[(1, j)] = s1
            temp_expr[(2, j)] = s2
            temp_expr[(3, j)] = s3
            temp_expr[(4, j)] = s4
            temp_expr[(5, j)] = s5
        now = tvm.const(0.0, "float32")
        for ii in range(m):
            for jj in range(alpha):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now

    N = get_const_int(X.shape[0])
    IH = get_const_int(X.shape[2])
    IW = get_const_int(X.shape[3])
    alpha = get_const_int(M.shape[0])

    K = get_const_int(M.shape[0]) * get_const_int(M.shape[4])
    P = get_const_int(M.shape[1]) * get_const_int(M.shape[5])

    A_T_dot_M = tvm.compute((K // VK, P // VP, m, alpha, VK, VP), compute_A_T_dot_M, name="A_T_dot_M")

    def compute_X_dot_A(k, b, eps, nu, kk, bb):
        temp_expr = {}

        for i in range(m):
            m1_add_m2 = A_T_dot_M[k][b][i][1][kk][bb] + A_T_dot_M[k][b][i][2][kk][bb]
            m1_sub_m2 = A_T_dot_M[k][b][i][1][kk][bb] - A_T_dot_M[k][b][i][2][kk][bb]
            m3_add_m4 = A_T_dot_M[k][b][i][3][kk][bb] + A_T_dot_M[k][b][i][4][kk][bb]
            m3_sub_m4 = A_T_dot_M[k][b][i][3][kk][bb] - A_T_dot_M[k][b][i][4][kk][bb]
            m5_add_m6 = A_T_dot_M[k][b][i][5][kk][bb] + A_T_dot_M[k][b][i][6][kk][bb]
            m5_sub_m6 = A_T_dot_M[k][b][i][5][kk][bb] - A_T_dot_M[k][b][i][6][kk][bb]
            s0 = A_T_dot_M[k][b][i][0][kk][bb] + m1_add_m2
            s5 = A_T_dot_M[k][b][i][7][kk][bb] + m1_sub_m2
            s1 = m1_sub_m2 + m5_sub_m6 * 16
            s4 = m1_add_m2 + m3_add_m4 * 16
            s2 = m1_add_m2 + 8 * m5_add_m6
            s3 = m1_sub_m2 + 8 * m3_sub_m4
            s0 = s0 + m5_add_m6 * 32
            s5 = s5 + m3_sub_m4 * 32
            s1 = s1 + m3_sub_m4 * 2
            s4 = s4 + m5_add_m6 * 2
            s0 = s0 + m3_add_m4
            s5 = s5 + m5_sub_m6
            s2 = s2 + m3_add_m4 * 4
            s3 = s3 + m5_sub_m6 * 4
            temp_expr[(i, 0)] = s0
            temp_expr[(i, 1)] = s1
            temp_expr[(i, 2)] = s2
            temp_expr[(i, 3)] = s3
            temp_expr[(i, 4)] = s4
            temp_expr[(i, 5)] = s5
        now = tvm.const(0.0, "float32")
        for ii in range(m):
            for jj in range(m):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now

    Y = tvm.compute((K // VK, P // VP, m, m, VK, VP), compute_X_dot_A, name="Y")
    OH = get_const_int((IH + 2 * HPAD - 3) // HSTR + 1)
    OW = get_const_int((IW + 2 * WPAD - 3) // WSTR + 1)
    nH, nW = get_const_int((OH + m-1) // m), get_const_int((OW + m-1) // m)

    # unpack output
    def _output(n, k, h, w):
        k_elem = k % VK
        k_tile = k // VK
        b = n * nH * nW + h // m * nW + w // m
        b_elem = b % VP
        b_tile = b // VP
        return Y[k_tile][b_tile][h % m][w % m][k_elem][b_elem]
    output = tvm.compute((N, K, OH, OW), _output,
                       name='output', tag='winograd_conv_output')
    return output


def decl_output_transform(cfg, X, M, VK, VP):
    N = get_const_int(X.shape[0])
    IH = get_const_int(X.shape[2])
    IW = get_const_int(X.shape[3])
    alpha = get_const_int(M.shape[0])

    K = get_const_int(M.shape[0]) * get_const_int(M.shape[4])
    P = get_const_int(M.shape[1]) * get_const_int(M.shape[5])

    # inverse transform
    A = const_matrix(A_data, 'A')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    Y = tvm.compute((K // VK, P // VP, m, m, VK, VP), lambda k, b, vh, vw, kk, bb:
                    tvm.sum(M[k][b][r_eps][r_nu][kk][bb] * A[r_eps][vh] * A[r_nu][vw],
                            axis=[r_eps, r_nu]), name='Y')
    OH = get_const_int((IH + 2 * HPAD - 3) // HSTR + 1)
    OW = get_const_int((IW + 2 * WPAD - 3) // WSTR + 1)
    nH, nW = get_const_int((OH + m-1) // m), get_const_int((OW + m-1) // m)

    # unpack output
    def _output(n, k, h, w):
        k_elem = k % VK
        k_tile = k // VK
        b = n * nH * nW + h // m * nW + w // m
        b_elem = b % VP
        b_tile = b // VP
        return Y[k_tile][b_tile][h % m][w % m][k_elem][b_elem]
    output = tvm.compute((N, K, OH, OW), _output,
                       name='output', tag='winograd_conv_output')

    return output

def schedule_output_transform(cfg, output):
    s = tvm.create_schedule(output.op)
    Y = output.op.input_tensors[0]

    cfg.define_knob('reorder_kk', [0, 1])
    cfg.define_knob('vectorize_bb', [1])
    cfg.define_knob('unroll_vh', [1])
    cfg.define_knob('unroll_vw', [1])
    cfg.define_knob('unroll_r_eps', [1])
    cfg.define_knob('unroll_r_nu', [1])
    cfg.define_knob('compute_temp2_at_temp1', [1])
    cfg.define_knob('M_read_cache', [0, 1])
    if cfg['use_minimal'].val:
        temp_1 = output.op.input_tensors[0]
        temp_2 = temp_1.op.input_tensors[0]
        M = temp_2.op.input_tensors[0]
        for temp in [temp_1, temp_2]:
            k, b, eps, nu, kk, bb = s[temp].op.axis
            if cfg['reorder_kk'].val:
                s[temp].reorder(k, b, kk, eps, nu, bb)
            if cfg['vectorize_bb'].val:
                s[temp].vectorize(bb)
            if cfg['unroll_r_eps']:
                s[temp].unroll(eps)
            if cfg['unroll_r_nu']:
                s[temp].unroll(nu)
        (k, b, eps, nu, kk, bb) = s[temp_1].op.axis
        if cfg['compute_temp2_at_temp1'].val:
            s[temp_2].compute_at(s[temp_1], b)
        if cfg['M_read_cache'].val:
            MM = s.cache_read(M, 'global', [temp_2])
            (k, b, eps, nu, kk, bb) = s[temp_2].op.axis
            s[MM].compute_at(s[temp_2], b)
        pass
    else:
        M, A = Y.op.input_tensors
        s[A].compute_inline()

        k, b, vh, vw, kk, bb = s[Y].op.axis

        if cfg['reorder_kk'].val:
            s[Y].reorder(k, b, kk, vh, vw, bb)

        if cfg['vectorize_bb'].val:
            s[Y].vectorize(bb)

        r_eps, r_nu = s[Y].op.reduce_axis
        n, co, h, w = s[output].op.axis


        if cfg['unroll_vh'].val:
            s[Y].unroll(vh)

        if cfg['unroll_vw'].val:
            s[Y].unroll(vw)

        if cfg['unroll_r_eps'].val:
            s[Y].unroll(r_eps)
        if cfg['unroll_r_nu'].val:
            s[Y].unroll(r_nu)
    return s


@autotvm.template
def output_transform_autotvm(dtype):
    cfg = autotvm.get_config()
    cfg.define_knob('VK', [2, 4, 8, 16])
    cfg.define_knob('VP', [4, 8, 16])
    VK = cfg['VK'].val
    VP = cfg['VP'].val
    X = tvm.placeholder(shape=(1, 64, 56, 56), dtype="float32", name="X")
    W = tvm.placeholder(shape=(64, 64, 56, 56), dtype="float32", name="W")
    N = get_const_int(X.shape[0])
    IH = get_const_int(X.shape[2])
    IW = get_const_int(X.shape[3])
    OH = get_const_int((IH + 2 * HPAD - 3) // HSTR + 1)
    OW = get_const_int((IW + 2 * WPAD - 3) // WSTR + 1)
    nH, nW = get_const_int((OH + m-1) // m), get_const_int((OW + m-1) // m)

    def round_up(a, b):
        return ((a + b - 1) // b) * b
    P = round_up(N * nH * nW, VP)
    K = get_const_int(W.shape[0])
    assert K % VK == 0
    assert P % VP == 0

    cfg.define_knob('use_minimal', [1])
    M = tvm.placeholder(shape=(K // VK, P // VP, alpha, alpha, VK, VP), name="M")
    if cfg['use_minimal'].val:
        output = decl_output_transform_minimal(cfg, X, M, VK, VP)
    else:
        output = decl_output_transform(cfg, X, M, VK, VP)
    s = schedule_output_transform(cfg, output)
    #print(tvm.lower(s, [X, M, output], simple_mode=True))
    return s, [X, M, output]


@autotvm.template
def conv2d_winograd_autotvm(s, ic, oc):
    cfg = autotvm.get_config()
    cfg.define_knob('unroll', [1])
    cfg.define_knob('compute_at', [0])
    cfg.define_knob('vectorize', [1])
    cfg.define_knob('tensorize', [1])
    cfg.define_knob('VK', [6])
    cfg.define_knob('VP', [8])
    for intermediate in ["M", "A_T_dot_M", "input_tile", "B_T_dot_X", "V"]:
        cfg.define_knob("{}_COMPUTE_AT".format(intermediate), [0, 1])
    for intermediate in ["input_tile", "V"]: # , "B_T_dot_X",
        cfg.define_knob("{}_REORDER_C".format(intermediate), [0, 1])

    cfg.define_knob('data_pad_inline', [0, 1])

    VK = cfg['VK'].val
    VP = cfg['VP'].val
    X = tvm.placeholder(shape=(1, ic, s, s), dtype="float32", name="X")
    W = tvm.placeholder(shape=(oc, ic, 3, 3), dtype="float32", name="W")

    Y, input_tile, U, output = decl_winograd(cfg, X, W, strides=1, padding=1, layout="NCHW", out_dtype="float32", VK=VK, VP=VP)
    s = schedule_winograd(cfg, Y, VK=VK, VP=VP)
    if cfg.flop == 0:
        cfg.add_flop(2 * ic * oc * s * s * 3 * 3)
    #print(tvm.lower(s, [X, W, output], simple_mode=True))
    return s, [input_tile, U, Y]
