'''
使用数值计算方法解决反应堆中子扩散问题

    ∂φ(x,t)/∂t = D∇²φ(x,t) - Σaφ(x,t) + S(x,t)

其中:
    φ(x,t) 为中子通量密度
    D 为中子扩散系数
    Σa 为吸收截面
    S(x,t) 为源项

边界条件:

    1) 真空边界条件:
        -D∇φ·n = 0, φ = 0
    2) 反射边界条件:
        -D∇φ·n = 0, ∂φ/∂n = 0

反应堆几何模型: 矩形堆芯
'''

import os
import argparse
import tomli
import numpy as np
import numba
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres
from scipy.sparse import csr_matrix

###### PART I 读取参数 ######

parser = argparse.ArgumentParser()
parser.add_argument("--refine", type=int, default=1)
parser.add_argument("--config", type=str, default="config.toml")
args = parser.parse_args()
refine = args.refine
config = args.config
if not os.path.exists(config):
    raise FileNotFoundError(f"config file {config} not found")
with open(config, "rb") as f:
    config = tomli.load(f)

predir = f"{config['title']}-{refine}"
os.makedirs(predir, exist_ok=True)

length_x = config["geometry"]["length_x"]
length_y = config["geometry"]["length_y"]
mesh_num_x = config["geometry"]["geo_num_x"]*refine
mesh_num_y = config["geometry"]["geo_num_y"]*refine
mesh_length_x = length_x/mesh_num_x
mesh_length_y = length_y/mesh_num_y
x = np.array([ (i+0.5)*mesh_length_x for i in range(mesh_num_x) ])
y = np.array([ (j+0.5)*mesh_length_y for j in range(mesh_num_y) ])

keff_ref = config["solution"]["k_eff"]

material = [{}]
for i_ in range(config["material"]["total"]):
    material.append(config["material"][f"mat_{i_+1}"])
D1, D2 = np.zeros(mesh_num_x*mesh_num_y), np.zeros(mesh_num_x*mesh_num_y)
a1, a2 = np.zeros(mesh_num_x*mesh_num_y), np.zeros(mesh_num_x*mesh_num_y)
nf1, nf2 = np.zeros(mesh_num_x*mesh_num_y), np.zeros(mesh_num_x*mesh_num_y)
s12 = np.zeros(mesh_num_x*mesh_num_y)
geo_length_x, geo_length_y = config["geometry"]["geo_length_x"], config["geometry"]["geo_length_y"]
geo_num_x, geo_num_y = config["geometry"]["geo_num_x"], config["geometry"]["geo_num_y"]
content = np.fromstring(
    config["geometry"]["content"], dtype=int, sep="\n"
    ).reshape(geo_num_x, geo_num_y)
ext_dist = config["boundary"]["ext_dist"]

###### PART II 数值计算 ######

#### PART II a 曲率修正 ####

@numba.jit(nopython=True)
def in_rect(x_, y_, x_start, y_start, x_length, y_length):
    '''判断是否位于反应堆内'''
    return x_start <= x_ <= x_start+x_length and y_start <= y_ <= y_start+y_length

@numba.jit(nopython=True)
def outside(x_, y_):
    '''未在矩形区域内或者在矩形区域内但材料为0'''
    return (not in_rect(x_, y_, 0, 0, length_x, length_y)) \
        or (content[int(x_//geo_length_x), int(y_//geo_length_y)] == 0)

Bz2 = config["material"]["Bz_sqr"]
for i_ in range(mesh_num_x):
    for j_ in range(mesh_num_y):
        xx, yy = x[i_], y[j_]
        if outside(xx, yy):
            continue
        mymat = material[content[int(xx//geo_length_x), int(yy//geo_length_y)]]
        idx = i_ + j_*mesh_num_x
        D1[idx], D2[idx] = mymat["d1"], mymat["d2"]
        a1[idx], a2[idx] = mymat["a1"], mymat["a2"]
        nf1[idx], nf2[idx] = mymat["nf1"], mymat["nf2"]
        s12[idx] = mymat["s12"]
        a1[idx] += D1[idx]*Bz2
        a2[idx] += D2[idx]*Bz2

#### PART II b 生成矩阵 ####

def gen_laplacian():
    '''生成laplacian矩阵'''
    @numba.jit(nopython=True)
    def idx_A(i, j):
        '''doc'''
        return i+j*mesh_num_x
    @numba.jit(nopython=True)
    def idx_B(i, j):
        '''doc'''
        return idx_A(i, j)+mesh_num_x*mesh_num_y
    @numba.jit(nopython=True)
    def gen_sub():
        '''doc'''
        # 生成一个满的拉普拉斯矩阵(-DΔ(u,v))
        triple_A = []
        for i in range(mesh_num_x):
            for j in range(mesh_num_y):
                if outside(x[i], y[j]):
                    # 设置 phi=0
                    triple_A.append((idx_A(i, j), idx_A(i, j), 1))
                    triple_A.append((idx_B(i, j), idx_B(i, j), 1))
                    continue

                # 1) 设置phi1
                temp = 0
                top, bot, lef, rig  = -1, -1, -1, -1
                val_up, val_rig = 1, 1
                d = D1[idx_A(i, j)] * ext_dist

                D1_top = 2/(1/D1[idx_A(i, j)] + 1/D1[idx_A(i, j+1)]
                            ) if not outside(x[i], y[j]+mesh_length_y) else D1[idx_A(i, j)]
                D1_bot = 2/(1/D1[idx_A(i, j)] + 1/D1[idx_A(i, j-1)]
                            ) if not outside(x[i], y[j]-mesh_length_y) else D1[idx_A(i, j)]
                D1_lef = 2/(1/D1[idx_A(i, j)] + 1/D1[idx_A(i-1, j)]
                            ) if not outside(x[i]-mesh_length_x, y[j]) else D1[idx_A(i, j)]
                D1_rig = 2/(1/D1[idx_A(i, j)] + 1/D1[idx_A(i+1, j)]
                            ) if not outside(x[i]+mesh_length_x, y[j]) else D1[idx_A(i, j)]

                # ∂n=0
                if i==0:
                    lef = 0
                if j==0:
                    bot = 0
                # phi=0
                if outside(x[i]+mesh_length_x, y[j]):
                    rig = 0
                    val_rig = mesh_length_x/(mesh_length_x/2+d)
                if outside(x[i], y[j]+mesh_length_y):
                    top = 0
                    val_up = mesh_length_y/(mesh_length_y/2+d)
                # update triple
                if top != 0:
                    triple_A.append((idx_A(i, j), idx_A(i, j+1), top*D1_top/(mesh_length_y**2)))
                if bot != 0:
                    triple_A.append((idx_A(i, j), idx_A(i, j-1), bot*D1_bot/(mesh_length_y**2)))
                if lef != 0:
                    triple_A.append((idx_A(i, j), idx_A(i-1, j), lef*D1_lef/(mesh_length_x**2)))
                if rig != 0:
                    triple_A.append((idx_A(i, j), idx_A(i+1, j), rig*D1_rig/(mesh_length_x**2)))

                temp += val_up*D1_top/(mesh_length_y**2)
                temp += D1_bot/(mesh_length_y**2) if bot != 0 else 0
                temp += D1_lef/(mesh_length_x**2) if lef != 0 else 0
                temp += val_rig*D1_rig/(mesh_length_x**2)
                triple_A.append((idx_A(i, j), idx_A(i, j), temp))

                # 2) 设置phi2
                temp = 0
                top, bot, lef, rig = -1, -1, -1, -1
                val_up, val_rig = 1, 1
                d = D2[idx_A(i, j)] * ext_dist

                D2_top = 2/(1/D2[idx_A(i, j)] + 1/D2[idx_A(i, j+1)]
                            ) if not outside(x[i], y[j]+mesh_length_y) else D2[idx_A(i, j)]
                D2_bot = 2/(1/D2[idx_A(i, j)] + 1/D2[idx_A(i, j-1)]
                            ) if not outside(x[i], y[j]-mesh_length_y) else D2[idx_A(i, j)]
                D2_lef = 2/(1/D2[idx_A(i, j)] + 1/D2[idx_A(i-1, j)]
                            ) if not outside(x[i]-mesh_length_x, y[j]) else D2[idx_A(i, j)]
                D2_rig = 2/(1/D2[idx_A(i, j)] + 1/D2[idx_A(i+1, j)]
                            ) if not outside(x[i]+mesh_length_x, y[j]) else D2[idx_A(i, j)]
                # ∂n=0
                if i==0:
                    lef = 0
                if j==0:
                    bot = 0
                # phi=0
                if outside(x[i]+mesh_length_x, y[j]):
                    rig = 0
                    val_rig = mesh_length_x/(mesh_length_x/2+d)
                if outside(x[i], y[j]+mesh_length_y):
                    top = 0
                    val_up = mesh_length_y/(mesh_length_y/2+d)
                # update triple
                if top != 0:
                    triple_A.append((idx_B(i, j), idx_B(i, j+1), top*D2_top/(mesh_length_y**2)))
                if bot != 0:
                    triple_A.append((idx_B(i, j), idx_B(i, j-1), bot*D2_bot/(mesh_length_y**2)))
                if lef != 0:
                    triple_A.append((idx_B(i, j), idx_B(i-1, j), lef*D2_lef/(mesh_length_x**2)))
                if rig != 0:
                    triple_A.append((idx_B(i, j), idx_B(i+1, j), rig*D2_rig/(mesh_length_x**2)))

                temp += val_up*D2_top/(mesh_length_y**2)
                temp += D2_bot/(mesh_length_y**2) if bot != 0 else 0
                temp += D2_lef/(mesh_length_x**2) if lef != 0 else 0
                temp += val_rig*D2_rig/(mesh_length_x**2)
                triple_A.append((idx_B(i, j), idx_B(i, j), temp))
        return triple_A

    triple_A = gen_sub()
    row, col, val = [], [], []
    for r, c, v in triple_A:
        row.append(r)
        col.append(c)
        val.append(v)
    A_ = csr_matrix((val, (row, col)), shape=(mesh_num_x*mesh_num_y*2, mesh_num_x*mesh_num_y*2))
    return A_

lap = gen_laplacian()

#### PART II c 线性算子 ####

# -D1Δu + (∑a1+∑1→2) u = λ (ν∑f1u + ν∑f2v)
# -D2Δv + (∑a2) v = ∑1→2 u

# linearoperator
def mat_A(phi_):
    '''doc'''
    res = lap*phi_
    res[:mesh_num_x*mesh_num_y] += (a1+s12)*phi_[:mesh_num_x*mesh_num_y]
    res[mesh_num_x*mesh_num_y:] += a2*phi_[mesh_num_x*mesh_num_y:]-s12*phi_[:mesh_num_x*mesh_num_y]
    return res
A = LinearOperator((mesh_num_x*mesh_num_y*2, mesh_num_x*mesh_num_y*2), mat_A)
def mat_B(phi_):
    '''doc'''
    res = np.zeros(mesh_num_x*mesh_num_y*2)
    res[:mesh_num_x*mesh_num_y] = nf1*phi_[:mesh_num_x*mesh_num_y]+nf2*phi_[mesh_num_x*mesh_num_y:]
    return res
B = LinearOperator((mesh_num_x*mesh_num_y*2, mesh_num_x*mesh_num_y*2), mat_B)


#### PART II d 源迭代 ####

phi = np.ones(mesh_num_x*mesh_num_y*2)
phi /= np.sqrt(np.sum(phi**2))
keff = 1
keff_his = []

def step(phi_, keff_):
    '''doc'''
    nxt_phi, _ = gmres(A, B*phi_/keff_)
    keff_ = keff_ * np.sum(B*nxt_phi) / (np.sum(B*phi_))
    return nxt_phi, keff_
for i_ in range(100):
    phi, keff = step(phi, keff)
    keff_his.append(keff)
    print(f"iter {i_}, {keff=:.5f}")
print(f"finish: {keff=:.5f}")

np.save(f'{predir}/phi.npy', phi)
keff_his = np.array(keff_his)
np.save(f'{predir}/keff.npy', keff_his)

###### PART III 做图 ######

xarr = [0, 10, 70, 90, 130, 150, 170]
yarr = xarr
def plot_phi(phi_):
    '''画出 phi 的图像'''
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    phi1 = phi_[:mesh_num_x*mesh_num_y]
    phi2 = phi_[mesh_num_x*mesh_num_y:]
    phi1 = phi1.reshape(mesh_num_y, mesh_num_x)
    phi2 = phi2.reshape(mesh_num_y, mesh_num_x)
    phi1max = np.max(np.abs(phi1))
    im = ax[0].imshow(phi1, cmap='jet', vmin=0, vmax=phi1max, origin='lower',
                      extent=(0, length_x, 0, length_y))
    fig.colorbar(im, ax=ax[0], orientation='vertical', fraction=0.046, pad=0.04)
    ax[0].set_title("$\\phi_1$")
    ax[0].grid(True, which='both', linestyle='--', linewidth=1, color='white')
    ax[0].set_xticks(xarr)
    ax[0].set_yticks(yarr)
    phi2max = np.max(np.abs(phi2))
    im = ax[1].imshow(phi2, cmap='jet', vmin=0, vmax=phi2max,
                      extent=(0, length_x, 0, length_y), origin='lower')
    fig.colorbar(im, ax=ax[1], orientation='vertical', fraction=0.046, pad=0.04)
    ax[1].set_title("$\\phi_2$")
    ax[1].grid(True, which='both', linestyle='--', linewidth=1, color='white')
    ax[1].set_xticks(xarr)
    ax[1].set_yticks(yarr)
    plt.tight_layout()
    plt.savefig(f'{predir}/phi.png')
    plt.close()
plot_phi(phi)

def plot_keff():
    '''画出 keff 的图像'''
    _fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(keff_his, label='$k_{eff}$='+f"{keff_his[-1]:.5f}")
    ax[0].plot([0, len(keff_his)], [keff_ref, keff_ref], 'r--', label="$k_{eff,ref}=$"f'{keff_ref}')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('keff')
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].legend()
    ax[1].plot(np.abs(keff_his-keff_ref)/keff_ref)
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Relative error of keff')
    ax[1].set_yscale('log')
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'{predir}/keff.png')
    plt.close()

plot_keff()
