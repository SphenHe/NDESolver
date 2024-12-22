"""
该程序用于求解中子扩散方程，计算反应堆中的中子通量和有效增殖因子。

主要步骤包括：
1. 读取参数：从配置文件中读取几何、材料和边界条件等参数。
2. 数值计算：
    a. 曲率修正：根据材料属性和几何信息，修正扩散系数和吸收系数。
    b. 生成矩阵：生成拉普拉斯矩阵的稀疏表示，处理边界条件。
    c. 线性算子：定义矩阵 A 和 B 的线性算子，用于求解线性方程组。
    d. 源迭代：通过源迭代方法，更新中子通量和有效增殖因子。
3. 做图：绘制中子通量和有效增殖因子的图像，并保存结果。

主要函数和类：
- in_rect(x_, y_, x_start, y_start, x_length, y_length): 判断是否位于反应堆内。
- outside(x_, y_): 判断是否在矩形区域外或者在矩形区域内但材料为0。
- gen_laplacian(): 生成拉普拉斯矩阵的稀疏表示。
- mat_A(phi_): 计算矩阵 A 乘以向量 phi_ 的结果。
- mat_B(phi_): 计算矩阵 B 乘以向量 phi_ 的结果。
- step(phi_, keff_): 执行一步源迭代，更新中子通量和有效增殖因子。
- plot_phi(phi_): 绘制中子通量的图像。
- plot_keff(): 绘制有效增殖因子的图像。

变量：
- refine: 网格细化倍数。
- config: 配置文件路径。
- predir: 结果保存目录。
- length_x, length_y: 几何尺寸。
- mesh_num_x, mesh_num_y: 网格数量。
- mesh_length_x, mesh_length_y: 网格尺寸。
- x, y: 网格中心坐标。
- keff_ref: 参考有效增殖因子。
- material: 材料属性列表。
- D1, D2: 扩散系数数组。
- a1, a2: 吸收系数数组。
- nf1, nf2: 中子产额数组。
- s12: 散射系数数组。
- geo_length_x, geo_length_y: 几何尺寸。
- geo_num_x, geo_num_y: 几何网格数量。
- content: 几何内容数组。
- ext_dist: 边界条件扩展距离。
- Bz2: 材料属性中的 Bz 平方。
- lap: 拉普拉斯矩阵。
- phi: 中子通量数组。
- keff: 有效增殖因子。
- keff_data: 有效增殖因子历史记录。
- xarr, yarr: 绘图坐标数组。

输出文件：
- phi.data: 中子通量数组。
- keff.data: 有效增殖因子历史记录数组。
- phi.png: 中子通量图像。
- keff.png: 有效增殖因子图像。
"""

import os
import argparse
import tomli
import numpy as np
import numba
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import tfqmr
from scipy.sparse import csr_matrix

def plot_phi(phi_):
    '''画出 phi 的图像'''
    # xarr = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]
    xarr = [i*10 for i in range(18)]
    yarr = xarr
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    phi1 = phi_[:mesh_num_x*mesh_num_y]
    phi2 = phi_[mesh_num_x*mesh_num_y:]
    phi1 = phi1.reshape(mesh_num_y, mesh_num_x)
    phi2 = phi2.reshape(mesh_num_y, mesh_num_x)
    phi1max = np.max(np.abs(phi1))
    im = ax[0].imshow(phi1, cmap='coolwarm', vmin=0, vmax=phi1max, origin='lower',
                      extent=(0, length_x, 0, length_y))
    fig.colorbar(im, ax=ax[0], orientation='vertical', fraction=0.046, pad=0.04)
    ax[0].set_title("$\\phi_1$ Neutron Flux")
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5, color='white')
    ax[0].set_xticks(xarr)
    ax[0].set_yticks(yarr)
    phi2max = np.max(np.abs(phi2))
    im = ax[1].imshow(phi2, cmap='coolwarm', vmin=0, vmax=phi2max,
                      extent=(0, length_x, 0, length_y), origin='lower')
    fig.colorbar(im, ax=ax[1], orientation='vertical', fraction=0.046, pad=0.04)
    ax[1].set_title("$\\phi_2$ Neutron Flux")
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5, color='white')
    ax[1].set_xticks(xarr)
    ax[1].set_yticks(yarr)
    plt.tight_layout()
    plt.savefig(f'{predir}/phi.pdf')
    plt.close()

def plot_keff():
    '''画出 keff 的图像'''
    _fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(keff_data, label='$k_{eff}$='+f"{keff_data[-1]:.6f}")
    ax[0].plot([0, len(keff_data)], [keff_ref, keff_ref], 'r--', label="$k_{eff,ref}=$"f'{keff_ref}')
    ax[0].plot([0, len(keff_data)], [keff, keff], 'g--', label="$k_{eff,final}$="f'{keff:.6f}')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('keff')
    ax[0].set_title('Effective Multiplication Factor')
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].legend()
    ax[1].plot(np.abs(keff_data-keff_ref)/keff_ref, label='Relative error')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Relative error of keff')
    ax[1].set_yscale('log')
    ax[1].set_title('Relative Error of Effective Multiplication Factor')
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'{predir}/keff.pdf')
    plt.close()

@numba.jit(nopython=True)
def in_rect(x_, y_, x_start, y_start, x_length, y_length):
    '''判断是否位于反应堆内'''
    return x_start <= x_ <= x_start+x_length and y_start <= y_ <= y_start+y_length

@numba.jit(nopython=True)
def outside(x_, y_):
    '''未在矩形区域内或者在矩形区域内但材料为0'''
    return (not in_rect(x_, y_, 0, 0, length_x, length_y)) \
        or (content[int(x_//geo_length_x), int(y_//geo_length_y)] == 0)

def mat_A(phi_):
    '''计算矩阵 A 乘以向量 phi_ 的结果'''
    res = lap*phi_
    res[:mesh_num_x*mesh_num_y] += (a1+s12)*phi_[:mesh_num_x*mesh_num_y]
    res[mesh_num_x*mesh_num_y:] += a2*phi_[mesh_num_x*mesh_num_y:]-s12*phi_[:mesh_num_x*mesh_num_y]
    return res
def mat_B(phi_):
    '''计算矩阵 B 乘以向量 phi_ 的结果'''
    res = np.zeros(mesh_num_x*mesh_num_y*2)
    res[:mesh_num_x*mesh_num_y] = nf1*phi_[:mesh_num_x*mesh_num_y]+nf2*phi_[mesh_num_x*mesh_num_y:]
    return res

def step(phi_, keff_):
    '''执行一步源迭代，更新中子通量和有效增殖因子'''
    nxt_phi, info = tfqmr(A, B*phi_/keff_)
    if info != 0:
        raise RuntimeError(f"TFQMR did not converge, info={info}")
    keff_ = keff_ * np.sum(B*nxt_phi) / (np.sum(B*phi_))
    return nxt_phi, keff_

def gen_laplacian():
    '''
    生成拉普拉斯矩阵。

    该函数为给定的网格生成拉普拉斯矩阵的稀疏表示。
    它处理边界条件并相应地更新矩阵。

    返回:
        scipy.sparse.csr_matrix: 生成的拉普拉斯矩阵，采用压缩稀疏行(CSR)格式。
    '''
    @numba.jit(nopython=True)
    def idx_A(i, j):
        '''
        计算二维网格在一维数组表示中的索引。

        parameters:
            i (int): 行索引。
            j (int): 列索引。

        returns:
            int: 对应于二维网格中 (i, j) 位置的一维数组中的索引。
        '''
        return i+j*mesh_num_x
    @numba.jit(nopython=True)
    def idx_B(i, j):
        '''
        计算给定索引 i 和 j 的索引 B。

        parameters:
            i (int): 行索引。
            j (int): 列索引。

        returns:
            int: 计算得到的索引 B。
        '''
        return idx_A(i, j)+mesh_num_x*mesh_num_y
    @numba.jit(nopython=True)
    def gen_sub():
        '''
        生成一个满的拉普拉斯矩阵(-DΔ(u,v))，以三元组列表的形式返回。
        该函数为给定的网格生成拉普拉斯矩阵的稀疏表示。
        它处理边界条件并相应地更新矩阵。

        returns:
            list of tuple: 表示拉普拉斯矩阵非零条目的三元组列表。
                   每个三元组的形式为 (行索引, 列索引, 值)。
        '''
        triple_A = []
        for i in range(mesh_num_x):
            for j in range(mesh_num_y):
                if outside(x[i], y[j]):
                    # 设置 phi=0
                    triple_A.append((idx_A(i, j), idx_A(i, j), 1))
                    triple_A.append((idx_B(i, j), idx_B(i, j), 1))
                    continue

                # 设置phi1
                tmp = 0
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
                if i==0: lef = 0
                if j==0: bot = 0
                # phi=0
                if outside(x[i]+mesh_length_x, y[j]): rig = 0; val_rig = mesh_length_x/(mesh_length_x/2+d)
                if outside(x[i], y[j]+mesh_length_y): top = 0; val_up = mesh_length_y/(mesh_length_y/2+d)
                # update triple
                if top != 0: triple_A.append((idx_A(i, j), idx_A(i, j+1), top*D1_top/(mesh_length_y**2)))
                if bot != 0: triple_A.append((idx_A(i, j), idx_A(i, j-1), bot*D1_bot/(mesh_length_y**2)))
                if lef != 0: triple_A.append((idx_A(i, j), idx_A(i-1, j), lef*D1_lef/(mesh_length_x**2)))
                if rig != 0: triple_A.append((idx_A(i, j), idx_A(i+1, j), rig*D1_rig/(mesh_length_x**2)))

                tmp += val_up*D1_top/(mesh_length_y**2)
                tmp += D1_bot/(mesh_length_y**2) if bot != 0 else 0
                tmp += D1_lef/(mesh_length_x**2) if lef != 0 else 0
                tmp += val_rig*D1_rig/(mesh_length_x**2)
                triple_A.append((idx_A(i, j), idx_A(i, j), tmp))

                # 设置phi2
                tmp = 0
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
                if i==0: lef = 0
                if j==0: bot = 0
                # phi=0
                if outside(x[i]+mesh_length_x, y[j]): rig = 0; val_rig = mesh_length_x/(mesh_length_x/2+d)
                if outside(x[i], y[j]+mesh_length_y): top = 0; val_up = mesh_length_y/(mesh_length_y/2+d)
                # update triple
                if top != 0: triple_A.append((idx_B(i, j), idx_B(i, j+1), top*D2_top/(mesh_length_y**2)))
                if bot != 0: triple_A.append((idx_B(i, j), idx_B(i, j-1), bot*D2_bot/(mesh_length_y**2)))
                if lef != 0: triple_A.append((idx_B(i, j), idx_B(i-1, j), lef*D2_lef/(mesh_length_x**2)))
                if rig != 0: triple_A.append((idx_B(i, j), idx_B(i+1, j), rig*D2_rig/(mesh_length_x**2)))

                tmp += val_up*D2_top/(mesh_length_y**2)
                tmp += D2_bot/(mesh_length_y**2) if bot != 0 else 0
                tmp += D2_lef/(mesh_length_x**2) if lef != 0 else 0
                tmp += val_rig*D2_rig/(mesh_length_x**2)
                triple_A.append((idx_B(i, j), idx_B(i, j), tmp))
        return triple_A

    triple_A = gen_sub()
    row, col, val = [], [], []
    for r, c, v in triple_A:
        row.append(r)
        col.append(c)
        val.append(v)
    A_ = csr_matrix((val, (row, col)), shape=(mesh_num_x*mesh_num_y*2, mesh_num_x*mesh_num_y*2))
    return A_

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
x = np.array(object=[ (i+0.5)*mesh_length_x for i in range(mesh_num_x) ])
y = np.array([ (j+0.5)*mesh_length_y for j in range(mesh_num_y) ])

keff_ref = config["solution"]["k_eff"]

material = [{}]
for i_ in range(config["material"]["total"]):
    material.append(config["material"][f"mat_{i_+1}"])
geo_length_x, geo_length_y = config["geometry"]["geo_length_x"], config["geometry"]["geo_length_y"]
geo_num_x, geo_num_y = config["geometry"]["geo_num_x"], config["geometry"]["geo_num_y"]
content = np.fromstring(
    config["geometry"]["content"], dtype=int, sep="\n"
    ).reshape(geo_num_x, geo_num_y)
ext_dist = config["boundary"]["ext_dist"]

###### PART II 数值计算 ######

#### PART II a 曲率修正 ####

D1, D2 = np.zeros(mesh_num_x*mesh_num_y), np.zeros(mesh_num_x*mesh_num_y)
a1, a2 = np.zeros(mesh_num_x*mesh_num_y), np.zeros(mesh_num_x*mesh_num_y)
nf1, nf2 = np.zeros(mesh_num_x*mesh_num_y), np.zeros(mesh_num_x*mesh_num_y)
s12 = np.zeros(mesh_num_x*mesh_num_y)

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

lap = gen_laplacian()

#### PART II c 线性算子 ####

# -D1Δu + (∑a1+∑1→2) u = λ (ν∑f1u + ν∑f2v)
# -D2Δv + (∑a2) v = ∑1→2 u

# linearoperator

A = LinearOperator((mesh_num_x*mesh_num_y*2, mesh_num_x*mesh_num_y*2), mat_A)
B = LinearOperator((mesh_num_x*mesh_num_y*2, mesh_num_x*mesh_num_y*2), mat_B)

#### PART II d 源迭代 ####

phi = np.ones(mesh_num_x*mesh_num_y*2)
phi /= np.sqrt(np.sum(phi**2))
keff = 1
keff_data = []

for i_ in range(50):
    phi, keff = step(phi, keff)
    keff_data.append(keff)
    print(f"iter {i_}, {keff=:.5f}")
print(f"finish: {keff=:.5f}")

np.save(f'{predir}/phi.data', phi)
keff_data = np.array(keff_data)
np.save(f'{predir}/keff.data', keff_data)

###### PART III 做图 ######
plot_phi(phi)
plot_keff()
