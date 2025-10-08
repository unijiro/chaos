import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from typing import Dict, Tuple

# ========================================
# パラメータ設定（論文に基づく）
# ========================================

class TSNParameters:
    """TSN Model Parameters based on Orima et al., ISCAS 2024"""
    def __init__(self):
        # 容量
        self.Cv = 50e-15    # 膜電位容量 [F]
        self.Cu = 200e-15   # 回復変数容量 [F]
        
        # 電圧
        self.VDD = 0.6      # 電源電圧 [V]
        self.VTH = 0.35     # スパイク閾値 [V]
        self.VL = 0.1       # リーク電圧 [V]
        
        # 制御パラメータ
        self.VA = 0.0       # パラメータa [V]
        self.VD = 0.4       # パラメータb [V]
        self.VC = 0.14      # リセット電圧c [V]
        
        # 電流
        self.iIN = 0.1 * 3.2e-9   # 入力電流 [A]
        
        # シミュレーション設定
        self.dt = 0.05e-6         # サンプリング周期 [s]
        self.time_length = 100e-3  # シミュレーション時間 [s] (25ms→100msに延長)
        self.spike_wd = 7.3e-6    # スパイク幅（不応期）[s]
        self.Tref = 7.3e-6        # 不応期 [s]

class DeviceParameters:
    """デバイスパラメータ（MATLABコードから）"""
    
    # 式(3): g1(vV) = K1 * exp(α1 * vV)
    K1 = 3e-12
    alpha1 = 24.712
    
    # 式(4): g2(VL) = K2 * exp(α2 * VL)
    K2 = 2e-12
    alpha2 = 28.344
    
    # 式(5): g3(vU) = K3 * exp(α3 * vU)
    K3 = 2e-12
    alpha3 = 26.05
    
    # 式(6): g4(vU) - 区分的指数関数
    K4a = 1.15e-12
    alpha4a = 27.0
    K4b = 1.27e-12
    alpha4b = 28.5
    threshold_coef1 = 0.0816
    threshold_coef2 = 5.5687
    K4c = 1e-12
    alpha4c = 73.857
    beta4c = 15.505
    
    # 式(7): g5(VD) = K5 * exp(α5 * VD) when vY(t) = VDD
    K5a = 1.77e-6
    alpha5a = -11.0
    K5b = 1.57e-5
    alpha5b = -28.0

dev_params = DeviceParameters()

# ========================================
# デバイス関数（論文の式(3)-(7)）
# ========================================

def safe_exp(x):
    """安全な指数関数（オーバーフロー防止）"""
    return np.exp(np.clip(x, -700, 700))

def g1(vV):
    """式(3): g1(vV) = K1 * exp(α1 * vV)"""
    return dev_params.K1 * safe_exp(dev_params.alpha1 * vV)

def g2(VL):
    """式(4): g2(VL) = K2 * exp(α2 * VL)"""
    return dev_params.K2 * safe_exp(dev_params.alpha2 * VL)

def g3(vU):
    """式(5): g3(vU) = K3 * exp(α3 * vU)"""
    return dev_params.K3 * safe_exp(dev_params.alpha3 * vU)

def g4(vU, VA):
    """式(6): g4(vU, VA) - 区分的指数関数"""
    if VA < 0.05:
        return dev_params.K4a * safe_exp(dev_params.alpha4a * VA)
    else:
        threshold = dev_params.threshold_coef1 * np.exp(dev_params.threshold_coef2 * VA)
        if vU > threshold:
            return dev_params.K4b * safe_exp(dev_params.alpha4b * VA)
        else:
            exponent = (dev_params.alpha4c * VA + dev_params.beta4c) * vU
            return dev_params.K4c * safe_exp(exponent)

def g5(VD):
    """式(7): g5(VD) = K5 * exp(α5 * VD) when vY(t) = VDD"""
    if VD < 0.15:
        return dev_params.K5a * safe_exp(dev_params.alpha5a * VD)
    else:
        return dev_params.K5b * safe_exp(dev_params.alpha5b * VD)

# ========================================
# 微分方程式（論文の式(1)(2)）
# ========================================

def fp(vV, VL, vU, u, Cv):
    """論文の式(1): Cv * dvV(t)/dt = g1(vV(t)) - g2(VL) - g3(vU(t)) + iIN(t)"""
    return (g1(vV) - g2(VL) - g3(vU) + u) / Cv

def fv(y, vU, VA, VD, Cu):
    """論文の式(2): Cu * dvU(t)/dt = -g4(vU(t)) + g5(VD)"""
    return (-g4(vU, VA) + y * g5(VD)) / Cu

# ========================================
# シミュレーション（MATLABのHeun法）
# ========================================

def simulate_tsn_matlab_style(params: TSNParameters):
    """MATLABのHeun法を再現したシミュレーション"""
    
    T = int(params.time_length / params.dt)
    
    vV = np.zeros(T) + 0.1
    vU = np.zeros(T)
    y = np.zeros(T)
    skip = -1
    
    u = params.iIN
    
    for t in range(2, T):
        # Heun法のステップ1（予測）
        dvV1 = fp(vV[t-1], params.VL, vU[t-1], u, params.Cv)
        dvU1 = fv(y[t-1], vU[t-1], params.VA, params.VD, params.Cu)
        
        vV_temp = vV[t-1] + params.dt * dvV1
        vU_temp = vU[t-1] + params.dt * dvU1
        
        # Heun法のステップ2（修正）
        dvV2 = fp(vV_temp, params.VL, vU_temp, u, params.Cv)
        dvU2 = fv(y[t-1], vU_temp, params.VA, params.VD, params.Cu)
        
        vV[t] = vV[t-1] + params.dt * (dvV1 + dvV2) / 2
        vU[t] = vU[t-1] + params.dt * (dvU1 + dvU2) / 2
        
        # スパイク処理（論文の式(8)(9)）
        if vV[t] > params.VTH:
            vV[t] = params.VC
            skip = int(round(2 * params.spike_wd / params.dt))
        
        if skip > -1:
            y[t] = 1
            skip -= 1
        
        vV[t] = np.clip(vV[t], 0, params.VDD)
        vU[t] = np.clip(vU[t], 0, params.VDD)
    
    t_array = np.arange(T) * params.dt
    
    return t_array, vV, vU, y

# ========================================
# データ保存・読み込み
# ========================================

def save_simulation_data(data_dict, filename='tsn_simulation_data.pkl'):
    """シミュレーションデータを保存"""
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Data saved to {filename}")

def load_simulation_data(filename='tsn_simulation_data.pkl'):
    """シミュレーションデータを読み込み"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found")
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f)
    print(f"Data loaded from {filename}")
    return data_dict

def run_and_save_simulations(vc_list, filename='tsn_simulation_data.pkl'):
    """複数のVC値でシミュレーションを実行してデータを保存"""
    
    print("\n" + "=" * 80)
    print("Running TSN Simulations")
    print("=" * 80)
    
    data_dict = {}
    
    for vc in vc_list:
        params = TSNParameters()
        params.VC = vc
        
        print(f"Simulating VC={vc:.3f}V...")
        t, vV, vU, y = simulate_tsn_matlab_style(params)
        
        data_dict[vc] = {
            't': t,
            'vV': vV,
            'vU': vU,
            'y': y,
            'params': {
                'Cv': params.Cv,
                'Cu': params.Cu,
                'VDD': params.VDD,
                'VTH': params.VTH,
                'VL': params.VL,
                'VA': params.VA,
                'VD': params.VD,
                'VC': params.VC,
                'iIN': params.iIN
            }
        }
        
        n_spikes = np.sum(np.diff(y) > 0.5)
        print(f"  → {n_spikes} spikes detected")
    
    save_simulation_data(data_dict, filename)
    print("\n" + "=" * 80)
    print(f"All simulations completed and saved to {filename}")
    print("=" * 80)
    
    return data_dict

# ========================================
# v-nullcline計算
# ========================================

def compute_v_nullcline(params, n_points=500):
    """v-nullcline: dvV/dt = 0"""
    vV_range = np.linspace(0, params.VDD, n_points)
    vU_null = []
    vV_null = []
    
    for vv in vV_range:
        target = g1(vv) - g2(params.VL) + params.iIN
        if target > 0:
            try:
                vu = np.log(target / dev_params.K3) / dev_params.alpha3
                if 0 <= vu <= params.VDD:
                    vV_null.append(vv)
                    vU_null.append(vu)
            except:
                pass
    
    return np.array(vV_null), np.array(vU_null)

# ========================================
# 位相平面プロット（正規化なしベクトル場）
# ========================================

def plot_phase_plane_no_normalization(data_dict, vc_list, 
                                     save_path_vy0='phase_plane_vy0.png',
                                     save_path_vyvdd='phase_plane_vyvdd.png'):
    """
    V_Y=0とV_Y=VDDの2パターンで位相平面プロットを作成
    ベクトル場は正規化なし（実際の大きさを反映）
    """
    
    n_plots = len(vc_list)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # ========================================
    # 図1: V_Y = 0 (通常状態)
    # ========================================
    
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes1 = axes1.flatten() if n_plots > 1 else [axes1]
    
    print("\n=== Creating Phase Plane Plots for V_Y = 0 (No Normalization) ===")
    
    for idx, vc in enumerate(vc_list):
        ax = axes1[idx]
        
        if vc not in data_dict:
            print(f"Warning: VC={vc} not found")
            continue
        
        data = data_dict[vc]
        vV = data['vV']
        vU = data['vU']
        params_dict = data['params']
        
        params = TSNParameters()
        for key, value in params_dict.items():
            setattr(params, key, value)
        
        # ========== ベクトル場（正規化なし）V_Y = 0 ==========
        vV_grid = np.linspace(0, params.VDD, 25)
        vU_grid = np.linspace(0, params.VDD, 25)
        VV_grid, VU_grid = np.meshgrid(vV_grid, vU_grid)
        
        dVV = np.zeros_like(VV_grid)
        dVU = np.zeros_like(VU_grid)
        
        for i in range(VV_grid.shape[0]):
            for j in range(VV_grid.shape[1]):
                dVV[i, j] = fp(VV_grid[i, j], params.VL, VU_grid[i, j], 
                              params.iIN, params.Cv)
                dVU[i, j] = fv(0, VU_grid[i, j], params.VA, params.VD, params.Cu)
        
        # スケーリング（見やすさのため、正規化ではない）
        # 最大値でスケールして矢印の長さを調整
        max_dVV = np.max(np.abs(dVV))
        max_dVU = np.max(np.abs(dVU))
        scale_factor = max(max_dVV, max_dVU)
        
        if scale_factor > 0:
            scale_value = 50 * scale_factor  # 矢印の長さ調整
        else:
            scale_value = 1
        
        ax.quiver(VV_grid, VU_grid, dVV, dVU,
                 color='gray', alpha=0.5, scale=scale_value, width=0.004,
                 headwidth=3, headlength=4, zorder=1)
        
        # ========== v-nullcline ==========
        vV_null, vU_null = compute_v_nullcline(params)
        ax.plot(vV_null, vU_null, 'r--', linewidth=2.5, 
                label='v-nullcline', zorder=3)
        
        # ========== 軌道 ==========
        ax.plot(vV, vU, 'k-', linewidth=2, label='Trajectory', zorder=4, alpha=0.8)
        
        # ========== リセット線 ==========
        ax.axvline(params.VTH, color='green', linestyle='--', 
                   linewidth=2, alpha=0.7, label='VTH (reset)', zorder=2)
        
        # ========== 注釈 ==========
        ax.text(0.05, 0.95, 
               'No u-nullcline\n(b=0 in Izhikevich)\nvU always decreases',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # ========== 設定 ==========
        ax.set_xlabel('V_V [V]', fontsize=12)
        ax.set_ylabel('V_U [V]', fontsize=12)
        ax.set_title(f'VC = {vc:.3f}V (V_Y = 0)', fontsize=13, fontweight='bold')
        ax.set_xlim(0, params.VDD)
        ax.set_ylim(0, params.VDD)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_aspect('equal')
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    for idx in range(n_plots, len(axes1)):
        axes1[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path_vy0, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path_vy0}")
    plt.show()
    
    # ========================================
    # 図2: V_Y = VDD (スパイク時)
    # ========================================
    
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes2 = axes2.flatten() if n_plots > 1 else [axes2]
    
    print("\n=== Creating Phase Plane Plots for V_Y = VDD (No Normalization) ===")
    
    for idx, vc in enumerate(vc_list):
        ax = axes2[idx]
        
        if vc not in data_dict:
            continue
        
        data = data_dict[vc]
        vV = data['vV']
        vU = data['vU']
        params_dict = data['params']
        
        params = TSNParameters()
        for key, value in params_dict.items():
            setattr(params, key, value)
        
        # ========== ベクトル場（正規化なし）V_Y = VDD ==========
        vV_grid = np.linspace(0, params.VDD, 25)
        vU_grid = np.linspace(0, params.VDD, 25)
        VV_grid, VU_grid = np.meshgrid(vV_grid, vU_grid)
        
        dVV = np.zeros_like(VV_grid)
        dVU = np.zeros_like(VU_grid)
        
        for i in range(VV_grid.shape[0]):
            for j in range(VV_grid.shape[1]):
                dVV[i, j] = fp(VV_grid[i, j], params.VL, VU_grid[i, j], 
                              params.iIN, params.Cv)
                dVU[i, j] = fv(1, VU_grid[i, j], params.VA, params.VD, params.Cu)
        
        # スケーリング
        max_dVV = np.max(np.abs(dVV))
        max_dVU = np.max(np.abs(dVU))
        scale_factor = max(max_dVV, max_dVU)
        
        if scale_factor > 0:
            scale_value = 50 * scale_factor
        else:
            scale_value = 1
        
        ax.quiver(VV_grid, VU_grid, dVV, dVU,
                 color='orange', alpha=0.5, scale=scale_value, width=0.004,
                 headwidth=3, headlength=4, zorder=1)
        
        # ========== v-nullcline ==========
        vV_null, vU_null = compute_v_nullcline(params)
        ax.plot(vV_null, vU_null, 'r--', linewidth=2.5, 
                label='v-nullcline', zorder=3)
        
        # ========== 軌道 ==========
        ax.plot(vV, vU, 'k-', linewidth=2, label='Trajectory', zorder=4, alpha=0.8)
        
        # ========== リセット線 ==========
        ax.axvline(params.VTH, color='green', linestyle='--', 
                   linewidth=2, alpha=0.7, label='VTH (reset)', zorder=2)
        
        # ========== 注釈 ==========
        g4_val = g4(0.1, params.VA)
        g5_val = g5(params.VD)
        
        ax.text(0.05, 0.95, 
               f'Spike state\ng5 active\nvU increases\n(dv_U/dt ≈ {(-g4_val + g5_val)/params.Cu:.1e} V/s)',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # ========== 設定 ==========
        ax.set_xlabel('V_V [V]', fontsize=12)
        ax.set_ylabel('V_U [V]', fontsize=12)
        ax.set_title(f'VC = {vc:.3f}V (V_Y = VDD)', fontsize=13, fontweight='bold')
        ax.set_xlim(0, params.VDD)
        ax.set_ylim(0, params.VDD)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_aspect('equal')
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    for idx in range(n_plots, len(axes2)):
        axes2[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path_vyvdd, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path_vyvdd}")
    plt.show()

# ========================================
# 時系列プロット関数
# ========================================

def plot_time_series_all(data_dict, vc_list, save_path='time_series_all.png'):
    """全てのVCについて、V_VとV_Uの時系列を表示"""
    
    n_plots = len(vc_list)
    n_cols = 2
    n_rows = n_plots
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 2.5*n_rows))
    
    print("\n=== Creating Time Series Plots ===")
    
    for idx, vc in enumerate(vc_list):
        if vc not in data_dict:
            print(f"Warning: VC={vc} not found")
            continue
        
        data = data_dict[vc]
        t = data['t']
        vV = data['vV']
        vU = data['vU']
        y = data['y']
        params_dict = data['params']
        
        t_ms = t * 1000
        
        # ========== 左列: V_V(t) ==========
        ax_left = axes[idx, 0]
        
        ax_left.plot(t_ms, vV, 'r-', linewidth=1.5, label='V_V')
        ax_left.axhline(params_dict['VTH'], color='green', linestyle='--', 
                       linewidth=1.5, alpha=0.7, label='VTH')
        ax_left.axhline(params_dict['VC'], color='orange', linestyle='--', 
                       linewidth=1.5, alpha=0.7, label='VC')
        
        ax_left.set_ylabel('V_V [V]', fontsize=11)
        ax_left.set_title(f'VC = {vc:.3f}V: Membrane Potential', 
                         fontsize=12, fontweight='bold')
        ax_left.grid(True, alpha=0.3, linestyle=':')
        ax_left.set_xlim(0, t_ms[-1])
        ax_left.set_ylim(0, params_dict['VDD'])
        ax_left.legend(loc='upper right', fontsize=9)
        
        if idx == n_plots - 1:
            ax_left.set_xlabel('Time [ms]', fontsize=11)
        else:
            ax_left.set_xticklabels([])
        
        # ========== 右列: V_U(t) ==========
        ax_right = axes[idx, 1]
        
        ax_right.plot(t_ms, vU, 'b-', linewidth=1.5, label='V_U')
        
        # スパイク時を背景色で示す
        spike_indices = np.where(y > 0.5)[0]
        if len(spike_indices) > 0:
            spike_starts = spike_indices[np.insert(np.diff(spike_indices) > 1, 0, True)]
            spike_ends = spike_indices[np.append(np.diff(spike_indices) > 1, True)]
            
            for start, end in zip(spike_starts[:20], spike_ends[:20]):
                ax_right.axvspan(t_ms[start], t_ms[end], 
                               alpha=0.2, color='orange', zorder=0)
        
        ax_right.set_ylabel('V_U [V]', fontsize=11)
        ax_right.set_title(f'VC = {vc:.3f}V: Recovery Variable', 
                          fontsize=12, fontweight='bold')
        ax_right.grid(True, alpha=0.3, linestyle=':')
        ax_right.set_xlim(0, t_ms[-1])
        ax_right.set_ylim(0, params_dict['VDD'])
        ax_right.legend(loc='upper right', fontsize=9)
        
        if idx == n_plots - 1:
            ax_right.set_xlabel('Time [ms]', fontsize=11)
        else:
            ax_right.set_xticklabels([])
        
        if len(spike_indices) > 0:
            ax_right.text(0.98, 0.05, 'Orange = Spike (V_Y=VDD)', 
                         transform=ax_right.transAxes, fontsize=8,
                         ha='right', va='bottom',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()

def plot_time_series_detailed(data_dict, vc, t_start=0, t_end=None, 
                              save_path=None):
    """単一VCについて、V_V、V_U、V_Yの詳細な時系列を表示（3段プロット）"""
    
    if vc not in data_dict:
        print(f"Error: VC={vc} not found")
        return
    
    data = data_dict[vc]
    t = data['t']
    vV = data['vV']
    vU = data['vU']
    y = data['y']
    params_dict = data['params']
    
    t_ms = t * 1000
    if t_end is None:
        t_end = t_ms[-1]
    
    mask = (t_ms >= t_start) & (t_ms <= t_end)
    t_plot = t_ms[mask]
    vV_plot = vV[mask]
    vU_plot = vU[mask]
    y_plot = y[mask]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # ========== 上段: V_V(t) ==========
    ax1.plot(t_plot, vV_plot, 'r-', linewidth=1.5, label='V_V (membrane potential)')
    ax1.axhline(params_dict['VTH'], color='green', linestyle='--', 
               linewidth=1.5, alpha=0.7, label='VTH (threshold)')
    ax1.axhline(params_dict['VC'], color='orange', linestyle='--', 
               linewidth=1.5, alpha=0.7, label='VC (reset)')
    
    ax1.set_ylabel('V_V [V]', fontsize=12, fontweight='bold')
    ax1.set_title(f'TSN Circuit Time Series: VC = {vc:.3f}V', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_ylim(0, params_dict['VDD'] * 1.05)
    ax1.legend(loc='upper right', fontsize=10)
    
    # ========== 中段: V_U(t) ==========
    ax2.plot(t_plot, vU_plot, 'b-', linewidth=1.5, label='V_U (recovery variable)')
    
    spike_indices_plot = np.where(y_plot > 0.5)[0]
    
    if len(spike_indices_plot) > 0:
        spike_starts = spike_indices_plot[np.insert(np.diff(spike_indices_plot) > 1, 0, True)]
        spike_ends = spike_indices_plot[np.append(np.diff(spike_indices_plot) > 1, True)]
        
        for start, end in zip(spike_starts, spike_ends):
            ax2.axvspan(t_plot[start], t_plot[end], 
                       alpha=0.2, color='orange', zorder=0)
    
    ax2.set_ylabel('V_U [V]', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.set_ylim(0, params_dict['VDD'] * 1.05)
    ax2.legend(loc='upper right', fontsize=10)
    
    # ========== 下段: V_Y(t) ==========
    y_voltage = y_plot * params_dict['VDD']
    
    ax3.plot(t_plot, y_voltage, 'k-', linewidth=1.5, label='V_Y (spike output)')
    ax3.fill_between(t_plot, 0, y_voltage, alpha=0.3, color='gray')
    
    ax3.set_xlabel('Time [ms]', fontsize=12, fontweight='bold')
    ax3.set_ylabel('V_Y [V]', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle=':')
    ax3.set_ylim(-0.05, params_dict['VDD'] * 1.1)
    ax3.legend(loc='upper right', fontsize=10)
    
    # スパイク数の表示
    n_spikes = len(spike_starts) if len(spike_indices_plot) > 0 else 0
    fig.text(0.99, 0.01, f'Total spikes in view: {n_spikes}', 
            ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def plot_time_series_zoom(data_dict, vc, save_path=None):
    """複数の時間スケールで表示（全体 + ズーム）"""
    
    if vc not in data_dict:
        print(f"Error: VC={vc} not found")
        return
    
    data = data_dict[vc]
    t = data['t']
    vV = data['vV']
    vU = data['vU']
    y = data['y']
    params_dict = data['params']
    
    t_ms = t * 1000
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # ========== 左列: 全体像 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    
    # V_V全体
    ax1.plot(t_ms, vV, 'r-', linewidth=1)
    ax1.axhline(params_dict['VTH'], color='g', linestyle='--', alpha=0.7)
    ax1.set_ylabel('V_V [V]', fontsize=11)
    ax1.set_title(f'Full Time Series (VC={vc:.3f}V)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, params_dict['VDD'])
    
    # V_U全体
    ax2.plot(t_ms, vU, 'b-', linewidth=1)
    ax2.set_ylabel('V_U [V]', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, params_dict['VDD'])
    
    # V_Y全体
    ax3.plot(t_ms, y * params_dict['VDD'], 'k-', linewidth=1)
    ax3.set_xlabel('Time [ms]', fontsize=11)
    ax3.set_ylabel('V_Y [V]', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.05, params_dict['VDD'] * 1.1)
    
    # ========== 右列: ズーム ==========
    ax4 = fig.add_subplot(gs[0, 1])
    ax5 = fig.add_subplot(gs[1, 1], sharex=ax4)
    ax6 = fig.add_subplot(gs[2, 1], sharex=ax4)
    
    # ズーム範囲を設定（スパイク間隔のばらつきを確認するため50msに延長）
    t_zoom_end = min(50.0, t_ms[-1])
    mask_zoom = t_ms <= t_zoom_end
    
    # V_Vズーム
    ax4.plot(t_ms[mask_zoom], vV[mask_zoom], 'r-', linewidth=1.5)
    ax4.axhline(params_dict['VTH'], color='g', linestyle='--', alpha=0.7, label='VTH')
    ax4.axhline(params_dict['VC'], color='orange', linestyle='--', alpha=0.7, label='VC')
    ax4.set_ylabel('V_V [V]', fontsize=11)
    ax4.set_title(f'Zoomed View (0-{t_zoom_end:.1f}ms)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, params_dict['VDD'])
    ax4.legend(fontsize=9)
    
    # V_Uズーム
    ax5.plot(t_ms[mask_zoom], vU[mask_zoom], 'b-', linewidth=1.5)
    
    # スパイク時の背景色
    y_zoom = y[mask_zoom]
    spike_idx = np.where(y_zoom > 0.5)[0]
    if len(spike_idx) > 0:
        spike_starts = spike_idx[np.insert(np.diff(spike_idx) > 1, 0, True)]
        spike_ends = spike_idx[np.append(np.diff(spike_idx) > 1, True)]
        for start, end in zip(spike_starts, spike_ends):
            ax5.axvspan(t_ms[mask_zoom][start], t_ms[mask_zoom][end], 
                       alpha=0.2, color='orange')
    
    ax5.set_ylabel('V_U [V]', fontsize=11)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, params_dict['VDD'])
    
    # V_Yズーム
    ax6.plot(t_ms[mask_zoom], y[mask_zoom] * params_dict['VDD'], 'k-', linewidth=1.5)
    ax6.fill_between(t_ms[mask_zoom], 0, y[mask_zoom] * params_dict['VDD'], 
                     alpha=0.3, color='gray')
    ax6.set_xlabel('Time [ms]', fontsize=11)
    ax6.set_ylabel('V_Y [V]', fontsize=11)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(-0.05, params_dict['VDD'] * 1.1)
    
    plt.suptitle(f'TSN Time Series Analysis: VC = {vc:.3f}V', 
                fontsize=14, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def plot_time_series_comparison(data_dict, save_path='time_series_comparison.png'):
    """比較：Regular Spiking vs Chaotic"""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    
    print("\n=== Creating Comparison Plot ===")
    
    for col_idx, (vc, label) in enumerate([(0.10, 'Regular Spiking'), 
                                            (0.13, 'Period-2 CH'), 
                                            (0.14, 'Chaotic')]):
        if vc not in data_dict:
            continue
            
        data = data_dict[vc]
        t_ms = data['t'] * 1000
        
        # 最初の50msを表示（スパイク間隔のばらつきを確認）
        mask = t_ms <= 50.0
        t_plot = t_ms[mask]
        vV_plot = data['vV'][mask]
        vU_plot = data['vU'][mask]
        
        # V_V
        axes[0, col_idx].plot(t_plot, vV_plot, 'r-', linewidth=1.5)
        axes[0, col_idx].axhline(data['params']['VTH'], color='g', 
                                linestyle='--', alpha=0.7)
        axes[0, col_idx].set_title(f'{label}\n(VC={vc:.2f}V)', 
                                   fontsize=12, fontweight='bold')
        axes[0, col_idx].set_ylabel('V_V [V]', fontsize=11)
        axes[0, col_idx].grid(True, alpha=0.3)
        axes[0, col_idx].set_ylim(0, data['params']['VDD'])
        
        if col_idx == 0:
            axes[0, col_idx].text(-0.15, 0.5, 'Membrane\nPotential', 
                                 transform=axes[0, col_idx].transAxes,
                                 fontsize=11, fontweight='bold',
                                 rotation=90, va='center')
        
        # V_U
        axes[1, col_idx].plot(t_plot, vU_plot, 'b-', linewidth=1.5)
        axes[1, col_idx].set_xlabel('Time [ms]', fontsize=11)
        axes[1, col_idx].set_ylabel('V_U [V]', fontsize=11)
        axes[1, col_idx].grid(True, alpha=0.3)
        axes[1, col_idx].set_ylim(0, data['params']['VDD'])
        
        if col_idx == 0:
            axes[1, col_idx].text(-0.15, 0.5, 'Recovery\nVariable', 
                                 transform=axes[1, col_idx].transAxes,
                                 fontsize=11, fontweight='bold',
                                 rotation=90, va='center')
    
    plt.suptitle('TSN Spiking Dynamics Comparison', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()

def plot_single_vc_comparison(data_dict, vc, save_path=None):
    """単一のVCについて、V_Y=0とV_Y=VDDを並べて表示（正規化なし）"""
    
    if vc not in data_dict:
        print(f"Error: VC={vc} not found")
        return
    
    data = data_dict[vc]
    vV = data['vV']
    vU = data['vU']
    params_dict = data['params']
    
    params = TSNParameters()
    for key, value in params_dict.items():
        setattr(params, key, value)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # v-nullcline
    vV_null, vU_null = compute_v_nullcline(params)
    
    # ========== 左: V_Y = 0 ==========
    vV_grid = np.linspace(0, params.VDD, 25)
    vU_grid = np.linspace(0, params.VDD, 25)
    VV_grid, VU_grid = np.meshgrid(vV_grid, vU_grid)
    
    dVV0 = np.zeros_like(VV_grid)
    dVU0 = np.zeros_like(VU_grid)
    
    for i in range(VV_grid.shape[0]):
        for j in range(VV_grid.shape[1]):
            dVV0[i, j] = fp(VV_grid[i, j], params.VL, VU_grid[i, j], 
                           params.iIN, params.Cv)
            dVU0[i, j] = fv(0, VU_grid[i, j], params.VA, params.VD, params.Cu)
    
    # スケーリング（ベクトルを5倍大きく表示）
    max_dVV0 = np.max(np.abs(dVV0))
    max_dVU0 = np.max(np.abs(dVU0))
    scale_factor0 = max(max_dVV0, max_dVU0)
    scale_value0 = 10 * scale_factor0 if scale_factor0 > 0 else 1  # 50→10に変更（5倍大きく）
    
    ax1.quiver(VV_grid, VU_grid, dVV0, dVU0,
              color='gray', alpha=0.6, scale=scale_value0, width=0.005)
    ax1.plot(vV_null, vU_null, 'r--', linewidth=2.5, label='v-nullcline')
    ax1.plot(vV, vU, 'k-', linewidth=2, label='Trajectory', alpha=0.8)
    ax1.axvline(params.VTH, color='green', linestyle='--', linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('V_V [V]', fontsize=13)
    ax1.set_ylabel('V_U [V]', fontsize=13)
    ax1.set_title(f'VC = {vc:.3f}V (V_Y = 0)', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, params.VDD)
    ax1.set_ylim(0, params.VDD)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.legend(fontsize=10)
    
    # ========== 右: V_Y = VDD ==========
    dVVvdd = np.zeros_like(VV_grid)
    dVUvdd = np.zeros_like(VU_grid)
    
    for i in range(VV_grid.shape[0]):
        for j in range(VV_grid.shape[1]):
            dVVvdd[i, j] = fp(VV_grid[i, j], params.VL, VU_grid[i, j], 
                             params.iIN, params.Cv)
            dVUvdd[i, j] = fv(1, VU_grid[i, j], params.VA, params.VD, params.Cu)
    
    # スケーリング（ベクトルを5倍大きく表示）
    max_dVVvdd = np.max(np.abs(dVVvdd))
    max_dVUvdd = np.max(np.abs(dVUvdd))
    scale_factorvdd = max(max_dVVvdd, max_dVUvdd)
    scale_valuevdd = 10 * scale_factorvdd if scale_factorvdd > 0 else 1  # 50→10に変更（5倍大きく）
    
    ax2.quiver(VV_grid, VU_grid, dVVvdd, dVUvdd,
              color='orange', alpha=0.6, scale=scale_valuevdd, width=0.005)
    ax2.plot(vV_null, vU_null, 'r--', linewidth=2.5, label='v-nullcline')
    ax2.plot(vV, vU, 'k-', linewidth=2, label='Trajectory', alpha=0.8)
    ax2.axvline(params.VTH, color='green', linestyle='--', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('V_V [V]', fontsize=13)
    ax2.set_ylabel('V_U [V]', fontsize=13)
    ax2.set_title(f'VC = {vc:.3f}V (V_Y = VDD)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, params.VDD)
    ax2.set_ylim(0, params.VDD)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

# ========================================
# メイン実行
# ========================================

if __name__ == "__main__":
    
    vc_list = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
    data_file = 'tsn_simulation_data_complete.pkl'
    
    # ========================================
    # データ準備
    # ========================================
    
    if not os.path.exists(data_file):
        print("Data file not found. Running simulations...")
        data_dict = run_and_save_simulations(vc_list, data_file)
    else:
        print("Loading existing data...")
        data_dict = load_simulation_data(data_file)
    
    # ========================================
    # 1. 位相平面プロット（正規化なし）
    # ========================================
    
    print("\n" + "=" * 80)
    print("1. Creating Phase Plane Plots (No Normalization)")
    print("=" * 80)
    
    plot_phase_plane_no_normalization(
        data_dict, vc_list,
        save_path_vy0='phase_plane_vy0_nonorm.png',
        save_path_vyvdd='phase_plane_vyvdd_nonorm.png'
    )
    
    # ========================================
    # 2. 単一VC比較（V_Y=0 vs V_Y=VDD）
    # ========================================
    
    print("\n" + "=" * 80)
    print("2. Creating Single VC Comparison (VC=0.14V)")
    print("=" * 80)
    
    plot_single_vc_comparison(
        data_dict, 0.14,
        save_path='phase_plane_vc014_comparison_nonorm.png'
    )
    
    # ========================================
    # 3. 全VCの時系列プロット
    # ========================================
    
    print("\n" + "=" * 80)
    print("3. Creating Time Series for All VCs")
    print("=" * 80)
    
    plot_time_series_all(
        data_dict, vc_list,
        save_path='time_series_all_vc.png'
    )
    
    # ========================================
    # 4. 詳細時系列（3段プロット）
    # ========================================
    
    print("\n" + "=" * 80)
    print("4. Creating Detailed Time Series (3-panel)")
    print("=" * 80)
    
    plot_time_series_detailed(
        data_dict, 0.14,
        save_path='time_series_vc014_detailed.png'
    )
    
    # ========================================
    # 5. ズーム付き時系列
    # ========================================
    
    print("\n" + "=" * 80)
    print("5. Creating Zoomed Time Series")
    print("=" * 80)
    
    plot_time_series_zoom(
        data_dict, 0.14,
        save_path='time_series_vc014_zoom.png'
    )
    
    # ========================================
    # 6. 比較プロット（RS vs CH vs Chaotic）
    # ========================================
    
    print("\n" + "=" * 80)
    print("6. Creating Comparison Plot")
    print("=" * 80)
    
    plot_time_series_comparison(
        data_dict,
        save_path='time_series_comparison.png'
    )
    
    # ========================================
    # サマリー
    # ========================================
    
    print("\n" + "=" * 80)
    print("ALL PLOTS COMPLETE!")
    print("=" * 80)
    
    print("\n【生成されたファイル】")
    print("\n■ 位相平面プロット（ベクトル場：正規化なし）:")
    print("  1. phase_plane_vy0_nonorm.png (7 plots, V_Y=0)")
    print("  2. phase_plane_vyvdd_nonorm.png (7 plots, V_Y=VDD)")
    print("  3. phase_plane_vc014_comparison_nonorm.png (side-by-side)")
    
    print("\n■ 時系列プロット:")
    print("  4. time_series_all_vc.png (All 7 VCs, V_V and V_U)")
    print("  5. time_series_vc014_detailed.png (3-panel: V_V, V_U, V_Y)")
    print("  6. time_series_vc014_zoom.png (Full + Zoomed view)")
    print("  7. time_series_comparison.png (RS vs CH vs Chaotic)")
    
    print("\n■ 合計: 7個の画像ファイル")
    print("  - 位相平面: 14枚（7×2パターン）+ 比較1枚")
    print("  - 時系列: 4個のファイル")
    
    print("\n" + "=" * 80)
    print("Key Features:")
    print("  ✓ Vector fields without normalization (actual magnitude)")
    print("  ✓ Two states: V_Y=0 (normal) and V_Y=VDD (spike)")
    print("  ✓ Time series with V_V, V_U, and V_Y")
    print("  ✓ Comparison across different dynamics (RS, CH, Chaotic)")
    print("=" * 80)
    
    # ========================================
    # 追加の解析情報
    # ========================================
    
    print("\n" + "=" * 80)
    print("Analysis Summary")
    print("=" * 80)
    
    for vc in vc_list:
        if vc in data_dict:
            data = data_dict[vc]
            n_spikes = np.sum(np.diff(data['y']) > 0.5)
            sim_time_ms = data['t'][-1] * 1000
            firing_rate = n_spikes / (sim_time_ms / 1000)  # Hz
            
            print(f"\nVC = {vc:.3f}V:")
            print(f"  Spikes: {n_spikes}")
            print(f"  Firing rate: {firing_rate:.2f} Hz")
            print(f"  Simulation time: {sim_time_ms:.1f} ms")
    
    print("\n" + "=" * 80)