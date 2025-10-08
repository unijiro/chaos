"""
VC=0.14V専用の高速プロット生成
ベクトル場を5倍大きく、時系列を100msで表示
"""

import numpy as np
import matplotlib.pyplot as plt
from TSN_sim import (
    TSNParameters, g1, g2, g3, g4, g5, fp, fv,
    simulate_tsn_matlab_style, compute_v_nullcline
)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_enhanced_comparison(vc=0.14, save_path='phase_plane_vc014_enhanced.png'):
    """
    強化版位相平面プロット
    - ベクトル場を5倍大きく表示
    - グリッドを細かく
    """
    
    print(f"\n{'='*80}")
    print(f"Enhanced Phase Plane Plot for VC={vc}V")
    print(f"{'='*80}")
    
    # パラメータ設定
    params = TSNParameters()
    params.VC = vc
    params.time_length = 100e-3  # 100ms
    
    # シミュレーション実行
    print("\nRunning simulation (100ms)...")
    t, vV, vU, y = simulate_tsn_matlab_style(params)
    
    n_spikes = np.sum(np.diff(y) > 0.5)
    print(f"Spikes detected: {n_spikes}")
    print(f"Simulation time: {params.time_length*1000:.1f} ms")
    
    if n_spikes > 0:
        spike_indices = np.where(np.diff(y) > 0.5)[0]
        isis = np.diff(t[spike_indices])
        print(f"ISI: mean={np.mean(isis)*1e3:.4f}ms, std={np.std(isis)*1e3:.4f}ms")
        print(f"CV: {np.std(isis)/np.mean(isis):.4f}")
    
    # プロット作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # v-nullcline計算
    vV_null, vU_null = compute_v_nullcline(params)
    
    # ========================================
    # 左: V_Y = 0（通常状態）
    # ========================================
    print("\nCreating phase plane for V_Y=0...")
    
    # ベクトル場（グリッドを細かく）
    vV_grid = np.linspace(0, params.VDD, 30)
    vU_grid = np.linspace(0, params.VDD, 30)
    VV_grid, VU_grid = np.meshgrid(vV_grid, vU_grid)
    
    dVV0 = np.zeros_like(VV_grid)
    dVU0 = np.zeros_like(VU_grid)
    
    for i in range(VV_grid.shape[0]):
        for j in range(VV_grid.shape[1]):
            dVV0[i, j] = fp(VV_grid[i, j], params.VL, VU_grid[i, j], 
                           params.iIN, params.Cv)
            dVU0[i, j] = fv(0, VU_grid[i, j], params.VA, params.VD, params.Cu)
    
    # スケーリング（5倍大きく）
    max_dVV0 = np.max(np.abs(dVV0))
    max_dVU0 = np.max(np.abs(dVU0))
    scale_factor0 = max(max_dVV0, max_dVU0)
    scale_value0 = 10 * scale_factor0 if scale_factor0 > 0 else 1
    
    ax1.quiver(VV_grid, VU_grid, dVV0, dVU0,
              color='gray', alpha=0.65, scale=scale_value0, width=0.006,
              headwidth=4, headlength=5, headaxislength=4.5)
    
    ax1.plot(vV_null, vU_null, 'r-', linewidth=3, label='v-nullcline', zorder=3)
    ax1.plot(vV, vU, 'k-', linewidth=2.5, label='Trajectory', alpha=0.85, zorder=4)
    ax1.axvline(params.VTH, color='green', linestyle='--', 
               linewidth=2.5, alpha=0.8, label=f'VTH={params.VTH:.2f}V', zorder=2)
    ax1.plot(vV[0], vU[0], 'go', markersize=10, label='Start', 
            markeredgecolor='darkgreen', markeredgewidth=2, zorder=5)
    
    ax1.set_xlabel('V_V [V]', fontsize=14, fontweight='bold')
    ax1.set_ylabel('V_U [V]', fontsize=14, fontweight='bold')
    ax1.set_title(f'Phase Plane: VC={vc:.3f}V (V_Y=0, Normal State)', 
                 fontsize=15, fontweight='bold')
    ax1.set_xlim(0, params.VDD)
    ax1.set_ylim(0, params.VDD)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal')
    ax1.legend(fontsize=11, loc='upper left')
    
    # ========================================
    # 右: V_Y = VDD（スパイク後）
    # ========================================
    print("Creating phase plane for V_Y=VDD...")
    
    dVVvdd = np.zeros_like(VV_grid)
    dVUvdd = np.zeros_like(VU_grid)
    
    for i in range(VV_grid.shape[0]):
        for j in range(VV_grid.shape[1]):
            dVVvdd[i, j] = fp(VV_grid[i, j], params.VL, VU_grid[i, j], 
                             params.iIN, params.Cv)
            dVUvdd[i, j] = fv(1, VU_grid[i, j], params.VA, params.VD, params.Cu)
    
    # スケーリング（5倍大きく）
    max_dVVvdd = np.max(np.abs(dVVvdd))
    max_dVUvdd = np.max(np.abs(dVUvdd))
    scale_factorvdd = max(max_dVVvdd, max_dVUvdd)
    scale_valuevdd = 10 * scale_factorvdd if scale_factorvdd > 0 else 1
    
    ax2.quiver(VV_grid, VU_grid, dVVvdd, dVUvdd,
              color='orange', alpha=0.65, scale=scale_valuevdd, width=0.006,
              headwidth=4, headlength=5, headaxislength=4.5)
    
    ax2.plot(vV_null, vU_null, 'r-', linewidth=3, label='v-nullcline', zorder=3)
    ax2.plot(vV, vU, 'k-', linewidth=2.5, label='Trajectory', alpha=0.85, zorder=4)
    ax2.axvline(params.VTH, color='green', linestyle='--', 
               linewidth=2.5, alpha=0.8, label=f'VTH={params.VTH:.2f}V', zorder=2)
    
    ax2.set_xlabel('V_V [V]', fontsize=14, fontweight='bold')
    ax2.set_ylabel('V_U [V]', fontsize=14, fontweight='bold')
    ax2.set_title(f'Phase Plane: VC={vc:.3f}V (V_Y=VDD, Post-Spike State)', 
                 fontsize=15, fontweight='bold')
    ax2.set_xlim(0, params.VDD)
    ax2.set_ylim(0, params.VDD)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_aspect('equal')
    ax2.legend(fontsize=11, loc='upper left')
    
    plt.suptitle(f'TSN Circuit Enhanced Phase Plane Analysis\n' +
                f'{n_spikes} spikes in {params.time_length*1000:.0f}ms ' +
                f'(Vector field magnified 5×)',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()
    
    return t, vV, vU, y

def plot_long_time_series(t, vV, vU, y, params, vc=0.14, 
                          save_path='time_series_vc014_long.png'):
    """
    100msの長時間時系列プロット
    スパイク間隔のばらつきを確認
    """
    
    print("\nCreating long time series plot...")
    
    t_ms = t * 1000
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    # ========== V_V ==========
    axes[0].plot(t_ms, vV, 'r-', linewidth=1.2)
    axes[0].axhline(params.VTH, color='green', linestyle='--', 
                   linewidth=2, alpha=0.7, label='VTH')
    axes[0].axhline(params.VC, color='orange', linestyle=':', 
                   linewidth=1.5, alpha=0.6, label='VC')
    axes[0].set_ylabel('V_V [V]', fontsize=13, fontweight='bold')
    axes[0].set_title(f'TSN Circuit Long Time Series (VC={vc:.3f}V, 100ms)', 
                     fontsize=15, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, params.VDD*1.05)
    
    # ========== V_U ==========
    axes[1].plot(t_ms, vU, 'b-', linewidth=1.2)
    
    # スパイク時の背景色
    spike_mask = y > 0.5
    spike_regions = []
    in_spike = False
    start_idx = 0
    
    for i, is_spike in enumerate(spike_mask):
        if is_spike and not in_spike:
            start_idx = i
            in_spike = True
        elif not is_spike and in_spike:
            spike_regions.append((t_ms[start_idx], t_ms[i]))
            in_spike = False
    
    for t_start, t_end in spike_regions:
        axes[1].axvspan(t_start, t_end, color='yellow', alpha=0.15)
    
    axes[1].set_ylabel('V_U [V]', fontsize=13, fontweight='bold')
    axes[1].legend(['V_U', 'Spike period'], fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, params.VDD*1.05)
    
    # ========== V_Y ==========
    axes[2].plot(t_ms, y * params.VDD, 'k-', linewidth=1.2)
    axes[2].fill_between(t_ms, 0, y * params.VDD, alpha=0.3, color='gray')
    axes[2].set_xlabel('Time [ms]', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('V_Y [V]', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-0.05, params.VDD*1.1)
    
    # スパイク統計を表示
    n_spikes = np.sum(np.diff(y) > 0.5)
    if n_spikes > 0:
        spike_indices = np.where(np.diff(y) > 0.5)[0]
        isis = np.diff(t[spike_indices]) * 1000  # ms
        
        stats_text = (f'Spikes: {n_spikes}\n'
                     f'Mean ISI: {np.mean(isis):.3f} ms\n'
                     f'Std ISI: {np.std(isis):.3f} ms\n'
                     f'CV: {np.std(isis)/np.mean(isis):.4f}')
        
        fig.text(0.98, 0.02, stats_text, 
                ha='right', va='bottom', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_isi_analysis(t, y, save_path='isi_analysis_vc014.png'):
    """ISI解析プロット"""
    
    print("\nCreating ISI analysis plot...")
    
    spike_indices = np.where(np.diff(y) > 0.5)[0]
    
    if len(spike_indices) < 2:
        print("Not enough spikes for ISI analysis")
        return
    
    spike_times = t[spike_indices] * 1000  # ms
    isis = np.diff(spike_times)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # ========== ISI系列 ==========
    axes[0].plot(range(len(isis)), isis, 'bo-', markersize=6, linewidth=1.5)
    axes[0].axhline(np.mean(isis), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(isis):.3f}ms')
    axes[0].fill_between(range(len(isis)), 
                        np.mean(isis) - np.std(isis),
                        np.mean(isis) + np.std(isis),
                        alpha=0.3, color='red', label='±1 SD')
    axes[0].set_xlabel('Spike Index', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('ISI [ms]', fontsize=13, fontweight='bold')
    axes[0].set_title('ISI Time Series', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # ========== ISIヒストグラム ==========
    axes[1].hist(isis, bins=30, color='steelblue', alpha=0.7, 
                edgecolor='black', linewidth=1.2)
    axes[1].axvline(np.mean(isis), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(isis):.3f}ms')
    axes[1].set_xlabel('ISI [ms]', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Count', fontsize=13, fontweight='bold')
    axes[1].set_title('ISI Distribution', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # ========== ISIリターンマップ ==========
    if len(isis) > 1:
        axes[2].plot(isis[:-1], isis[1:], 'bo', markersize=8, alpha=0.7)
        
        # 対角線
        isi_range = [min(isis), max(isis)]
        axes[2].plot(isi_range, isi_range, 'k--', alpha=0.5, 
                    linewidth=2, label='Identity line')
        
        axes[2].set_xlabel('ISI_n [ms]', fontsize=13, fontweight='bold')
        axes[2].set_ylabel('ISI_n+1 [ms]', fontsize=13, fontweight='bold')
        axes[2].set_title(f'ISI Return Map (CV={np.std(isis)/np.mean(isis):.4f})', 
                         fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_aspect('equal')
    
    plt.suptitle(f'ISI Analysis (VC=0.14V, {len(isis)} intervals)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def main():
    print("\n" + "="*80)
    print("Enhanced Plotting for VC=0.14V")
    print("="*80)
    
    # 1. 強化版位相平面プロット
    t, vV, vU, y = plot_enhanced_comparison(vc=0.14)
    
    params = TSNParameters()
    params.VC = 0.14
    
    # 2. 長時間時系列プロット
    plot_long_time_series(t, vV, vU, y, params)
    
    # 3. ISI解析プロット
    plot_isi_analysis(t, y)
    
    print("\n" + "="*80)
    print("All plots completed!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. phase_plane_vc014_enhanced.png")
    print("  2. time_series_vc014_long.png")
    print("  3. isi_analysis_vc014.png")
    print("="*80)

if __name__ == "__main__":
    main()
