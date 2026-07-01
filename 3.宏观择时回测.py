"""
大盘宏观择时回测
- 信号来源: macro_timing_result.xlsx
- 执行时点: 信号月次月open
- 仓位映射: 强烈买入100% / 买入70% / 不操作30% / 卖出10% / 强烈卖出0%
- 空仓收益: 0
- 成本: 买入0.02% / 卖出0.07%
- Benchmark: 同期满仓持有
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings('ignore')

# ── 中文字体 ──────────────────────────────────
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
except:
    pass

# ──────────────────────────────────────────────
# 1. 读取数据
# ──────────────────────────────────────────────
DATA_ROOT = r'输入文件存放地址'
result = pd.read_excel(f'{DATA_ROOT}\\macro_timing_result.xlsx')
price  = pd.read_excel(f'{DATA_ROOT}\\index_price.xlsx')

result['date'] = pd.to_datetime(result['date']) + pd.offsets.MonthEnd(0)
price['date']  = pd.to_datetime(price['trade_date']) + pd.offsets.MonthEnd(0)
price = price[['date','open','close']].sort_values('date').reset_index(drop=True)

# ──────────────────────────────────────────────
# 2. 仓位映射
# ──────────────────────────────────────────────
POSITION_MAP = {
    '强烈买入': 1.00,
    '买入':    0.70,
    '不操作':  0.30,
    '卖出':    0.10,
    '强烈卖出': 0.00,
}

valid = result[result['signal'].isin(POSITION_MAP)].copy()
valid['target_pos'] = valid['signal'].map(POSITION_MAP)
valid = valid[['date','signal','target_pos']].sort_values('date').reset_index(drop=True)

# ──────────────────────────────────────────────
# 3. 构建执行序列
# 信号在t月末产生 → t+1月的open执行
# 持有至t+2月open（即下一个信号的执行时点）
# ──────────────────────────────────────────────

# 把open价格对齐到信号月（即信号月的next_open = 次月open）
price_map = dict(zip(price['date'], price['open']))
close_map = dict(zip(price['date'], price['close']))

# 为每条信号找到执行价（次月open）
valid['exec_date']  = valid['date'] + pd.offsets.MonthEnd(1)
valid['exec_price'] = valid['exec_date'].map(price_map)

# 删除找不到执行价的行（数据末尾）
valid = valid.dropna(subset=['exec_price']).reset_index(drop=True)

# ──────────────────────────────────────────────
# 4. 逐月回测
# ──────────────────────────────────────────────
COST_BUY  = 0.0002   # 买入单边
COST_SELL = 0.0007   # 卖出单边（含印花税）

nav        = 1.0     # 策略净值
bm_nav     = 1.0     # benchmark净值
cur_pos    = 0.0     # 当前仓位
cur_price  = None    # 当前持仓成本价（open价）

records = []

for i, row in valid.iterrows():
    exec_price = row['exec_price']
    target_pos = row['target_pos']
    exec_date  = row['exec_date']

    # 计算本期持仓收益（从上一个exec到本期exec）
    if i == 0:
        # 第一期：建仓，无持仓收益
        period_ret = 0.0
    else:
        prev_exec_price = valid.loc[i-1, 'exec_price']
        price_ret = exec_price / prev_exec_price - 1
        period_ret = cur_pos * price_ret

    # 更新净值（先计算持仓收益）
    nav     *= (1 + period_ret)
    bm_ret   = (exec_price / valid.loc[0, 'exec_price'] - 1) if i > 0 else 0.0
    bm_nav   = (1 + bm_ret)  # benchmark从第一期exec价格起算

    # 计算交易成本
    pos_change = target_pos - cur_pos
    cost = 0.0
    if pos_change > 0:
        cost = pos_change * COST_BUY
    elif pos_change < 0:
        cost = abs(pos_change) * COST_SELL
    nav *= (1 - cost)

    records.append({
        'date':       exec_date,
        'signal':     row['signal'],
        'target_pos': target_pos,
        'exec_price': exec_price,
        'period_ret': round(period_ret, 6),
        'cost':       round(cost, 6),
        'nav':        round(nav, 6),
        'bm_nav':     round(bm_nav, 6),
    })

    cur_pos   = target_pos
    cur_price = exec_price

bt = pd.DataFrame(records)

# Benchmark净值序列（满仓持有，从第一个exec_price起）
first_price = bt.loc[0, 'exec_price']
bt['bm_nav'] = price[price['date'].isin(bt['date'])].set_index('date')['open'].reindex(bt['date'].values).values / first_price
bt['bm_nav'] = bt['bm_nav'].ffill()

# ──────────────────────────────────────────────
# 5. 绩效指标
# ──────────────────────────────────────────────
def calc_metrics(nav_series, dates, label):
    nav   = nav_series.values
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    total_ret   = nav[-1] / nav[0] - 1
    ann_ret     = (nav[-1] / nav[0]) ** (1 / years) - 1
    monthly_ret = pd.Series(nav).pct_change().dropna()
    ann_vol     = monthly_ret.std() * np.sqrt(12)
    sharpe      = ann_ret / ann_vol if ann_vol > 0 else np.nan
    rolling_max = pd.Series(nav).cummax()
    drawdown    = (pd.Series(nav) / rolling_max - 1)
    max_dd      = drawdown.min()
    calmar      = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    print(f"\n{'─'*40}")
    print(f"  {label}")
    print(f"{'─'*40}")
    print(f"  区间:       {dates.iloc[0].strftime('%Y-%m')} ~ {dates.iloc[-1].strftime('%Y-%m')}")
    print(f"  累计收益:   {total_ret*100:.2f}%")
    print(f"  年化收益:   {ann_ret*100:.2f}%")
    print(f"  年化波动:   {ann_vol*100:.2f}%")
    print(f"  Sharpe:     {sharpe:.3f}")
    print(f"  最大回撤:   {max_dd*100:.2f}%")
    print(f"  Calmar:     {calmar:.3f}")
    return {'ann_ret': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe, 'max_dd': max_dd}

print("=" * 50)
print("宏观择时回测绩效")
print("=" * 50)
m1 = calc_metrics(bt['nav'],    bt['date'], '策略')
m2 = calc_metrics(bt['bm_nav'], bt['date'], 'Benchmark（满仓持有）')

# ──────────────────────────────────────────────
# 6. 绘图
# ──────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                         gridspec_kw={'height_ratios': [3, 1, 1]})
fig.suptitle('大盘宏观择时回测', fontsize=15, fontweight='bold', y=0.98)

# 6.1 净值曲线
ax1 = axes[0]
ax1.plot(bt['date'], bt['nav'],    label='策略',       color='#E84040', linewidth=2)
ax1.plot(bt['date'], bt['bm_nav'], label='Benchmark',  color='#4472C4', linewidth=1.5, alpha=0.8)
ax1.set_ylabel('净值', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title('净值曲线', fontsize=11)

# 标注绩效
txt = (f"策略: 年化{m1['ann_ret']*100:.1f}%  Sharpe {m1['sharpe']:.2f}  最大回撤{m1['max_dd']*100:.1f}%\n"
       f"基准: 年化{m2['ann_ret']*100:.1f}%  Sharpe {m2['sharpe']:.2f}  最大回撤{m2['max_dd']*100:.1f}%")
ax1.text(0.02, 0.97, txt, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 6.2 仓位变化
ax2 = axes[1]
colors = {'强烈买入':'#C00000','买入':'#FF6666','不操作':'#AAAAAA','卖出':'#6699FF','强烈卖出':'#003399'}
bar_colors = bt['signal'].map(colors).fillna('#AAAAAA')
ax2.bar(bt['date'], bt['target_pos'], color=bar_colors, width=20, alpha=0.85)
ax2.set_ylabel('仓位', fontsize=11)
ax2.set_ylim(0, 1.1)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_title('仓位变化', fontsize=11)

# 图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=v, label=k) for k,v in colors.items()]
ax2.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=5)

# 6.3 回撤
ax3 = axes[2]
strat_dd  = (bt['nav'] / bt['nav'].cummax() - 1) * 100
bm_dd     = (bt['bm_nav'] / bt['bm_nav'].cummax() - 1) * 100
ax3.fill_between(bt['date'], strat_dd, 0, alpha=0.5, color='#E84040', label='策略回撤')
ax3.fill_between(bt['date'], bm_dd,   0, alpha=0.3, color='#4472C4', label='基准回撤')
ax3.set_ylabel('回撤 (%)', fontsize=11)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_title('回撤', fontsize=11)

for ax in axes:
    ax.set_xlim(bt['date'].min(), bt['date'].max())

plt.tight_layout()
plt.show()


# 保存回测明细
print(bt)
bt.to_csv(f'{DATA_ROOT}\\macro_backtest_detail.csv', index=False)
print("明细已保存至 macro_backtest_detail.xlsx")

# ──────────────────────────────────────────────
# 7. 信号预测效果显著性检验
# ──────────────────────────────────────────────
from scipy import stats

# 7.1 构造信号+未来收益对照表

sig_ret = bt[['date', 'signal', 'exec_price']].copy()
sig_ret['ret_1m'] = sig_ret['exec_price'].pct_change(1).shift(-1)
sig_ret['ret_3m'] = sig_ret['exec_price'].pct_change(3).shift(-3)
sig_ret = sig_ret.dropna(subset=['signal'])

SIGNAL_ORDER = ['强烈买入', '买入', '不操作', '卖出', '强烈卖出']

print("\n" + "=" * 80)
print("信号预测效果显著性检验")
print("=" * 80)

for horizon, col in [('1个月', 'ret_1m'), ('3个月', 'ret_3m')]:
    print(f"\n{'─' * 70}")
    print(f"  预测期：{horizon}")
    print(f"{'─' * 70}")

    # ── 检验一：每档收益均值是否显著异于0 ──
    print(f"\n  【检验一】各档位收益均值 vs 0（单样本t检验）")
    print(f"  {'信号':<8} {'N':>4} {'均值':>8} {'标准差':>8} {'t统计量':>9} {'p值':>8} {'显著':>4}")
    print(f"  {'─' * 60}")

    group_stats = {}
    for sig in SIGNAL_ORDER:
        sub = sig_ret[sig_ret['signal'] == sig][col].dropna()
        if len(sub) < 3:
            continue
        t, p = stats.ttest_1samp(sub, 0)
        stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
        print(f"  {sig:<8} {len(sub):>4} {sub.mean() * 100:>7.2f}% {sub.std() * 100:>7.2f}% "
              f"{t:>9.3f} {p:>8.4f} {stars:>4}")
        group_stats[sig] = sub

    # ── 检验二：相邻档位收益是否单调递减 ──
    print(f"\n  【检验二】相邻档位单调性检验（双样本t检验）")
    print(f"  {'对比组':<20} {'均值差':>8} {'t统计量':>9} {'p值':>8} {'单调':>4} {'显著':>4}")
    print(f"  {'─' * 60}")

    pairs = [('强烈买入', '买入'), ('买入', '不操作'), ('不操作', '卖出'), ('卖出', '强烈卖出')]
    for sig_high, sig_low in pairs:
        if sig_high not in group_stats or sig_low not in group_stats:
            continue
        a, b = group_stats[sig_high], group_stats[sig_low]
        t, p = stats.ttest_ind(a, b, equal_var=False)  # Welch t检验
        diff = a.mean() - b.mean()
        monotone = '✓' if diff > 0 else '✗'
        stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
        print(f"  {sig_high} > {sig_low:<8} {diff * 100:>7.2f}% {t:>9.3f} {p:>8.4f} "
              f"{monotone:>4} {stars:>4}")

print("\n  显著性标注: *** p<0.01  ** p<0.05  * p<0.1")
