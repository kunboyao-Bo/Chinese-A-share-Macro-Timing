"""
大盘宏观择时 - 信号IC测试
目标变量: 下N个月指数收益率 (N=1,2,3)
信号: PMI动量、PPI方向、PE历史分位、情绪Z-score
"""

import pandas as pd
import numpy as np
from scipy import stats

# ──────────────────────────────────────────────
# 1. 读取数据
# ──────────────────────────────────────────────
DATA_ROOT = r'D:\学习\量化数据\股票宏观'
pmi  = pd.read_excel(f'{DATA_ROOT}\\cn_pmi.xlsx',        parse_dates=['Date']).rename(columns={'Date':'date'})
ppi  = pd.read_excel(f'{DATA_ROOT}\\cn_ppi.xlsx',        parse_dates=['Date']).rename(columns={'Date':'date'})
pe   = pd.read_excel(f'{DATA_ROOT}\\index_PE.xlsx',      parse_dates=['Date']).rename(columns={'Date':'date'})
price= pd.read_excel(f'{DATA_ROOT}\\index_price.xlsx',   parse_dates=['trade_date']).rename(columns={'trade_date':'date'})
sent = pd.read_excel(f'{DATA_ROOT}\\index_sentiment.xlsx', parse_dates=['Date']).rename(columns={'Date':'date'})

# 统一到月末
for df in [pmi, ppi, pe, price, sent]:
    df['date'] = df['date'] + pd.offsets.MonthEnd(0)

# ──────────────────────────────────────────────
# 2. 构造目标变量: 未来N月收益率
# ──────────────────────────────────────────────
price = price[['date','close']].sort_values('date')
price['ret_12m'] = price['close'].pct_change(12).shift(-12)   # 下1个月
price['ret_2m'] = price['close'].pct_change(2).shift(-2)   # 下2个月
price['ret_3m'] = price['close'].pct_change(3).shift(-3)   # 下3个月

# ──────────────────────────────────────────────
# 3. 构造信号
# ──────────────────────────────────────────────

## 3.1 PMI信号
pmi = pmi.sort_values('date')
pmi['pmi_mom']     = pmi['PMI'].diff(1)                     # 环比变化
pmi['pmi_3m_slope']= pmi['PMI'].diff(3)                     # 3个月趋势
pmi['pmi_vs50']    = pmi['PMI'] - 50                        # 相对荣枯线

## 3.2 PPI信号
ppi = ppi.sort_values('date')
ppi['ppi_mom']     = ppi['PPI'].diff(1)                     # 环比加速/减速
ppi['ppi_3m_slope']= ppi['PPI'].diff(3)                     # 3个月趋势
ppi['ppi_raw']     = ppi['PPI']                             # 绝对值

## 3.3 PE历史分位（需要外部长历史数据叠加当前数据）
# 当前数据从2018年起，你有2008年起的完整数据后替换路径即可
# 临时用当前窗口，分位数意义有限，仅作示意
pe = pe.sort_values('date')
pe['pe_pct'] = pe['PE'].expanding().rank(pct=True)                     # 历史百分位（上线后换长窗口）
pe['pe_mom'] = pe['PE'].diff(1)                             # PE环比变化（可用短窗口）
pe['pe_zscore'] = (pe['PE'] - pe['PE'].rolling(24).mean()) / pe['PE'].rolling(24).std()

## 3.4 情绪信号（成交额Z-score）
sent = sent.sort_values('date')
sent['turnover_zscore'] = (
    (sent['Turnover'] - sent['Turnover'].rolling(12).mean())
    / sent['Turnover'].rolling(12).std()
)
sent['volume_zscore'] = (
    (sent['Volume'] - sent['Volume'].rolling(12).mean())
    / sent['Volume'].rolling(12).std()
)

# ──────────────────────────────────────────────
# 4. 合并
# ──────────────────────────────────────────────
# PPI延迟一个月对齐（反映实际发布滞后）
ppi_lagged = ppi[['date','ppi_mom','ppi_3m_slope','ppi_raw']].copy()
ppi_lagged['date'] = ppi_lagged['date'] + pd.offsets.MonthEnd(1)

df = (price
      .merge(pmi[['date','pmi_mom','pmi_3m_slope','pmi_vs50']], on='date', how='left')
      .merge(ppi_lagged,                                         on='date', how='left')
      .merge(pe[['date','pe_pct','pe_mom','pe_zscore']],         on='date', how='left')
      .merge(sent[['date','turnover_zscore','volume_zscore']],   on='date', how='left')
      .sort_values('date')
)

# ──────────────────────────────────────────────
# 5. IC计算函数
# ──────────────────────────────────────────────
def calc_ic(df, signal_col, ret_col, method='pearson'):
    """计算单信号IC及统计量"""
    sub = df[[signal_col, ret_col]].dropna()
    if len(sub) < 10:
        return None

    x, y = sub[signal_col], sub[ret_col]

    if method == 'spearman':
        ic, pval = stats.spearmanr(x, y)
    else:
        ic, pval = stats.pearsonr(x, y)

    # 滚动IC序列（用于ICIR）
    roll_ic = []
    window = 12
    for i in range(window, len(sub)):
        xi, yi = x.iloc[i-window:i], y.iloc[i-window:i]
        if xi.std() == 0 or yi.std() == 0:
            continue
        if method == 'spearman':
            r, _ = stats.spearmanr(xi, yi)
        else:
            r, _ = stats.pearsonr(xi, yi)
        roll_ic.append(r)

    roll_ic = np.array(roll_ic)
    icir = roll_ic.mean() / roll_ic.std() if roll_ic.std() > 0 else np.nan
    t_stat = ic * np.sqrt(len(sub) - 2) / np.sqrt(1 - ic**2) if abs(ic) < 1 else np.nan

    return {
        'IC均值':   round(ic, 4),
        'IC_std':  round(np.std(roll_ic), 4) if len(roll_ic) > 0 else np.nan,
        'ICIR':    round(icir, 4),
        't统计量': round(t_stat, 4),
        'p值':     round(pval, 4),
        'N':       len(sub),
        '正IC占比': round((roll_ic > 0).mean(), 3) if len(roll_ic) > 0 else np.nan,
    }

# ──────────────────────────────────────────────
# 6. 批量测试
# ──────────────────────────────────────────────
SIGNALS = [
    'pmi_mom', 'pmi_3m_slope', 'pmi_vs50',
    'ppi_mom', 'ppi_3m_slope', 'ppi_raw',
    'pe_pct', 'pe_mom', 'pe_zscore',
    'turnover_zscore', 'volume_zscore',
]
TARGETS  = ['ret_12m', 'ret_2m', 'ret_3m']
METHODS  = ['pearson', 'spearman']

results = []
for method in METHODS:
    for sig in SIGNALS:
        for tgt in TARGETS:
            r = calc_ic(df, sig, tgt, method=method)
            if r:
                results.append({'方法': method, '信号': sig, '目标': tgt, **r})

result_df = pd.DataFrame(results)

# ──────────────────────────────────────────────
# 7. 输出结果
# ──────────────────────────────────────────────

print("宏观择时信号 IC 测试结果")


for method in METHODS:
    print(f"\n{'─'*40} {method.upper()} IC {'─'*40}")
    sub = result_df[result_df['方法'] == method].copy()
    sub = sub.drop(columns='方法')
    # 按|IC均值|降序
    sub['|IC|'] = sub['IC均值'].abs()
    sub = sub.sort_values(['目标', '|IC|'], ascending=[True, False]).drop(columns='|IC|')
    # 标记显著信号
    sub['显著'] = sub['p值'].apply(lambda x: '✓' if x < 0.1 else '')
    print(sub.to_string(index=False))

# 最优信号汇总
print("\n" + "=" * 90)
print("各预测期最优信号 (按|ICIR|排序，仅显示p<0.1)")
print("=" * 90)
sig_df = result_df[result_df['p值'] < 0.1].copy()
sig_df['|ICIR|'] = sig_df['ICIR'].abs()
best = (sig_df.sort_values('|ICIR|', ascending=False)
              .groupby('目标')
              .head(5)
              .sort_values(['目标', '|ICIR|'], ascending=[True, False]))
print(best[['方法','信号','目标','IC均值','ICIR','t统计量','p值','N']].to_string(index=False))