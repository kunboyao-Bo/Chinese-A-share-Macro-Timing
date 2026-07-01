"""
大盘宏观择时模型 - 三因子等权打分
信号: pe_pct + ppi_raw + turnover_zscore
输出: 强烈买入 / 买入 / 不操作 / 卖出 / 强烈卖出
"""

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# 1. 读取数据
# ──────────────────────────────────────────────
DATA_ROOT = r'D:\学习\量化数据\股票宏观'
pmi  = pd.read_excel(f'{DATA_ROOT}\\cn_pmi.xlsx',        parse_dates=['Date']).rename(columns={'Date':'date'})
ppi  = pd.read_excel(f'{DATA_ROOT}\\cn_ppi.xlsx',        parse_dates=['Date']).rename(columns={'Date':'date'})
pe   = pd.read_excel(f'{DATA_ROOT}\\index_PE.xlsx',      parse_dates=['Date']).rename(columns={'Date':'date'})
price= pd.read_excel(f'{DATA_ROOT}\\index_price.xlsx',   parse_dates=['trade_date']).rename(columns={'trade_date':'date'})
sent = pd.read_excel(f'{DATA_ROOT}\\index_sentiment.xlsx', parse_dates=['Date']).rename(columns={'Date':'date'})

for df in [pe, ppi, sent, price]:
    df['date'] = df['date'] + pd.offsets.MonthEnd(0)

# ──────────────────────────────────────────────
# 2. 构造信号
# ──────────────────────────────────────────────

# 2.1 PE历史分位（expanding，无前视）
pe = pe.sort_values('date')
pe['pe_pct'] = pe['PE'].expanding().rank(pct=True)

# 2.2 PPI（滞后1个月，反映实际发布时间）
ppi = ppi.sort_values('date')
ppi['ppi_raw'] = ppi['PPI']
ppi['date'] = ppi['date'] + pd.offsets.MonthEnd(1)   # lag 1个月

# 2.3 成交额Z-score（12个月滚动）
sent = sent.sort_values('date')
sent['turnover_zscore'] = (
    (sent['Turnover'] - sent['Turnover'].rolling(12).mean())
    / sent['Turnover'].rolling(12).std()
)

# ──────────────────────────────────────────────
# 3. 打分规则
# ──────────────────────────────────────────────
def score_pe(x):
    if pd.isna(x):   return np.nan
    if x < 0.30:     return  1     # 低估
    if x > 0.70:     return -1     # 高估
    return 0

def score_ppi(x):
    if pd.isna(x):   return np.nan
    if x < -1.0:     return  1     # 通缩，均值回归预期
    if x > 3.0:      return -1     # 过热
    return 0

def score_turnover(x):
    if pd.isna(x):   return np.nan
    if x < -0.8:     return  1     # 冷清，筹码出清
    if x > 1.2:      return -1     # 过热
    return 0

SCORE_MAP = {
    2: '强烈买入',
    1: '买入',
    0: '不操作',
   -1: '卖出',
   -2: '强烈卖出',
   -3: '强烈卖出',
    3: '强烈买入',
}

# ──────────────────────────────────────────────
# 4. 合并
# ──────────────────────────────────────────────
df = (price[['date', 'close']]
      .merge(pe[['date', 'pe_pct']],              on='date', how='left')
      .merge(ppi[['date', 'ppi_raw']],            on='date', how='left')
      .merge(sent[['date', 'turnover_zscore']],   on='date', how='left')
      .sort_values('date')
      .reset_index(drop=True)
)

# ──────────────────────────────────────────────
# 5. 计算得分与信号
# ──────────────────────────────────────────────
df['s_pe']       = df['pe_pct'].apply(score_pe)
df['s_ppi']      = df['ppi_raw'].apply(score_ppi)
df['s_turnover'] = df['turnover_zscore'].apply(score_turnover)

df['total_score'] = df[['s_pe', 's_ppi', 's_turnover']].sum(axis=1, skipna=False)
df['total_score_int'] = df['total_score'].fillna(0).astype(int)
df['signal'] = df['total_score_int'].map(SCORE_MAP)

# 任一信号缺失时标记
df.loc[df['total_score'].isna(), 'signal'] = '数据不足'

# ──────────────────────────────────────────────
# 6. 输出
# ──────────────────────────────────────────────
DISPLAY_COLS = ['date', 'close', 'pe_pct', 'ppi_raw', 'turnover_zscore',
                's_pe', 's_ppi', 's_turnover', 'total_score', 'signal']

print("=" * 100)
print("大盘宏观择时模型")
print("=" * 100)
print(df[DISPLAY_COLS].to_string(index=False))

# 最新信号
latest = df[df['signal'] != '数据不足'].iloc[-1]
print("\n" + "=" * 100)
print(f"【最新信号】{latest['date'].strftime('%Y-%m')}  →  {latest['signal']}")
print(f"  PE分位: {latest['pe_pct']:.2f}  (得分 {int(latest['s_pe']):+d})")
print(f"  PPI:    {latest['ppi_raw']:.2f}%  (得分 {int(latest['s_ppi']):+d})")
print(f"  成交额Z: {latest['turnover_zscore']:.2f}  (得分 {int(latest['s_turnover']):+d})")
print(f"  综合得分: {int(latest['total_score'])}")
print("=" * 100)

# 信号分布统计
print("\n信号分布：")
print(df['signal'].value_counts())

# 保存结果
df[DISPLAY_COLS].to_csv(f'{DATA_ROOT}\\macro_timing_result.csv', index=False)
print("\n结果已保存至 macro_timing_result.xlsx")
