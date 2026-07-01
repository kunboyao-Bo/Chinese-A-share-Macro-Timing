"""
大盘宏观择时模型 - 三因子等权打分 + 动量过滤器
信号: pe_pct + ppi_raw + turnover_zscore
过滤: 动量过滤器（OR逻辑：上涨月数>=7 或 12个月收益>-20%）
输出: 强烈买入 / 买入 / 不操作 / 卖出 / 强烈卖出
"""

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# 1. 读取数据
# ──────────────────────────────────────────────
DATA_ROOT = r'输入文件路径'
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

# 2.4 动量过滤器（双条件 OR）
price_mom = price[['date', 'close']].sort_values('date').copy()
price_mom['ret_1m']  = price_mom['close'].pct_change(1)
price_mom['ret_12m'] = price_mom['close'].pct_change(12).shift(1)  # 跳过最近1个月，无前视
price_mom['mom_up_count'] = price_mom['ret_1m'].rolling(12).apply(
    lambda x: (x > 0).sum(), raw=True
)
# 条件A：持续性动量，12个月中上涨月份数 >= 7
# 条件B：幅度动量，过去12个月收益 > -20%
# OR逻辑：任一满足即放行，都不满足才降档
price_mom['cond_a']     = price_mom['mom_up_count'] >= 7
price_mom['cond_b']     = price_mom['ret_12m'] > -0.20
price_mom['mom_filter'] = price_mom['cond_a'] | price_mom['cond_b']

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
      .merge(pe[['date', 'pe_pct']],                           on='date', how='left')
      .merge(ppi[['date', 'ppi_raw']],                         on='date', how='left')
      .merge(sent[['date', 'turnover_zscore']],                on='date', how='left')
      .merge(price_mom[['date', 'mom_up_count', 'ret_12m', 'cond_a', 'cond_b', 'mom_filter']], on='date', how='left')
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
df['signal_raw'] = df['total_score_int'].map(SCORE_MAP)   # 过滤前原始信号

# ──────────────────────────────────────────────
# 5.1 动量过滤器：买入信号降档
# 规则：趋势偏弱（mom_filter=False）时
#   强烈买入 → 买入
#   买入     → 不操作
#   其他信号不变
# ──────────────────────────────────────────────
DOWNGRADE_MAP = {'强烈买入': '买入', '买入': '不操作'}

def apply_mom_filter(row):
    if row['signal_raw'] in ('数据不足', None):
        return row['signal_raw']
    # 卖出方向不干预
    if row['signal_raw'] in ('卖出', '强烈卖出', '不操作'):
        return row['signal_raw']
    # 买入方向：趋势偏弱时降档
    if row['mom_filter'] is False or row['mom_filter'] != row['mom_filter']:  # False或NaN
        return DOWNGRADE_MAP.get(row['signal_raw'], row['signal_raw'])
    return row['signal_raw']

df['signal'] = df.apply(apply_mom_filter, axis=1)

# 任一信号缺失时标记
df.loc[df['total_score'].isna(), 'signal'] = '数据不足'

# ──────────────────────────────────────────────
# 6. 输出
# ──────────────────────────────────────────────
DISPLAY_COLS = ['date', 'close', 'pe_pct', 'ppi_raw', 'turnover_zscore',
                's_pe', 's_ppi', 's_turnover', 'total_score',
                'mom_up_count', 'ret_12m', 'cond_a', 'cond_b', 'mom_filter', 'signal_raw', 'signal']

print("=" * 120)
print("大盘宏观择时模型（含动量过滤器）")
print("=" * 120)
print(df[DISPLAY_COLS].to_string(index=False))

# 最新信号
latest = df[~df['signal'].isin(['数据不足'])].iloc[-1]
print("\n" + "=" * 120)
print(f"【最新信号】{latest['date'].strftime('%Y-%m')}  →  {latest['signal']}"
      + (f"  （原始: {latest['signal_raw']}，动量降档）"
         if latest['signal'] != latest['signal_raw'] else ""))
print(f"  PE分位:    {latest['pe_pct']:.2f}  (得分 {int(latest['s_pe']):+d})")
print(f"  PPI:       {latest['ppi_raw']:.2f}%  (得分 {int(latest['s_ppi']):+d})")
print(f"  成交额Z:   {latest['turnover_zscore']:.2f}  (得分 {int(latest['s_turnover']):+d})")
print(f"  综合得分:  {int(latest['total_score'])}")
print(f"  动量上涨月数: {int(latest['mom_up_count']) if pd.notna(latest['mom_up_count']) else 'N/A'}/12  "
      f"过滤器: {'放行' if latest['mom_filter'] else '降档'}")
print("=" * 120)

# 统计过滤器触发次数
filtered = df[(df['signal_raw'].isin(['强烈买入','买入'])) &
              (df['signal'] != df['signal_raw'])]
print(f"\n动量过滤器共触发 {len(filtered)} 次降档")
print(filtered[['date','signal_raw','signal','mom_up_count']].to_string(index=False))

print("\n最终信号分布：")
print(df['signal'].value_counts())

# 保存结果
df[DISPLAY_COLS].to_csv(f'{DATA_ROOT}\\macro_timing_result_momentum.csv', index=False)
print("\n结果已保存至 macro_timing_result_1.xlsx")
