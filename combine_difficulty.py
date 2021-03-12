import pandas as pd 

tenk = 'kitti_{}_val_difficulty_v2.csv'.format('10k')
sevenk = 'kitti_{}_val_difficulty_v2.csv'.format('7k')
fourk = 'kitti_{}_val_difficulty_v2.csv'.format('4k')
sixk = 'kitti_{}_val_difficulty_v2.csv'.format('6k')
normal = 'kitti_val_difficulty_v2.csv'


df_10 = pd.read_csv(tenk, index_col=0)
df_7 = pd.read_csv(sevenk, index_col=0)
df_4 = pd.read_csv(fourk, index_col=0)
df_6 = pd.read_csv(sixk, index_col=0)
df_ctrl = pd.read_csv(normal, index_col=0)

df_10['attenuation'] = 'severe'
df_7['attenuation'] = 'moderate'
df_4['attenuation'] = 'low'
df_6['attenuation'] = 'uniform'
df_ctrl['attenuation'] = 'none'

new_df = pd.concat([df_10, df_7, df_4, df_6, df_ctrl], ignore_index=True)
new_df.to_csv('kitti_val_difficulty_w_attenuation_v2.csv')