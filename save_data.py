from __future__ import print_function, division
import pandas as pd, numpy as np, scipy.stats as ss
import torch

def simple_imputer(df):
    idx = pd.IndexSlice
    df = df.copy()
    if len(df.columns.names) > 2: df.columns = df.columns.droplevel(('label', 'LEVEL1', 'LEVEL2'))
    
    df_out = df.loc[:, idx[:, ['mean', 'count']]]
    icustay_means = df_out.loc[:, idx[:, 'mean']].groupby(ID_COLS).mean()
    
    df_out.loc[:,idx[:,'mean']] = df_out.loc[:,idx[:,'mean']].groupby(ID_COLS).fillna(
        method='ffill'
    ).groupby(ID_COLS).fillna(icustay_means).fillna(0)
    
    df_out.loc[:, idx[:, 'count']] = (df.loc[:, idx[:, 'count']] > 0).astype(float)
    df_out.rename(columns={'count': 'mask'}, level='Aggregation Function', inplace=True)
    
    is_absent = (1 - df_out.loc[:, idx[:, 'mask']])
    hours_of_absence = is_absent.cumsum()
    time_since_measured = hours_of_absence - hours_of_absence[is_absent==0].fillna(method='ffill')
    time_since_measured.rename(columns={'mask': 'time_since_measured'}, level='Aggregation Function', inplace=True)

    df_out = pd.concat((df_out, time_since_measured), axis=1)
    df_out.loc[:, idx[:, 'time_since_measured']] = df_out.loc[:, idx[:, 'time_since_measured']].fillna(100)
    
    df_out.sort_index(axis=1, inplace=True)
    return df_out


DATA_FILEPATH     = 'MIMIC/all_hourly_data.h5'
MIMIC_FOLDERPATH = '/home/huan/Documents/MIMIC3/1.4'
GAP_TIME          = 6  # In hours
WINDOW_SIZE       = 24 # In hours
SEED              = 1
ID_COLS           = ['subject_id', 'hadm_id', 'icustay_id']
np.random.seed(SEED)
torch.manual_seed(SEED)
data_full_lvl2 = pd.read_hdf(DATA_FILEPATH, 'vitals_labs')
statics        = pd.read_hdf(DATA_FILEPATH, 'patients')

Ys = statics[statics.max_hours > WINDOW_SIZE + GAP_TIME][['mort_hosp', 'mort_icu', 'los_icu']]
Ys['los_3'] = Ys['los_icu'] > 3
Ys['los_7'] = Ys['los_icu'] > 7
Ys.drop(columns=['los_icu'], inplace=True)
Ys.astype(float)

lvl2  = data_full_lvl2[
    (data_full_lvl2.index.get_level_values('icustay_id').isin(set(Ys.index.get_level_values('icustay_id')))) &
    (data_full_lvl2.index.get_level_values('hours_in') < WINDOW_SIZE)]
# raw.columns = raw.columns.droplevel(level=['label', 'LEVEL1', 'LEVEL2'])

train_frac, test_frac = 0.7, 0.3
lvl2_subj_idx, Ys_subj_idx = [df.index.get_level_values('subject_id') for df in (lvl2, Ys)]
lvl2_subjects = set(lvl2_subj_idx)
# assert lvl2_subjects == set(Ys_subj_idx), "Subject ID pools differ!"


np.random.seed(SEED)
subjects, N = np.random.permutation(list(lvl2_subjects)), len(lvl2_subjects)
N_train, N_test = int(train_frac * N), int(test_frac * N)
train_subj = subjects[:N_train]
test_subj  = subjects[N_train:]

[(lvl2_train, lvl2_test), (Ys_train, Ys_test)] = [
    [df[df.index.get_level_values('subject_id').isin(s)] for s in (train_subj,test_subj)] \
    for df in (lvl2, Ys)]

idx = pd.IndexSlice
lvl2_means, lvl2_stds = lvl2_train.loc[:, idx[:,'mean']].mean(axis=0), lvl2_train.loc[:, idx[:,'mean']].std(axis=0)


lvl2_train.loc[:, idx[:,'mean']] = (lvl2_train.loc[:, idx[:,'mean']] - lvl2_means)/lvl2_stds
lvl2_test.loc[:, idx[:,'mean']] = (lvl2_test.loc[:, idx[:,'mean']] - lvl2_means)/lvl2_stds

lvl2_train, lvl2_test = [
    simple_imputer(df) for df in (lvl2_train, lvl2_test)
]
lvl2_flat_train, lvl2_flat_test = [
    df.pivot_table(index=['subject_id', 'hadm_id', 'icustay_id'], columns=['hours_in']) for df in (
        lvl2_train, lvl2_test
    )
]

lvl2_flat_train.to_csv('m_train.csv',index=True)
lvl2_flat_test.to_csv('m_test.csv',index=True)
Ys_train.to_csv('my_train.csv',index=True)
Ys_test.to_csv('my_test.csv',index=True)


tmp_train = pd.read_csv('m_train.csv',index_col=[0,1,2], header=[0,1,2])
tmp_f_dim = tmp_train.shape[1]
label_train =  pd.read_csv('my_train.csv',index_col=[0,1,2])
tmp_test = pd.read_csv('m_test.csv',index_col=[0,1,2], header=[0,1,2])
label_test =  pd.read_csv('my_test.csv',index_col=[0,1,2])
Diagnoses = pd.read_csv(os.path.join(MIMIC_FOLDERPATH, 'DIAGNOSES_ICD.csv'))

stat_train = statics[statics.index.isin(tmp_train.index)]
train = tmp_train.join(stat_train['diagnosis_at_admission'])
train.columns.values[-1] = ('static','code','diagnosis')
train = train.join(stat_train['ethnicity'])
train.columns.values[-1] = ('static','code','ethnicity')
train = train.join(stat_train['admission_type'])
train.columns.values[-1] = ('static','code','admission_type')
train.columns = pd.MultiIndex.from_tuples(train.columns, \
                                          names=['LEVEL2','	Aggregation Function','hours_in'])

stat_train = train['static']


stat_test = statics[statics.index.isin(tmp_test.index)]
test = tmp_test.join(stat_test['diagnosis_at_admission'])
test.columns.values[-1] = ('static','code','diagnosis')
test = test.join(stat_train['ethnicity'])
test.columns.values[-1] = ('static','code','ethnicity')
test = test.join(stat_train['admission_type'])
test.columns.values[-1] = ('static','code','admission_type')
test.columns = pd.MultiIndex.from_tuples(test.columns, \
                                          names=['LEVEL2','	Aggregation Function','hours_in'])

stat_test= test['static']

stat_train.to_csv('ms_train.csv',index=True)
stat_test.to_csv('ms_test.csv',index=True)