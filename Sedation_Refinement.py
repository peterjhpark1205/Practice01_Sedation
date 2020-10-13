##################################

# REFINE RAW DATA

# This is for scenarios regarding SNUCH's Sedation and MAC treatment

# Date : 2020.09.17
# Made by : Peter JH Park

###################################


import os, sys, csv
import numpy as np
import pandas as pd
import sklearn as sk
import re
import statsmodels.formula.api as sm


SedSub = pd.read_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\RawData\\SedScene_Ult\\Sedation_Subjects.csv', encoding = 'utf-8')
SedPrice = pd.read_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\RawData\\SedScene_Ult\\Sedation_Price.csv', encoding = 'utf-8')
MACPrice = pd.read_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\RawData\\SedScene_Ult\\MAC_Price.csv', encoding = 'utf-8')
OutPrice = pd.read_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\RawData\\SedScene_Ult\\OutpatientCare_Price.csv', encoding = 'utf-8')


print(SedPrice)
print(MACPrice)
print(OutPrice)

### Preprocess the raw dataframe
SedSub['date'] = SedSub['date'].str.replace('-', '')
SedSub['date'] = SedSub['date'].astype(int)

SedSub['exam1'].fillna('noexam', inplace=True)
SedSub['exam2'].fillna('noexam', inplace=True)
SedSub['exam3'].fillna('noexam', inplace=True)
SedSub['exam4'].fillna('noexam', inplace=True)
SedSub['exam5'].fillna('noexam', inplace=True)

SedSub['exam_loc1'].fillna('noeloc', inplace=True)
SedSub['exam_loc2'].fillna('noeloc', inplace=True)
SedSub['exam_loc3'].fillna('noeloc', inplace=True)
SedSub['exam_loc4'].fillna('noeloc', inplace=True)
SedSub['exam_loc5'].fillna('noeloc', inplace=True)

SedSub['drug_midazolam']=SedSub['drug_midazolam'].replace("'", 0)

SedSub['drug_pocral'].fillna(0, inplace=True)
SedSub['drug_midazolam'].fillna(0, inplace=True)
SedSub['drug_ketamine'].fillna(0, inplace=True)
SedSub['drug_pethidine'].fillna(0, inplace=True)
SedSub['drug_etc'].fillna('nodrug', inplace=True)

SedSub['drug_pocral']=SedSub['drug_pocral'].astype(int)
SedSub['drug_midazolam']=SedSub['drug_midazolam'].astype(int)
SedSub['drug_ketamine']=SedSub['drug_ketamine'].astype(int)
SedSub['drug_pethidine']=SedSub['drug_pethidine'].astype(int)

SedSub['drug_pocral']=SedSub['drug_pocral'].astype(str)
SedSub['drug_midazolam']=SedSub['drug_midazolam'].astype(str)
SedSub['drug_ketamine']=SedSub['drug_ketamine'].astype(str)
SedSub['drug_pethidine']=SedSub['drug_pethidine'].astype(str)

SedSub['drug_pocral']=SedSub['drug_pocral'].str.replace('.0', '')
SedSub['drug_midazolam']=SedSub['drug_midazolam'].str.replace('.0', '')
SedSub['drug_ketamine']=SedSub['drug_ketamine'].str.replace('.0', '')
SedSub['drug_pethidine']=SedSub['drug_pethidine'].str.replace('.0', '')

SedSub['drug_pocral']='poc'+SedSub['drug_pocral']
SedSub['drug_midazolam']='mid'+SedSub['drug_midazolam']
SedSub['drug_ketamine']='keta'+SedSub['drug_ketamine']
SedSub['drug_pethidine']='peth'+SedSub['drug_pethidine']


SedSub['mon_start'].fillna('0:00', inplace=True)
SedSub['mon_end'].fillna('0:00', inplace=True)
SedSub['mon_span'].fillna('0:00', inplace=True)

SedSub['mon_exe'].fillna('X', inplace=True)

SedSub['fee_num'].fillna(0, inplace=True)
SedSub['fee_num'] = SedSub['fee_num'].astype(int)
SedPrice['fee_num'] = SedPrice['fee_num'].astype(int)
MACPrice['fee_num'] = MACPrice['fee_num'].astype(int)
SedSub['fee_omitreas'].fillna('noreason', inplace=True)

SedSub['note'].fillna('nonotes', inplace=True)

print(SedSub['fee_num'])
print(SedSub)
#########################################################

SedSub['exam1']=SedSub['exam1'].str.replace(', ', '/')
SedSub['exam2']=SedSub['exam2'].str.replace(', ', '/')
SedSub['exam3']=SedSub['exam3'].str.replace(', ', '/')
SedSub['exam4']=SedSub['exam4'].str.replace(', ', '/')
SedSub['exam5']=SedSub['exam5'].str.replace(', ', '/')

SedSub['exams'] = SedSub['exam1'] + ', ' + SedSub['exam2'] + ', ' + SedSub['exam3'] + ', ' + SedSub['exam4'] + ', ' + SedSub['exam5']
SedSub['exams'] = SedSub['exams'].str.replace('본원내시경', '내시경')
SedSub['exams'] = SedSub['exams'].str.replace('내시경', 'Endoscope')
SedSub['exams'] = SedSub['exams'].str.replace('혈관조영술', 'angio')
SedSub['exams'] = SedSub['exams'].str.replace('골수검사', 'bonemarrow')
SedSub['exams'] = SedSub['exams'].str.replace('요추천자', 'lumbar')
SedSub['exams'] = SedSub['exams'].str.replace('근육생검', 'musclebiopsy')
SedSub['exams'] = SedSub['exams'].str.replace('ct', 'CT')
SedSub['exams'] = SedSub['exams'].str.replace('x-ray', 'X-ray')
SedSub['exams'] = SedSub['exams'].str.replace('cardioversion', 'Cardioversion')
#########################################################


SedSub['mon_span'] = SedSub['mon_span'].str.split(':')
SedSub[['mon_span_h','mon_span_m']] = pd.DataFrame(SedSub.mon_span.tolist())
SedSub['mon_span_h'] = SedSub['mon_span_h'].astype(int)
SedSub['mon_span_m'] = SedSub['mon_span_m'].astype(int)
SedSub['mon_span'] = SedSub['mon_span_h']*60 + SedSub['mon_span_m']
SedSub['mon_span']=SedSub['mon_span'].round(2)

print(SedSub.mon_span)
#########################################################


SedSub = SedSub[(SedSub.mon_exe=='O')|(SedSub.mon_exe=='X')]
#########################################################


SedSub.loc[SedSub['inout'] != '외래', 'inout'] = 'in'
SedSub.loc[SedSub['inout'] == '외래', 'inout'] = 'out'

print(SedSub.inout)
#########################################################


sedsub_day = SedSub[SedSub['p_age'].str.contains('d|일|D')].reset_index()
sedsub_month = SedSub[SedSub['p_age'].str.contains('m|M')].reset_index()

sedsub_day['p_age'] = sedsub_day['p_age'].str.replace('d', '')
sedsub_day['p_age'] = sedsub_day['p_age'].str.replace('일', '')
sedsub_day['p_age'] = sedsub_day['p_age'].str.replace('D', '')
sedsub_month['p_age'] = sedsub_month['p_age'].str.replace('m', '')
sedsub_month['p_age'] = sedsub_month['p_age'].str.replace('M', '')

sedsub_day['p_age'] = sedsub_day['p_age'].astype(int)
sedsub_month['p_age'] = sedsub_month['p_age'].astype(int)

sedsub_day['p_age'] = sedsub_day['p_age'] / 360
sedsub_month['p_age'] = sedsub_month['p_age'] / 12

print(sedsub_day['p_age'])
print(sedsub_month['p_age'])

SedSub = SedSub[~SedSub.p_age.str.endswith(('d', '일', 'D', 'm', 'M'))]
SedSub['p_age'] = SedSub['p_age'].astype(int)

SedSub = pd.concat([SedSub, sedsub_month, sedsub_day], sort=True).reset_index()

SedSub.p_age = SedSub.p_age.round(3)
print(SedSub.p_age)
#########################################################


SedSub.loc[(SedSub['fee_sed']=='X') & (SedSub['p_age']<=0.078), 'fee_num'] = 1
SedSub.loc[(SedSub['fee_sed']=='X') & (0.078<SedSub['p_age']) & (SedSub['p_age']<1), 'fee_num'] = 2
SedSub.loc[(SedSub['fee_sed']=='X') & (1<=SedSub['p_age']) & (SedSub['p_age']<6), 'fee_num'] = 3
SedSub.loc[(SedSub['fee_sed']=='X') & (SedSub['p_age']>=6), 'fee_num'] = 4

SedSub.loc[(SedSub['p_name']=='양재혁') & (SedSub['p_age']==2), 'fee_num'] = 3 # adjustment
#########################################################


SedSub.out_num=np.nan

SedSub.loc[(SedSub['p_age']<1)&(SedSub['inout']=='out'), 'out_num'] = 11
SedSub.loc[(1<=SedSub['p_age'])&(SedSub['p_age']<6)&(SedSub['inout']=='out'), 'out_num'] = 12
SedSub.loc[(SedSub['p_age']>=6)&(SedSub['inout']=='out'), 'out_num'] = 13
SedSub['out_num'].fillna(0, inplace=True)
#########################################################


SedSub = SedSub.merge(SedPrice, on=['fee_num'], how='left')
SedSub = SedSub.merge(MACPrice, on=['fee_num'], how='left')
SedSub = SedSub.merge(OutPrice, on=['out_num'], how='left')

SedSub['fee_macscore'].fillna(0, inplace=True)
SedSub['fee_tmacscore'].fillna(0, inplace=True)
SedSub['fee_macpriceadd'].fillna(0, inplace=True)
SedSub['fee_tmacpriceadd'].fillna(0, inplace=True)
SedSub['out_price'].fillna(0, inplace=True)
#########################################################


SedSub.loc[((SedSub.fee_tmacpriceadd)==0)|((SedSub.mon_span-30)<=0), 'fee_tmacpriceaddr'] = 0
SedSub.loc[((SedSub.fee_tmacpriceadd)!=0)&((SedSub.mon_span-30)>0)&(((SedSub.mon_span-30)%15)!=0), 'fee_tmacpriceaddr'] = SedSub.fee_tmacpriceadd*(((SedSub.mon_span-30)//15)+1)
SedSub.loc[((SedSub.fee_tmacpriceadd)!=0)&((SedSub.mon_span-30)>0)&(((SedSub.mon_span-30)%15)==0), 'fee_tmacpriceaddr'] = SedSub.fee_tmacpriceadd*((SedSub.mon_span-30)//15)

SedSub['fee_macpricetot']=SedSub['fee_macpriceadd']+SedSub['fee_tmacpriceaddr']
#########################################################


SedSub.loc[(SedSub.fee_num==1)|(SedSub.fee_num==2)|(SedSub.fee_num==3)|(SedSub.fee_num==4), 'fee_tsedscore'] = (SedSub.fee_tmacscore*(SedSub.fee_sedscore/SedSub.fee_macscore)).round(2)
SedSub.loc[~(SedSub.fee_num==1)|(SedSub.fee_num==2)|(SedSub.fee_num==3)|(SedSub.fee_num==4), 'fee_tsedscore'] = 0

SedSub['fee_tsedprice'] = (SedSub['fee_tsedscore']*76.2).round(-1)


SedSub.loc[(SedSub.fee_tsedscore==0)|((SedSub.fee_tsedscore!=0)&((SedSub.mon_span-30)<=0)), 'fee_tsedpricer'] = 0
SedSub.loc[(SedSub.fee_tsedscore!=0)&((SedSub.mon_span-30)>0)&(((SedSub.mon_span-30)%15)!=0), 'fee_tsedpricer'] = SedSub.fee_tsedprice*(((SedSub.mon_span-30)//15)+1)
SedSub.loc[(SedSub.fee_tsedscore!=0)&((SedSub.mon_span-30)>0)&(((SedSub.mon_span-30)%15)==0), 'fee_tsedpricer'] = SedSub.fee_tsedprice*((SedSub.mon_span-30)//15)

SedSub['fee_sedpricetot'] = SedSub['fee_sedprice'] + SedSub['fee_tsedpricer']

#########################################################
DrugSpan=SedSub[['date', 'inout', 'p_id', 'p_name', 'p_sex', 'p_age', 'exam1', 'exam2', 'exam3', 'exam4', 'exam5', 'exams',
                 'drug_pocral', 'drug_midazolam', 'drug_ketamine', 'drug_pethidine', 'drug_etc', 'mon_span', 'mon_exe']]

DrugSpan=DrugSpan[DrugSpan.drug_etc=='nodrug']

DrugSpan=DrugSpan[['date', 'inout', 'p_id', 'p_name', 'p_sex', 'p_age', 'exam1', 'exam2', 'exam3', 'exam4', 'exam5', 'exams',
                   'drug_pocral', 'drug_midazolam', 'drug_ketamine', 'drug_pethidine', 'mon_span', 'mon_exe']]

SedSub_d = SedSub[['date', 'inout', 'p_id', 'p_name', 'p_sex', 'p_age', 'out_num', 'out_price', 'exam1', 'exam2', 'exam3', 'exam4', 'exam5', 'exams',
                     'mon_span', 'mon_exe', 'fee_sed', 'fee_omitreas', 'fee_num', 'fee_sedscore', 'fee_tsedscore', 'fee_macscore', 'fee_tmacscore',
                     'fee_sedprice', 'fee_tsedprice', 'fee_tsedpricer', 'fee_sedpricetot',
                     'fee_macpriceadd', 'fee_tmacpriceadd', 'fee_tmacpriceaddr', 'fee_macpricetot']]


SedSub['date'] = SedSub['date'].astype(str)
SedSub['date'] = SedSub['date'].str[:6]
SedSub['date'] = SedSub['date'].astype(int)




####casenum adjustment####
Adjustment4Cases_case_out=SedSub.where(SedSub.inout=='out').groupby('date').agg({'p_id':pd.Series.count}).reset_index()
Adjustment4Cases_case_in=SedSub.where(SedSub.inout=='in').groupby('date').agg({'p_id':pd.Series.count}).reset_index()
Adjustment4Cases_case_all=SedSub.groupby('date').agg({'p_id':pd.Series.count}).reset_index()

Adjustment4Cases_case_out=Adjustment4Cases_case_out.rename(columns={'p_id':'f_case_out'})
Adjustment4Cases_case_in=Adjustment4Cases_case_in.rename(columns={'p_id':'f_case_in'})
Adjustment4Cases_case_all=Adjustment4Cases_case_all.rename(columns={'p_id':'f_case_all'})

Adjustment4Cases_case=Adjustment4Cases_case_out.merge(Adjustment4Cases_case_in, on='date', how='left').merge(Adjustment4Cases_case_all, on='date', how='left')


Adjustment4Cases_monexe_out=SedSub.where((SedSub.inout=='out')&(SedSub.mon_exe=='O')).groupby('date').agg({'mon_exe':pd.Series.count}).reset_index()
Adjustment4Cases_monexe_in=SedSub.where((SedSub.inout=='in')&(SedSub.mon_exe=='O')).groupby('date').agg({'mon_exe':pd.Series.count}).reset_index()
Adjustment4Cases_monexe_all=SedSub.where(SedSub.mon_exe=='O').groupby('date').agg({'mon_exe':pd.Series.count}).reset_index()

Adjustment4Cases_monexe_out=Adjustment4Cases_monexe_out.rename(columns={'mon_exe':'monexe_out'})
Adjustment4Cases_monexe_in=Adjustment4Cases_monexe_in.rename(columns={'mon_exe':'monexe_in'})
Adjustment4Cases_monexe_all=Adjustment4Cases_monexe_all.rename(columns={'mon_exe':'monexe_all'})

Adjustment4Cases_monexe=Adjustment4Cases_monexe_out.merge(Adjustment4Cases_monexe_in, on='date', how='left').merge(Adjustment4Cases_monexe_all, on='date', how='left')


Adjustment4Cases=Adjustment4Cases_case.merge(Adjustment4Cases_monexe, on='date', how='left')


########################################################################UPDATED MONTHLY#############################################################################
monrate_out=[0.471, 0.490, 0.482, 0.436, 0.441, 0.414, 0.400, 0.379, 0.331, 0.426, 0.416, 0.417, 0.427, 0.420, 0.403, 0.391]
monrate_in=[0.974, 0.980, 0.984, 0.956, 0.923, 0.947, 0.959, 0.978, 0.942, 0.968, 0.995, 0.980, 0.981, 0.977, 0.978, 0.969]
monrate_all=[0.669, 0.688, 0.675, 0.634, 0.629, 0.626, 0.618, 0.600, 0.554, 0.655, 0.668, 0.634, 0.650, 0.636, 0.615, 0.605]
########################################################################UPDATED MONTHLY#############################################################################

monrate={'monrate_out':monrate_out, 'monrate_in':monrate_in, 'monrate_all':monrate_all}
monrate=pd.DataFrame(monrate, columns=['monrate_out', 'monrate_in', 'monrate_all'])

Adjustment4Cases=pd.concat([Adjustment4Cases, monrate], axis=1)

Adjustment4Cases.date=Adjustment4Cases.date.astype(int)
Adjustment4Cases.set_index('date', drop=True, inplace=True)
Adjustment4Cases=Adjustment4Cases.drop(202009)

Adjustment4Cases['r_case_out'] = np.nan
Adjustment4Cases['r_case_in'] = np.nan
Adjustment4Cases['r_case_all'] = np.nan
Adjustment4Cases['adjust_out'] = np.nan
Adjustment4Cases['adjust_in'] = np.nan
Adjustment4Cases['adjust_all'] = np.nan

Adjustment4Cases.r_case_out = Adjustment4Cases.monexe_out/Adjustment4Cases.monrate_out
Adjustment4Cases.r_case_in = Adjustment4Cases.monexe_in/Adjustment4Cases.monrate_in
Adjustment4Cases.r_case_all = Adjustment4Cases.monexe_all/Adjustment4Cases.monrate_all

Adjustment4Cases.adjust_out = Adjustment4Cases.r_case_out/Adjustment4Cases.f_case_out
Adjustment4Cases.adjust_in = Adjustment4Cases.r_case_in/Adjustment4Cases.f_case_in
Adjustment4Cases.adjust_all = Adjustment4Cases.r_case_all/Adjustment4Cases.f_case_all

#Adjustment4Cases.reset_index(inplace=True)

with pd.option_context('display.max_rows', None,'display.max_columns', None):
    print(Adjustment4Cases)

Adjustment4Cases=Adjustment4Cases.drop(['f_case_out','f_case_in','f_case_all',
                                        'monexe_out','monexe_in','monexe_all',
                                        'monrate_out','monrate_in','monrate_all',
                                        'r_case_out','r_case_in','r_case_all'], axis=1)

print(Adjustment4Cases)






SedSub.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\SedScene_Ult\\SedSub.csv', index=True, encoding='utf-8-sig')
SedSub_d.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\SedScene_Ult\\SedSub_d.csv', index=True, encoding='utf-8-sig')
DrugSpan.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\SedScene_Ult\\DrugSpan.csv', index=True, encoding='utf-8-sig')
Adjustment4Cases.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\SedScene_Ult\\Adjustment4Cases.csv', index=True, encoding='utf-8-sig')





