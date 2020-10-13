###################################

# This is for scenarios regarding SNUCH's Sedation and MAC treatment

# Date : 2020.07.21
# Made by : Peter JH Park

###################################


import os, sys, csv
import numpy as np
import pandas as pd
import sklearn as sk
import re
import statsmodels.formula.api as sm


SedSub = pd.read_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\RawData\\20200721_SedScene\\Sedation_Subjects.csv', encoding = 'utf-8')
SedPrice = pd.read_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\RawData\\20200721_SedScene\\Sedation_Price.csv', encoding = 'utf-8')
MACPrice = pd.read_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\RawData\\20200721_SedScene\\MAC_Price.csv', encoding = 'utf-8')
OutPrice = pd.read_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\RawData\\20200721_SedScene\\OutpatientCare_Price.csv', encoding = 'utf-8')


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

SedSub.date_num=np.nan
SedSub.loc[(SedSub.date%200000==1905), 'date_num']=1
SedSub.loc[(SedSub.date%200000==1906), 'date_num']=2
SedSub.loc[(SedSub.date%200000==1907), 'date_num']=3
SedSub.loc[(SedSub.date%200000==1908), 'date_num']=4
SedSub.loc[(SedSub.date%200000==1909), 'date_num']=5
SedSub.loc[(SedSub.date%200000==1910), 'date_num']=6
SedSub.loc[(SedSub.date%200000==1911), 'date_num']=7
SedSub.loc[(SedSub.date%200000==1912), 'date_num']=8
SedSub.loc[(SedSub.date%200000==2001), 'date_num']=9
SedSub.loc[(SedSub.date%200000==2002), 'date_num']=10
SedSub.loc[(SedSub.date%200000==2003), 'date_num']=11
SedSub.loc[(SedSub.date%200000==2004), 'date_num']=12
SedSub.loc[(SedSub.date%200000==2005), 'date_num']=13
SedSub.loc[(SedSub.date%200000==2006), 'date_num']=14
SedSub.loc[(SedSub.date%200000==2007), 'date_num']=15
SedSub.loc[(SedSub.date%200000==2008), 'date_num']=16
SedSub['date_num'].fillna(0, inplace=True)
#########################################################


SedSub = SedSub[['date_num', 'date', 'inout', 'p_id', 'p_name', 'p_sex', 'p_age', 'out_num', 'out_price', 'exam1', 'exam2', 'exam3', 'exam4', 'exam5', 'exams',
                 'mon_span', 'mon_exe', 'fee_sed', 'fee_omitreas', 'fee_num', 'fee_sedscore', 'fee_tsedscore', 'fee_macscore', 'fee_tmacscore',
                 'fee_sedprice', 'fee_tsedprice', 'fee_tsedpricer', 'fee_sedpricetot',
                 'fee_macpriceadd', 'fee_tmacpriceadd', 'fee_tmacpriceaddr', 'fee_macpricetot']]

SedSub = SedSub[SedSub.date_num!=0]





print(SedSub.fee_sedprice)
print(SedSub.info())


### Existig Result
result_now = pd.DataFrame(columns=['date_num', 'date', 'c_day', 'out_casenum', 'out_exenum', 'out_monspan', 'out_noconsent', 'out_exerev', 'out_caserev',
                                   'in_casenum', 'in_exenum', 'in_monspan', 'in_noconsent', 'in_exerev', 'in_caserev',
                                   'tot_casenum', 'tot_exenum', 'tot_monspan', 'tot_yesmonnoconsent', 'tot_noconsent', 'tot_yesfee','tot_exerev', 'tot_caserev', 'num_under30span'])


result_now['date_num'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
result_now['c_day'] = [21, 19, 23, 21, 19, 20, 21, 21, 20, 20, 22, 20, 19, 22, 23]

result_now.set_index('date_num', inplace=True)

print(result_now)
################################################################################################

index_num = 0

#1905

for index_num in range(len(result_now['c_day'])):
    result_now.loc[index_num+1,'date']=SedSub[(SedSub.date_num==index_num+1)].date.unique()
    result_now.loc[index_num+1,'out_casenum']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')].p_id.count()
    result_now.loc[index_num+1,'out_exenum']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')&(SedSub.mon_exe=='O')].p_id.count()
    result_now.loc[index_num+1,'out_monspan']=(SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')&(SedSub.mon_exe=='O')].mon_span.sum()
                                             /(SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')&(SedSub.mon_exe=='O')].mon_span.count()+0.00001)).round(2)
    result_now.loc[index_num+1, 'out_noconsent']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')&(SedSub.fee_sed=='X')&(SedSub.fee_omitreas=='동의서')].p_id.count()
    result_now.loc[index_num+1,'out_exerev']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')&((SedSub.fee_sed=='O')|(SedSub.fee_sed=='해당없음'))].fee_sedprice.sum()
    result_now.loc[index_num+1,'out_caserev']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')].fee_sedprice.sum()
    result_now.loc[index_num+1,'in_casenum']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')].p_id.count()
    result_now.loc[index_num+1,'in_exenum']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')&(SedSub.mon_exe=='O')].p_id.count()
    result_now.loc[index_num+1,'in_monspan']=(SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')&(SedSub.mon_exe=='O')].mon_span.sum()
                                            /(SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')&(SedSub.mon_exe=='O')].mon_span.count()+0.00001)).round(2)
    result_now.loc[index_num+1, 'in_noconsent']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')&(SedSub.fee_sed=='X')&(SedSub.fee_omitreas=='동의서')].p_id.count()
    result_now.loc[index_num+1,'in_exerev']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')&((SedSub.fee_sed=='O')|(SedSub.fee_sed=='해당없음'))].fee_sedprice.sum()
    result_now.loc[index_num+1,'in_caserev']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')].fee_sedprice.sum()
    result_now.loc[index_num+1,'tot_casenum']=SedSub[(SedSub.date_num==index_num+1)].p_id.count()
    result_now.loc[index_num+1,'tot_exenum']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')].p_id.count()
    result_now.loc[index_num+1,'tot_monspan']=(SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')].mon_span.sum()
                                             /(SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')].mon_span.count()+0.00001)).round(2)
    result_now.loc[index_num+1, 'tot_yesmonnoconsent']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')&(SedSub.fee_sed=='X')&(SedSub.fee_omitreas=='동의서')].p_id.count()
    result_now.loc[index_num+1, 'tot_noconsent']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.fee_sed=='X')&(SedSub.fee_omitreas=='동의서')].p_id.count()
    result_now.loc[index_num+1,'tot_yesfee']=SedSub[(SedSub.date_num==index_num+1)&((SedSub.fee_sed=='O')|(SedSub.fee_sed=='해당없음'))].fee_sed.count()
    result_now.loc[index_num+1,'tot_exerev']=SedSub[(SedSub.date_num==index_num+1)&((SedSub.fee_sed=='O')|(SedSub.fee_sed=='해당없음'))].fee_sedprice.sum()
    result_now.loc[index_num+1,'tot_caserev']=SedSub[(SedSub.date_num==index_num+1)].fee_sedprice.sum()


print(result_now)




## Scenario1

result1 = pd.DataFrame(columns=['date_num', 'date', 'c_day', 'out_outcasenum', 'out_outrev',
                                'MRI_outcasenum', 'MRI_outmonspan', 'MRI_outrev', 'MRI_incasenum', 'MRI_inmonspan', 'MRI_inrev', 'MRI_totcasenum', 'MRI_totmonspan', 'MRI_totrev',
                                'angio_outcasenum', 'angio_outmonspan', 'angio_outrev', 'angio_incasenum', 'angio_inmonspan', 'angio_inrev', 'angio_totcasenum', 'angio_totmonspan', 'angio_totrev',
                                'bonemarrow_outcasenum', 'bonemarrow_outmonspan', 'bonemarrow_outrev', 'bonemarrow_incasenum', 'bonemarrow_inmonspan', 'bonemarrow_inrev', 'bonemarrow_totcasenum', 'bonemarrow_totmonspan', 'bonemarrow_totrev',
                                'lumbar_outcasenum', 'lumbar_outmonspan', 'lumbar_outrev', 'lumbar_incasenum', 'lumbar_inmonspan', 'lumbar_inrev', 'lumbar_totcasenum', 'lumbar_totmonspan', 'lumbar_totrev',
                                'musclebiopsy_outcasenum', 'musclebiopsy_outmonspan', 'musclebiopsy_outrev', 'musclebiopsy_incasenum', 'musclebiopsy_inmonspan', 'musclebiopsy_inrev', 'musclebiopsy_totcasenum', 'musclebiopsy_totmonspan', 'musclebiopsy_totrev',
                                'else_outcasenum', 'else_outmonspan', 'else_outrev', 'else_incasenum', 'else_inmonspan', 'else_inrev', 'else_totcasenum', 'else_totmonspan', 'else_totrev',
                                'tot_outcasenum', 'tot_outmonspan', 'tot_outrev', 'tot_incasenum', 'tot_inmonspan', 'tot_inrev', 'tot_totcasenum', 'tot_totmonspan', 'tot_totrev'])

result1['date_num'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
result1['c_day'] = [21, 19, 23, 21, 19, 20, 21, 21, 20, 20, 22, 20, 19, 22, 23]

result1.set_index('date_num', inplace=True)

print(result1)
################################################################################################

index_num = 0
exam_name = ['MRI', 'angio', 'bonemarrow', 'lumbar', 'musclebiopsy']


for index_num in range(len(result1['c_day'])):
    result1.loc[index_num+1, 'date']=SedSub[(SedSub.date_num==index_num+1)].date.unique()
    result1.loc[index_num+1, 'out_outcasenum']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')].p_id.count()
    result1.loc[index_num+1, 'out_outrev']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')].out_price.sum()
    for e_name in exam_name:
        result1.loc[index_num+1, e_name+'_outcasenum']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.inout=='out')].p_id.count()
        result1.loc[index_num+1, e_name+'_outmonspan']=(SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.mon_exe=='O')].mon_span.sum()/
                                                        (SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.mon_exe=='O')].mon_span.count()+0.00001)).round(2)
        result1.loc[index_num+1, e_name+'_outrev']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.inout=='out')].fee_macpricetot.sum()
        result1.loc[index_num+1, e_name+'_incasenum']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.inout=='in')].p_id.count()
        result1.loc[index_num+1, e_name+'_inmonspan']=(SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.mon_exe=='O')].mon_span.sum()/
                                                       (SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.mon_exe=='O')].mon_span.count()+0.00001)).round(2)
        result1.loc[index_num+1, e_name+'_inrev']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.inout=='in')].fee_macpricetot.sum()
        result1.loc[index_num+1, e_name+'_totcasenum']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')].p_id.count()
        result1.loc[index_num+1, e_name+'_totmonspan']=(SedSub[(SedSub.date_num==index_num+1)&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.mon_exe=='O')].mon_span.sum()/
                                                        (SedSub[(SedSub.date_num==index_num+1)&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.mon_exe=='O')].mon_span.count()+0.00001)).round(2)
        result1.loc[index_num+1, e_name+'_totrev']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')].fee_macpricetot.sum()
    result1.loc[index_num+1, 'else_outcasenum']=SedSub[(SedSub.date_num==index_num+1)&
                                                  (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))
                                                  &(SedSub.inout=='out')].p_id.count()
    result1.loc[index_num+1, 'else_outmonspan']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')&
                                                  (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))
                                                  &(SedSub.mon_exe=='O')].mon_span.sum()/\
                                           SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')&
                                                  (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))
                                                  &(SedSub.mon_exe=='O')].mon_span.count()
    result1.loc[index_num+1, 'else_outrev']=SedSub[(SedSub.date_num==index_num+1)&
                                              (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                 (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                 (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                 (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                 (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))
                                              &(SedSub.inout=='out')].fee_sedprice.sum()
    result1.loc[index_num+1, 'else_incasenum']=SedSub[(SedSub.date_num==index_num+1)&
                                                 (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))
                                                 &(SedSub.inout=='in')].p_id.count()
    result1.loc[index_num+1, 'else_inmonspan']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')&
                                                 (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))
                                                 &(SedSub.mon_exe=='O')].mon_span.sum()/\
                                          SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')&
                                                 (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))
                                                 &(SedSub.mon_exe=='O')].mon_span.count()
    result1.loc[index_num+1, 'else_inrev']=SedSub[(SedSub.date_num==index_num+1)&
                                             (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))
                                             &(SedSub.inout=='in')].fee_sedprice.sum()
    result1.loc[index_num+1, 'else_totcasenum']=SedSub[(SedSub.date_num==index_num+1)&
                                                  (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))].p_id.count()
    result1.loc[index_num+1, 'else_totmonspan']=SedSub[(SedSub.date_num==index_num+1)&
                                                  (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))
                                                  &(SedSub.mon_exe=='O')].mon_span.sum()/\
                                           SedSub[(SedSub.date_num==index_num+1)&
                                                   (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                      (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                      (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                      (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                      (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))
                                                   &(SedSub.mon_exe=='O')].mon_span.count()
    result1.loc[index_num+1, 'else_totrev']=SedSub[(SedSub.date_num==index_num+1)&
                                              (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                 (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                 (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                 (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                 (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))].fee_sedprice.sum()
    result1.loc[index_num+1, 'tot_outcasenum']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')].p_id.count()
    result1.loc[index_num+1, 'tot_outmonspan']=(SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')&(SedSub.mon_exe=='O')].mon_span.sum()/
                                               (SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')&(SedSub.mon_exe=='O')].mon_span.count()+0.00001)).round(2)
    result1.loc[index_num+1, 'tot_incasenum']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')].p_id.count()
    result1.loc[index_num+1, 'tot_inmonspan']=(SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')&(SedSub.mon_exe=='O')].mon_span.sum()/
                                               (SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')&(SedSub.mon_exe=='O')].mon_span.count()+0.00001)).round(2)
    result1.loc[index_num+1, 'tot_totcasenum']=SedSub[(SedSub.date_num==index_num+1)].p_id.count()
    result1.loc[index_num+1, 'tot_totmonspan']=(SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')].mon_span.sum()/
                                               (SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')].mon_span.count()+0.00001)).round(2)
result1['tot_outrev']=result1['out_outrev']+result1['MRI_outrev']+result1['angio_outrev']+result1['bonemarrow_outrev']+result1['lumbar_outrev']+result1['musclebiopsy_outrev']+result1['else_outrev']
result1['tot_inrev']=result1['MRI_inrev']+result1['angio_inrev']+result1['bonemarrow_inrev']+result1['lumbar_inrev']+result1['musclebiopsy_inrev']+result1['else_inrev']
result1['tot_totrev']=result1['out_outrev']+result1['MRI_totrev']+result1['angio_totrev']+result1['bonemarrow_totrev']+result1['lumbar_totrev']+result1['musclebiopsy_totrev']+result1['else_totrev']


print(result1)




## Scenario2

result2 = pd.DataFrame(columns=['date_num', 'date', 'c_day',
                                'MRI_outcasenum', 'MRI_outmonspan', 'MRI_outrev', 'MRI_incasenum', 'MRI_inmonspan', 'MRI_inrev', 'MRI_totcasenum', 'MRI_totmonspan', 'MRI_totrev',
                                'CT_outcasenum', 'CT_outmonspan', 'CT_outrev', 'CT_incasenum', 'CT_inmonspan', 'CT_inrev', 'CT_totcasenum', 'CT_totmonspan', 'CT_totrev',
                                'angio_outcasenum', 'angio_outmonspan', 'angio_outrev', 'angio_incasenum', 'angio_inmonspan', 'angio_inrev', 'angio_totcasenum', 'angio_totmonspan', 'angio_totrev',
                                'bonemarrow_outcasenum', 'bonemarrow_outmonspan', 'bonemarrow_outrev', 'bonemarrow_incasenum', 'bonemarrow_inmonspan', 'bonemarrow_inrev', 'bonemarrow_totcasenum', 'bonemarrow_totmonspan', 'bonemarrow_totrev',
                                'lumbar_outcasenum', 'lumbar_outmonspan', 'lumbar_outrev', 'lumbar_incasenum', 'lumbar_inmonspan', 'lumbar_inrev', 'lumbar_totcasenum', 'lumbar_totmonspan', 'lumbar_totrev',
                                'musclebiopsy_outcasenum', 'musclebiopsy_outmonspan', 'musclebiopsy_outrev', 'musclebiopsy_incasenum', 'musclebiopsy_inmonspan', 'musclebiopsy_inrev', 'musclebiopsy_totcasenum', 'musclebiopsy_totmonspan', 'musclebiopsy_totrev',
                                'else_outcasenum', 'else_outmonspan', 'else_outrev', 'else_incasenum', 'else_inmonspan', 'else_inrev', 'else_totcasenum', 'else_totmonspan', 'else_totrev',
                                'tot_outcasenum', 'tot_outmonspan', 'tot_outrev', 'tot_incasenum', 'tot_inmonspan', 'tot_inrev', 'tot_totcasenum', 'tot_totmonspan', 'tot_totrev'])

result2['date_num'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
result2['c_day'] = [21, 19, 23, 21, 19, 20, 21, 21, 20, 20, 22, 20, 19, 22, 23]

result2.set_index('date_num', inplace=True)

print(result2)
################################################################################################

index_num = 0
exam_name2 = ['MRI', 'CT', 'angio', 'bonemarrow', 'lumbar', 'musclebiopsy']


for index_num in range(len(result2['c_day'])):
    result2.loc[index_num+1, 'date']=SedSub[(SedSub.date_num==index_num+1)].date.unique()
    for e_name in exam_name2:
        result2.loc[index_num+1, e_name+'_outcasenum']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.inout=='out')].p_id.count()
        result2.loc[index_num+1, e_name+'_outmonspan']=(SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.mon_exe=='O')].mon_span.sum()/
                                                        (SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.mon_exe=='O')].mon_span.count()+0.00001)).round(2)
        result2.loc[index_num+1, e_name+'_outsedrev']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.inout=='out')&(SedSub.fee_sed=='O')].fee_sedprice.sum()
        result2.loc[index_num+1, e_name+'_outrev']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.inout=='out')].fee_macpricetot.sum()
        result2.loc[index_num+1, e_name+'_incasenum']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.inout=='in')].p_id.count()
        result2.loc[index_num+1, e_name+'_inmonspan']=(SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.mon_exe=='O')].mon_span.sum()/
                                                       (SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.mon_exe=='O')].mon_span.count()+0.00001)).round(2)
        result2.loc[index_num+1, e_name+'_insedrev']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.inout=='in')&(SedSub.fee_sed=='O')].fee_sedprice.sum()
        result2.loc[index_num+1, e_name+'_inrev']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.inout=='in')].fee_macpricetot.sum()
        result2.loc[index_num+1, e_name+'_totcasenum']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')].p_id.count()
        result2.loc[index_num+1, e_name+'_totmonspan']=(SedSub[(SedSub.date_num==index_num+1)&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.mon_exe=='O')].mon_span.sum()/
                                                        (SedSub[(SedSub.date_num==index_num+1)&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.mon_exe=='O')].mon_span.count()+0.00001)).round(2)
        result2.loc[index_num+1, e_name+'_totsedrev']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')&(SedSub.fee_sed=='O')].fee_sedprice.sum()
        result2.loc[index_num+1, e_name+'_totrev']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.exams==e_name+', noexam, noexam, noexam, noexam')].fee_macpricetot.sum()
    result2.loc[index_num+1, 'else_outcasenum']=SedSub[(SedSub.date_num==index_num+1)&
                                                  (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='CT, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))
                                                  &(SedSub.inout=='out')].p_id.count()
    result2.loc[index_num+1, 'else_outmonspan']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')&
                                                  (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams == 'CT, noexam, noexam, noexam, noexam') |
                                                     (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))
                                                  &(SedSub.mon_exe=='O')].mon_span.sum()/\
                                           SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')&
                                                  (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams == 'CT, noexam, noexam, noexam, noexam') |
                                                     (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))
                                                  &(SedSub.mon_exe=='O')].mon_span.count()
    result2.loc[index_num+1, 'else_outrev']=SedSub[(SedSub.date_num==index_num+1)&
                                              (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                 (SedSub.exams == 'CT, noexam, noexam, noexam, noexam') |
                                                 (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                 (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                 (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                 (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))
                                              &(SedSub.inout=='out')].fee_sedprice.sum()
    result2.loc[index_num+1, 'else_incasenum']=SedSub[(SedSub.date_num==index_num+1)&
                                                 (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams == 'CT, noexam, noexam, noexam, noexam') |
                                                    (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))
                                                 &(SedSub.inout=='in')].p_id.count()
    result2.loc[index_num+1, 'else_inmonspan']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')&
                                                 (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams == 'CT, noexam, noexam, noexam, noexam') |
                                                    (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))
                                                 &(SedSub.mon_exe=='O')].mon_span.sum()/\
                                          SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')&
                                                 (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams == 'CT, noexam, noexam, noexam, noexam') |
                                                    (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                    (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))
                                                 &(SedSub.mon_exe=='O')].mon_span.count()
    result2.loc[index_num+1, 'else_inrev']=SedSub[(SedSub.date_num==index_num+1)&
                                             (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                (SedSub.exams == 'CT, noexam, noexam, noexam, noexam') |
                                                (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))
                                             &(SedSub.inout=='in')].fee_sedprice.sum()
    result2.loc[index_num+1, 'else_totcasenum']=SedSub[(SedSub.date_num==index_num+1)&
                                                  (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams == 'CT, noexam, noexam, noexam, noexam') |
                                                     (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))].p_id.count()
    result2.loc[index_num+1, 'else_totmonspan']=SedSub[(SedSub.date_num==index_num+1)&
                                                  (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams == 'CT, noexam, noexam, noexam, noexam') |
                                                     (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                     (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))
                                                  &(SedSub.mon_exe=='O')].mon_span.sum()/\
                                           SedSub[(SedSub.date_num==index_num+1)&
                                                   (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                      (SedSub.exams == 'CT, noexam, noexam, noexam, noexam') |
                                                      (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                      (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                      (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                      (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))
                                                   &(SedSub.mon_exe=='O')].mon_span.count()
    result2.loc[index_num+1, 'else_totrev']=SedSub[(SedSub.date_num==index_num+1)&
                                              (~((SedSub.exams=='MRI, noexam, noexam, noexam, noexam')|
                                                 (SedSub.exams == 'CT, noexam, noexam, noexam, noexam') |
                                                 (SedSub.exams=='angio, noexam, noexam, noexam, noexam')|
                                                 (SedSub.exams=='bonemarrow, noexam, noexam, noexam, noexam')|
                                                 (SedSub.exams=='lumbar, noexam, noexam, noexam, noexam')|
                                                 (SedSub.exams=='musclebiopsy, noexam, noexam, noexam, noexam')))].fee_sedprice.sum()
    result2.loc[index_num+1, 'tot_outcasenum']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')].p_id.count()
    result2.loc[index_num+1, 'tot_outmonspan']=(SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')&(SedSub.mon_exe=='O')].mon_span.sum()/
                                               (SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')&(SedSub.mon_exe=='O')].mon_span.count()+0.00001)).round(2)
    result2.loc[index_num+1, 'tot_incasenum']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')].p_id.count()
    result2.loc[index_num+1, 'tot_inmonspan']=(SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')&(SedSub.mon_exe=='O')].mon_span.sum()/
                                               (SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')&(SedSub.mon_exe=='O')].mon_span.count()+0.00001)).round(2)
    result2.loc[index_num+1, 'tot_totcasenum']=SedSub[(SedSub.date_num==index_num+1)].p_id.count()
    result2.loc[index_num+1, 'tot_totmonspan']=(SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')].mon_span.sum()/
                                               (SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')].mon_span.count()+0.00001)).round(2)
result2['tot_outrev']=result2['MRI_outrev']+result2['CT_outrev']+result2['angio_outrev']+result2['bonemarrow_outrev']+result2['lumbar_outrev']+result2['musclebiopsy_outrev']+result2['else_outrev']
result2['tot_inrev']=result2['MRI_inrev']+result2['CT_inrev']+result2['angio_inrev']+result2['bonemarrow_inrev']+result2['lumbar_inrev']+result2['musclebiopsy_inrev']+result2['else_inrev']
result2['tot_totrev']=result2['MRI_totrev']+result2['CT_totrev']+result2['angio_totrev']+result2['bonemarrow_totrev']+result2['lumbar_totrev']+result2['musclebiopsy_totrev']+result2['else_totrev']


print(result2)






SedSub.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\SedSub.csv', index=True, encoding='utf-8-sig')
SedSub_d.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\SedSub_d.csv', index=True, encoding='utf-8-sig')
result_now.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\result_now.csv', index=True, encoding='utf-8-sig')
result1.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\result1.csv', index=True, encoding='utf-8-sig')
result2.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\result2.csv', index=True, encoding='utf-8-sig')
DrugSpan.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\DrugSpan.csv', index=True, encoding='utf-8-sig')




