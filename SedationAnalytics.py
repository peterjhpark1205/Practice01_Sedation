###################################

# This is for scenarios regarding SNUCH's Sedation and MAC treatment

# Date : 2020.07.21
# Made by : Peter JH Park

###################################


import os, sys, csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import re
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.regressionplots import abline_plot
import scipy.stats as stats
import seaborn as sns
import pickle
import matplotlib.font_manager as fm



font_list = [font.name for font in fm.fontManager.ttflist]
print(font_list)



SedSub = pd.read_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\SedSub.csv', encoding = 'utf-8')
SedSub_d = pd.read_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\SedSub_d.csv', encoding = 'utf-8')
caseadjust = pd.read_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\RawData\\20200721_SedScene\\inout_caseadjust.csv', encoding = 'utf-8')



"""  LinearRegression1_monitoring execution : why it cannot gor straight up  (Monthly Basis)"""

result_mon = pd.DataFrame(columns=['date_num', 'date_m', 'c_day', 'monexe_num', 'moncase_num_out', 'moncase_num_in', 'monexe_rate', 'monspan_max', 'monspan_avg', 'monspan_med',
                                   'over30_span', 'over60_span', 'over90_span', 'over120_span', 'under30_span', 'under60_span', 'under90_span', 'under120_span'])


result_mon['date_num'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
result_mon['c_day'] = [21, 19, 23, 21, 19, 20, 21, 21, 20, 20, 22, 20, 19, 22]
result_mon['moncase_num'] = [964, 886, 1147, 997, 937, 902, 942, 960, 952, 809, 865, 898, 895, 1007]

result_mon.set_index('date_num', inplace=True)

print(result_mon)
################################################################################################

index_num = 0

#1905

for index_num in range(len(result_mon['c_day'])):
    result_mon.loc[index_num+1,'date_m']=SedSub[(SedSub.date_num==index_num+1)].date.unique()
    result_mon.loc[index_num+1,'monexe_num']=(SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')].p_id.count())
    result_mon.loc[index_num+1,'moncase_num_out']=(SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='out')].p_id.count())
    result_mon.loc[index_num+1,'moncase_num_in']=(SedSub[(SedSub.date_num==index_num+1)&(SedSub.inout=='in')].p_id.count())
    result_mon.loc[index_num+1,'monspan_max']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')].mon_span.max().round(0)
    result_mon.loc[index_num+1,'monspan_avg']=((SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')].mon_span.sum()
                                             /(SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')].mon_span.count()+0.00001))).round(0)
    result_mon.loc[index_num+1,'monspan_med']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')].mon_span.sum().round(0)
    result_mon.loc[index_num+1,'over30_span']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')&(SedSub.mon_span>30)].p_id.count()
    result_mon.loc[index_num+1,'over60_span']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')&(SedSub.mon_span>60)].p_id.count()
    result_mon.loc[index_num+1,'over90_span']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')&(SedSub.mon_span>90)].p_id.count()
    result_mon.loc[index_num+1,'over120_span']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')&(SedSub.mon_span>120)].p_id.count()
    result_mon.loc[index_num+1,'under30_span']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')&(SedSub.mon_span<=30)].p_id.count()
    result_mon.loc[index_num+1,'under60_span']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')&(SedSub.mon_span<=60)].p_id.count()
    result_mon.loc[index_num+1,'under90_span']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')&(SedSub.mon_span<=90)].p_id.count()
    result_mon.loc[index_num+1,'under120_span']=SedSub[(SedSub.date_num==index_num+1)&(SedSub.mon_exe=='O')&(SedSub.mon_span<=120)].p_id.count()

result_mon.moncase_num_out.fillna(0, inplace=True)
result_mon.moncase_num_in.fillna(0, inplace=True)

result_mon = result_mon.merge(caseadjust, on=['date_m'], how='left')
result_mon['moncase_num_out']=result_mon['moncase_num_out']*result_mon['out_adjust']
result_mon['moncase_num_in']=result_mon['moncase_num_in']*result_mon['in_adjust']
result_mon['moncase_num']=(result_mon['moncase_num_out']+result_mon['moncase_num_in']).round(0)

result_mon['monexe_rate'] = (result_mon['monexe_num']/result_mon['moncase_num']*100).round(2)

model1_mon = sm.OLS.from_formula(formula="monexe_num ~ moncase_num + over30_span + over60_span + over90_span + over120_span + under30_span + under60_span + under90_span + under120_span", data=result_mon)
output1_mon = model1_mon.fit()
print(output1_mon.summary())

model2_mon = sm.OLS.from_formula(formula="monexe_num ~ moncase_num + monspan_max + monspan_avg + monspan_med", data=result_mon)
output2_mon = model2_mon.fit()
print(output2_mon.summary())


model3_mon = sm.OLS.from_formula(formula="monexe_rate ~ moncase_num + over30_span + over60_span + over90_span + over120_span + under30_span + under60_span + under90_span + under120_span", data=result_mon)
output3_mon = model3_mon.fit()
print(output3_mon.summary())


model4_mon = sm.OLS.from_formula(formula="monexe_rate ~ moncase_num + monspan_max + monspan_avg + monspan_med", data=result_mon)
output4_mon = model4_mon.fit()
print(output4_mon.summary())



result_mon.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\result_mon.csv', index=True, encoding='utf-8-sig')





"""  LinearRegression2_monitoring execution : why it cannot gor straight up  (Daily Basis)"""

#df.where(df.Status == 'X').groupby('Month').Status.count()

SedSub_d_exspanmean = SedSub_d.where(SedSub_d.mon_exe=='O').groupby(['exams'], as_index=True).agg({'mon_span':pd.Series.mean})
SedSub_d_exspanmean = SedSub_d_exspanmean.rename(columns={'mon_span':'monexp_span'})
SedSub_d = SedSub_d.merge(SedSub_d_exspanmean, on=['exams'], how='left')
SedSub_d.loc[SedSub_d.mon_exe=='O', 'monexp_span'] = SedSub_d.mon_span
SedSub_d['monexp_span'].fillna(0, inplace=True)
SedSub_d = SedSub_d[~((SedSub_d.mon_exe=='X')&(SedSub_d.monexp_span==0))].reset_index()
SedSub_d_exe = SedSub_d[SedSub_d.mon_exe=='O']
totspan_avg = SedSub_d_exe.mon_span.mean()
totspan_med = SedSub_d_exe.mon_span.median()
print(totspan_avg)

monexe_num = SedSub_d.where(SedSub_d.mon_exe=='O').groupby(['date'], as_index=True).agg({'p_id': pd.Series.count})
moncase_num_in = SedSub_d.where(SedSub_d.inout=='in').groupby(['date'], as_index=False).agg({'p_id': pd.Series.count})
moncase_num_out = SedSub_d.where(SedSub_d.inout=='out').groupby(['date'], as_index=False).agg({'p_id': pd.Series.count})
monspan_max = SedSub_d.where(SedSub_d.mon_exe=='O').groupby(['date'], as_index=True).agg({'mon_span':pd.Series.max})
monspan_avg = SedSub_d.where(SedSub_d.mon_exe=='O').groupby(['date'], as_index=True).agg({'mon_span':pd.Series.mean})
monspan_med = SedSub_d.where(SedSub_d.mon_exe=='O').groupby(['date'], as_index=True).agg({'mon_span':pd.Series.median})
monspan_overavg = SedSub_d.where((SedSub_d.mon_exe=='O')&(SedSub_d.mon_span>SedSub_d_exe.mon_span.mean())).groupby(['date'], as_index=True).agg({'p_id': pd.Series.count})
monspan_underavg = SedSub_d.where((SedSub_d.mon_exe=='O')&(SedSub_d.mon_span<=SedSub_d_exe.mon_span.mean())).groupby(['date'], as_index=True).agg({'p_id': pd.Series.count})
monspan_overmed = SedSub_d.where((SedSub_d.mon_exe=='O')&(SedSub_d.mon_span>SedSub_d_exe.mon_span.median())).groupby(['date'], as_index=True).agg({'p_id': pd.Series.count})
monspan_undermed = SedSub_d.where((SedSub_d.mon_exe=='O')&(SedSub_d.mon_span<=SedSub_d_exe.mon_span.median())).groupby(['date'], as_index=True).agg({'p_id': pd.Series.count})
over30_span = SedSub_d.where((SedSub_d.mon_exe=='O')&(SedSub_d.mon_span>30)).groupby(['date'], as_index=True).agg({'p_id': pd.Series.count})
over60_span = SedSub_d.where((SedSub_d.mon_exe=='O')&(SedSub_d.mon_span>60)).groupby(['date'], as_index=True).agg({'p_id': pd.Series.count})
over90_span = SedSub_d.where((SedSub_d.mon_exe=='O')&(SedSub_d.mon_span>90)).groupby(['date'], as_index=True).agg({'p_id': pd.Series.count})
over120_span = SedSub_d.where((SedSub_d.mon_exe=='O')&(SedSub_d.mon_span>120)).groupby(['date'], as_index=True).agg({'p_id': pd.Series.count})
under30_span = SedSub_d.where((SedSub_d.mon_exe=='O')&(SedSub_d.mon_span<=30)).groupby(['date'], as_index=True).agg({'p_id': pd.Series.count})
under60_span = SedSub_d.where((SedSub_d.mon_exe=='O')&(SedSub_d.mon_span<=60)).groupby(['date'], as_index=True).agg({'p_id': pd.Series.count})
under90_span = SedSub_d.where((SedSub_d.mon_exe=='O')&(SedSub_d.mon_span<=90)).groupby(['date'], as_index=True).agg({'p_id': pd.Series.count})
under120_span = SedSub_d.where((SedSub_d.mon_exe=='O')&(SedSub_d.mon_span<=120)).groupby(['date'], as_index=True).agg({'p_id': pd.Series.count})


print(monexe_num)

monexe_num=monexe_num.rename(columns={'p_id' : 'monexe_num'})
moncase_num_in=moncase_num_in.rename(columns={'p_id' : 'moncase_num_in'})
moncase_num_out=moncase_num_out.rename(columns={'p_id' : 'moncase_num_out'})
monspan_max=monspan_max.rename(columns={'mon_span' : 'monspan_max'})
monspan_avg=monspan_avg.rename(columns={'mon_span' : 'monspan_avg'})
monspan_med=monspan_med.rename(columns={'mon_span' : 'monspan_med'})
monspan_overavg=monspan_overavg.rename(columns={'p_id' : 'monspan_overavg'})
monspan_underavg=monspan_underavg.rename(columns={'p_id' : 'monspan_underavg'})
monspan_overmed=monspan_overmed.rename(columns={'p_id' : 'monspan_overmed'})
monspan_undermed=monspan_undermed.rename(columns={'p_id' : 'monspan_undermed'})
over30_span=over30_span.rename(columns={'p_id' : 'over30_span'})
over60_span=over60_span.rename(columns={'p_id' : 'over60_span'})
over90_span=over90_span.rename(columns={'p_id' : 'over90_span'})
over120_span=over120_span.rename(columns={'p_id' : 'over120_span'})
under30_span=under30_span.rename(columns={'p_id' : 'under30_span'})
under60_span=under60_span.rename(columns={'p_id' : 'under60_span'})
under90_span=under90_span.rename(columns={'p_id' : 'under90_span'})
under120_span=under120_span.rename(columns={'p_id' : 'under120_span'})


result_mon_d = monexe_num.merge(moncase_num_out, on=['date'], how='left').merge(moncase_num_in, on=['date'], how='left')\
    .merge(monspan_max, on=['date'], how='left').merge(monspan_avg, on=['date'], how='left').merge(monspan_med, on=['date'], how='left')\
    .merge(monspan_overavg, on=['date'], how='left').merge(monspan_underavg, on=['date'], how='left').merge(monspan_overmed, on=['date'], how='left').merge(monspan_undermed, on=['date'], how='left')\
    .merge(over30_span, on=['date'], how='left').merge(over60_span, on=['date'], how='left').merge(over90_span, on=['date'], how='left').merge(over120_span, on=['date'], how='left')\
    .merge(under30_span, on=['date'], how='left').merge(under60_span, on=['date'], how='left').merge(under90_span, on=['date'], how='left').merge(under120_span, on=['date'], how='left')


result_mon_d.fillna(0, inplace=True)

result_mon_d['date_m']=result_mon_d['date']
result_mon_d['date_m']=result_mon_d['date_m'].astype(str)
result_mon_d['date_m']=result_mon_d['date_m'].str[:6]
result_mon_d['date_m']=result_mon_d['date_m'].astype(int)
result_mon_d=result_mon_d[result_mon_d['date_m']!=202007]

result_mon_d=result_mon_d.merge(caseadjust, on=['date_m'], how='left')
result_mon_d['moncase_num_out']=result_mon_d['moncase_num_out']*result_mon_d['out_adjust']
result_mon_d['moncase_num_in']=result_mon_d['moncase_num_in']*result_mon_d['in_adjust']
result_mon_d['moncase_num']=(result_mon_d['moncase_num_out']+result_mon_d['moncase_num_in']).round(0)

result_mon_d['monexe_rate'] = (result_mon_d['monexe_num']/result_mon_d['moncase_num']*100).round(2)

result_mon_d['monspan_totavg']=SedSub_d_exe.mon_span.mean()
result_mon_d['monspan_totmed']=SedSub_d_exe.mon_span.median()

result_mon_d = result_mon_d[['date', 'monexe_num', 'moncase_num', 'monexe_rate', 'monspan_max', 'monspan_avg', 'monspan_med',
                             'monspan_totavg', 'monspan_overavg', 'monspan_underavg', 'monspan_totmed', 'monspan_overmed', 'monspan_undermed',
                             'over30_span', 'over60_span', 'over90_span', 'over120_span', 'under30_span', 'under60_span', 'under90_span', 'under120_span']]
result_mon_d.reset_index(inplace=True)

##*****
model1_mon_d = sm.OLS.from_formula(formula="monexe_num ~ moncase_num", data=result_mon_d)
output1_mon_d = model1_mon_d.fit()
print(output1_mon_d.summary())

##*****
model2a_mon_d = sm.OLS.from_formula(formula="monexe_num ~ under120_span", data=result_mon_d) # under30_span + under90_span + under60_span +
output2a_mon_d = model2a_mon_d.fit()
print(output2a_mon_d.summary())

"""
model2b_mon_d = sm.OLS.from_formula(formula="monexe_num ~ over60_span + over120_span", data=result_mon_d) # over90_span + over30_span + 
output2b_mon_d = model2b_mon_d.fit()
print(output2b_mon_d.summary())


model3_mon_d = sm.OLS.from_formula(formula="monexe_num ~ monspan_avg", data=result_mon_d) #  monspan_med + monspan_max +
output3_mon_d = model3_mon_d.fit()
print(output3_mon_d.summary())


model4_mon_d = sm.OLS.from_formula(formula="monexe_rate ~ moncase_num", data=result_mon_d)
output4_mon_d = model4_mon_d.fit()
print(output4_mon_d.summary())


model5a_mon_d = sm.OLS.from_formula(formula="monexe_rate ~ under60_span", data=result_mon_d) # under30_span + under90_span + under120_span
output5a_mon_d = model5a_mon_d.fit()
print(output5a_mon_d.summary())


model5b_mon_d = sm.OLS.from_formula(formula="monexe_rate ~ over60_span", data=result_mon_d) # over90_span + over30_span + over120_span
output5b_mon_d = model5b_mon_d.fit()
print(output5b_mon_d.summary())


model6_mon_d = sm.OLS.from_formula(formula="monexe_rate ~ monspan_avg", data=result_mon_d) #  + monspan_med + monspan_max
output6_mon_d = model6_mon_d.fit()
print(output6_mon_d.summary())
"""

##*****
model7a_mon_d = sm.OLS.from_formula(formula="monexe_num ~ monspan_underavg", data=result_mon_d) # monspan_overavg +
output7a_mon_d = model7a_mon_d.fit()
print(output7a_mon_d.summary())

model7b_mon_d = sm.OLS.from_formula(formula="monexe_num ~ monspan_undermed", data=result_mon_d) # monspan_overmed +
output7b_mon_d = model7b_mon_d.fit()
print(output7b_mon_d.summary())

"""
model8a_mon_d = sm.OLS.from_formula(formula="monexe_rate ~ monspan_overavg", data=result_mon_d) #  + monspan_underavg
output8a_mon_d = model8a_mon_d.fit()
print(output8a_mon_d.summary())
"""

model8b_mon_d = sm.OLS.from_formula(formula="monexe_rate ~ monspan_undermed", data=result_mon_d) # monspan_overmed +
output8b_mon_d = model8b_mon_d.fit()
print(output8b_mon_d.summary())

'''
model9_mon_d = sm.OLS.from_formula(formula="moncase_num ~ monspan_underavg", data=result_mon_d) # monspan_overavg +
output9_mon_d = model9_mon_d.fit()
print(output9_mon_d.summary())

model10_mon_d = sm.OLS.from_formula(formula="moncase_num ~ monspan_overavg", data=result_mon_d) # monspan_overavg +
output10_mon_d = model10_mon_d.fit()
print(output10_mon_d.summary())
'''


#################################### Linear Regression Graphs
plt.rcParams['font.family']='Malgun Gothic'

sns.set(font='Malgun Gothic', rc={'axes.unicode_minus':False}, style='whitegrid', font_scale=1)



slope, intercept, r_value, p_value, std_err = stats.linregress(result_mon_d['moncase_num'], result_mon_d['monexe_num'])
slope1, intercept1, r_value1, p_value1, std_err1 = slope, intercept, pow(r_value, 2), p_value, std_err
annot_kws = {'prop': {'family':'Calibri', 'weight':'bold', 'size':20}}
plot1=sns.jointplot(x='moncase_num', y='monexe_num', data=result_mon_d, kind='reg', annot_kws=annot_kws,
                    line_kws={'color':'orangered'}, scatter_kws={'color':'darkgreen', 'edgecolor':'white'}, x_jitter=.9, y_jitter=.9)
plot1=plot1.set_axis_labels(xlabel='모니터링 대상건수', ylabel='모니터링 시행건수', fontsize=15, weight='bold')
showlinreg1, = plot1.ax_joint.plot([], [], linestyle="", alpha=0)
plot1.ax_joint.legend([showlinreg1], ['y={0:.3f}x+{1:.3f}, R$^2$: {2:.3f}, p_value: {3:.3f}'.format(slope1,intercept1,r_value1,p_value1)], **annot_kws)
#plot1=plot1.annotate(stats.pearsonr)


slope, intercept, r_value, p_value, std_err = stats.linregress(result_mon_d['monspan_underavg'], result_mon_d['monexe_num'])
slope2, intercept2, r_value2, p_value2, std_err2 = slope, intercept, pow(r_value, 2), p_value, std_err
plot2=sns.jointplot(x='monspan_underavg', y='monexe_num', data=result_mon_d, kind='reg', annot_kws=annot_kws,
                    line_kws={'color':'red'}, scatter_kws={'color':'darkblue', 'edgecolor':'white'}, x_jitter=.9, y_jitter=.9)
plot2=plot2.set_axis_labels(xlabel='<=평균 모니터링 시간', ylabel='모니터링 시행건수', fontsize=15, weight='bold')
showlinreg2, = plot2.ax_joint.plot([], [], linestyle="", alpha=0)
plot2.ax_joint.legend([showlinreg2], ['y={0:.3f}x+{1:.3f}, R$^2$: {2:.3f}, p_value: {3:.3f}'.format(slope2,intercept2,r_value2,p_value2)], **annot_kws)
#plot2=plot2.annotate(stats.pearsonr)


slope, intercept, r_value, p_value, std_err = stats.linregress(result_mon_d['monspan_underavg'], result_mon_d['moncase_num'])
slope3, intercept3, r_value3, p_value3, std_err3 = slope, intercept, pow(r_value, 2), p_value, std_err
plot3=sns.jointplot(x='monspan_underavg', y='moncase_num', data=result_mon_d, kind='reg', annot_kws=annot_kws,
                    line_kws={'color':'gold'}, scatter_kws={'color':'indigo', 'edgecolor':'white'}, x_jitter=.9, y_jitter=.9)
plot3=plot3.set_axis_labels(xlabel='<=평균 모니터링 시간', ylabel='모니터링 대상건수', fontsize=15, weight='bold')
showlinreg3, = plot3.ax_joint.plot([], [], linestyle="", alpha=0)
plot3.ax_joint.legend([showlinreg3], ['y={0:.3f}x+{1:.3f}, R$^2$: {2:.3f}, p_value: {3:.3f}'.format(slope3,intercept3,r_value3,p_value3)], **annot_kws)
#plot3=plot3.annotate(stats.pearsonr)

slope, intercept, r_value, p_value, std_err = stats.linregress(result_mon_d['under120_span'], result_mon_d['monexe_num'])
slope4, intercept4, r_value4, p_value4, std_err4 = slope, intercept, pow(r_value, 2), p_value, std_err
annot_kws = {'prop': {'family':'Calibri', 'weight':'bold', 'size':20}}
plot4=sns.jointplot(x='under120_span', y='monexe_num', data=result_mon_d, kind='reg', annot_kws=annot_kws,
                    line_kws={'color':'pink'}, scatter_kws={'color':'midnightblue', 'edgecolor':'white'}, x_jitter=.9, y_jitter=.9)
plot4=plot4.set_axis_labels(xlabel='<=120분', ylabel='모니터링 시행건수', fontsize=15, weight='bold')
showlinreg4, = plot4.ax_joint.plot([], [], linestyle="", alpha=0)
plot4.ax_joint.legend([showlinreg4], ['y={0:.3f}x+{1:.3f}, R$^2$: {2:.3f}, p_value: {3:.3f}'.format(slope4,intercept4,r_value4,p_value4)], **annot_kws)
#plot4=plot4.annotate(stats.pearsonr)

slope, intercept, r_value, p_value, std_err = stats.linregress(result_mon_d['monspan_undermed'], result_mon_d['monexe_num'])
slope5, intercept5, r_value5, p_value5, std_err5 = slope, intercept, pow(r_value, 2), p_value, std_err
plot5=sns.jointplot(x='monspan_underavg', y='monexe_num', data=result_mon_d, kind='reg', annot_kws=annot_kws,
                    line_kws={'color':'red'}, scatter_kws={'color':'darkblue', 'edgecolor':'white'}, x_jitter=.9, y_jitter=.9)
plot5=plot5.set_axis_labels(xlabel='<=전체 모니터링 시간 중간값', ylabel='모니터링 시행건수', fontsize=15, weight='bold')
showlinreg5, = plot5.ax_joint.plot([], [], linestyle="", alpha=0)
plot5.ax_joint.legend([showlinreg5], ['y={0:.3f}x+{1:.3f}, R$^2$: {2:.3f}, p_value: {3:.3f}'.format(slope5,intercept5,r_value5,p_value5)], **annot_kws)
#plot5=plot5.annotate(stats.pearsonr)



#################################### Line Graphs
fig1, ax1 = plt.subplots()

ax1.set_xlabel('시간순', color='black', weight='bold')
ax1.tick_params(axis='x', labelcolor='black')
ax1.set_ylabel('모니터링 시행건수', color='orange', weight='bold')
ax1.plot(result_mon_d.index, result_mon_d.monexe_num, color='orange')
ax1.tick_params(axis='y', labelcolor='black')
plt.grid(b=None)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('모니터링 대상건수', color='green', weight='bold')  # we already handled the x-label with ax1
ax2.plot(result_mon_d.index, result_mon_d.moncase_num, color='green')
ax2.tick_params(axis='y', labelcolor='black')
plt.grid(b=None)

fig1.tight_layout()  # otherwise the right y-label is slightly clipped


fig2, ax1 = plt.subplots()

ax1.set_xlabel('시간순', color='black', weight='bold')
ax1.tick_params(axis='x', labelcolor='black')
ax1.set_ylabel('모니터링 시행건수', color='red', weight='bold')
ax1.plot(result_mon_d.index, result_mon_d.monexe_num, color='red')
ax1.tick_params(axis='y', labelcolor='black')
plt.grid(b=None)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('<=평균 모니터링 시간', color='blue', weight='bold')  # we already handled the x-label with ax1
ax2.plot(result_mon_d.index, result_mon_d.monspan_underavg, color='blue')
ax2.tick_params(axis='y', labelcolor='black')
plt.grid(b=None)

fig2.tight_layout()  # otherwise the right y-label is slightly clipped


fig3, ax1 = plt.subplots()

ax1.set_xlabel('시간순', color='black', weight='bold')
ax1.tick_params(axis='x', labelcolor='black')
ax1.set_ylabel('모니터링 대상건수', color='gold', weight='bold')
ax1.plot(result_mon_d.index, result_mon_d.moncase_num, color='gold')
ax1.tick_params(axis='y', labelcolor='black')
plt.grid(b=None)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('<=평균 모니터링 시간', color='purple', weight='bold')  # we already handled the x-label with ax1
ax2.plot(result_mon_d.index, result_mon_d.monspan_underavg, color='purple')
ax2.tick_params(axis='y', labelcolor='black')
plt.grid(b=None)

fig3.tight_layout()  # otherwise the right y-label is slightly clipped


fig4, ax1 = plt.subplots()

ax1.set_xlabel('시간순', color='black', weight='bold')
ax1.tick_params(axis='x', labelcolor='black')
ax1.set_ylabel('모니터링 시행건수', color='red', weight='bold')
ax1.plot(result_mon_d.index, result_mon_d.monexe_num, color='red')
ax1.tick_params(axis='y', labelcolor='black')
plt.grid(b=None)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('<=120분', color='blue', weight='bold')  # we already handled the x-label with ax1
ax2.plot(result_mon_d.index, result_mon_d.under120_span, color='blue')
ax2.tick_params(axis='y', labelcolor='black')
plt.grid(b=None)

fig4.tight_layout()  # otherwise the right y-label is slightly clipped


fig5, ax1 = plt.subplots()

ax1.set_xlabel('시간순', color='black', weight='bold')
ax1.tick_params(axis='x', labelcolor='black')
ax1.set_ylabel('전체 평균 모니터링 시간', color='navy', weight='bold')
ax1.plot(result_mon_d.index, result_mon_d.monspan_totavg, color='navy')
ax1.tick_params(axis='y', labelcolor='black')
plt.grid(b=None)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('일별 평균 모니터링 시간', color='brown', weight='bold')  # we already handled the x-label with ax1
ax2.plot(result_mon_d.index, result_mon_d.monspan_avg, marker='o', color='brown')
ax2.tick_params(axis='y', labelcolor='black')
plt.grid(b=None)

fig5.tight_layout()  # otherwise the right y-label is slightly clipped


fig6, ax1 = plt.subplots()

ax1.set_xlabel('시간순', color='black', weight='bold')
ax1.tick_params(axis='x', labelcolor='black')
ax1.set_ylabel('모니터링 시행건수', color='red', weight='bold')
ax1.plot(result_mon_d.index, result_mon_d.monexe_num, color='red')
ax1.tick_params(axis='y', labelcolor='black')
plt.grid(b=None)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('<=전체 모니터링 시간 중간값', color='blue', weight='bold')  # we already handled the x-label with ax1
ax2.plot(result_mon_d.index, result_mon_d.monspan_undermed, color='blue')
ax2.tick_params(axis='y', labelcolor='black')
plt.grid(b=None)

fig6.tight_layout()  # otherwise the right y-label is slightly clipped


fig7, ax1 = plt.subplots()

ax1.set_xlabel('시간순', color='black', weight='bold')
ax1.tick_params(axis='x', labelcolor='black')
ax1.set_ylabel('전체 모니터링 시간 중간값', color='navy', weight='bold')
ax1.plot(result_mon_d.index, result_mon_d.monspan_totmed, color='navy')
ax1.tick_params(axis='y', labelcolor='black')
plt.grid(b=None)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('일별 평균 모니터링 시간', color='brown', weight='bold')  # we already handled the x-label with ax1
ax2.plot(result_mon_d.index, result_mon_d.monspan_avg, marker='o', color='brown')
ax2.tick_params(axis='y', labelcolor='black')
plt.grid(b=None)

fig7.tight_layout()  # otherwise the right y-label is slightly clipped


##**##
#plt.show()



result_mon_d.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\result_mon_d.csv', index=True, encoding='utf-8-sig')




spanrank = SedSub_d[SedSub_d['exams'].str.contains(', noexam, noexam, noexam, noexam')]
spanrank = spanrank[['exams', 'monexp_span']]
spanrank['exams']=spanrank['exams'].str.replace(', noexam, noexam, noexam, noexam', '')

#
spanrank120n=spanrank.where(spanrank.monexp_span>120).groupby(['exams']).agg({'monexp_span': pd.Series.count})
spanrank120r=spanrank.groupby(['exams']).agg({'monexp_span': pd.Series.count})
spanrank120r=spanrank120r.rename(columns={'monexp_span':'monexp_rate'})
spanrank120=spanrank120n.merge(spanrank120r, on='exams', how='left')
spanrank120['monexp_rate']=((spanrank120['monexp_span']/spanrank120['monexp_rate'])*100).round(2)

spanrank120_rate=spanrank120.sort_values('monexp_rate', ascending=False)
spanrank120_rate['monexp_rate']=spanrank120_rate['monexp_rate'].astype(str)+'%'
spanrank120_rate.reset_index(inplace=True); spanrank120_rate.reset_index(inplace=True)
spanrank120_rate['rank']=spanrank120_rate['index']+1
spanrank120_rate=spanrank120_rate.rename(columns={'rank':'순위', 'exams':'검사명', 'monexp_span':'120분시간초과_건수', 'monexp_rate':'120분시간초과_비율'})
spanrank120_rate=spanrank120_rate[['순위', '검사명', '120분시간초과_비율', '120분시간초과_건수']]

spanrank120_num=spanrank120.sort_values('monexp_span', ascending=False)
spanrank120_num['monexp_rate']=spanrank120_num['monexp_rate'].astype(str)+'%'
spanrank120_num.reset_index(inplace=True); spanrank120_num.reset_index(inplace=True)
spanrank120_num['rank']=spanrank120_num['index']+1
spanrank120_num=spanrank120_num.rename(columns={'rank':'순위', 'exams':'검사명', 'monexp_span':'120분시간초과_건수', 'monexp_rate':'120분시간초과_비율'})
spanrank120_num=spanrank120_num[['순위', '검사명', '120분시간초과_비율', '120분시간초과_건수']]

##
spanrank_avgn=spanrank.where(spanrank.monexp_span>totspan_avg).groupby(['exams']).agg({'monexp_span': pd.Series.count})
spanrank_avgr=spanrank.groupby(['exams']).agg({'monexp_span': pd.Series.count})
spanrank_avgr=spanrank_avgr.rename(columns={'monexp_span':'monexp_rate'})
spanrank_avg=spanrank_avgn.merge(spanrank_avgr, on='exams', how='left')
spanrank_avg['monexp_rate']=((spanrank_avg['monexp_span']/spanrank_avg['monexp_rate'])*100).round(2)

spanrank_avg_rate=spanrank_avg.sort_values('monexp_rate', ascending=False)
spanrank_avg_rate['monexp_rate']=spanrank_avg_rate['monexp_rate'].astype(str)+'%'
spanrank_avg_rate.reset_index(inplace=True); spanrank_avg_rate.reset_index(inplace=True)
spanrank_avg_rate['rank']=spanrank_avg_rate['index']+1
spanrank_avg_rate=spanrank_avg_rate.rename(columns={'rank':'순위', 'exams':'검사명', 'monexp_span':'평균시간초과_건수', 'monexp_rate':'평균시간초과_비율'})
spanrank_avg_rate=spanrank_avg_rate[['순위', '검사명', '평균시간초과_비율', '평균시간초과_건수']]

spanrank_avg_num=spanrank_avg.sort_values('monexp_span', ascending=False)
spanrank_avg_num['monexp_rate']=spanrank_avg_num['monexp_rate'].astype(str)+'%'
spanrank_avg_num.reset_index(inplace=True); spanrank_avg_num.reset_index(inplace=True)
spanrank_avg_num['rank']=spanrank_avg_num['index']+1
spanrank_avg_num=spanrank_avg_num.rename(columns={'rank':'순위', 'exams':'검사명', 'monexp_span':'평균시간초과_건수', 'monexp_rate':'평균시간초과_비율'})
spanrank_avg_num=spanrank_avg_num[['순위', '검사명', '평균시간초과_비율', '평균시간초과_건수']]

###
spanrank_medn=spanrank.where(spanrank.monexp_span>totspan_med).groupby(['exams']).agg({'monexp_span': pd.Series.count})
spanrank_medr=spanrank.groupby(['exams']).agg({'monexp_span': pd.Series.count})
spanrank_medr=spanrank_medr.rename(columns={'monexp_span':'monexp_rate'})
spanrank_med=spanrank_medn.merge(spanrank_medr, on='exams', how='left')
spanrank_med['monexp_rate']=((spanrank_med['monexp_span']/spanrank_med['monexp_rate'])*100).round(2)

spanrank_med_rate=spanrank_med.sort_values('monexp_rate', ascending=False)
spanrank_med_rate['monexp_rate']=spanrank_med_rate['monexp_rate'].astype(str)+'%'
spanrank_med_rate.reset_index(inplace=True); spanrank_med_rate.reset_index(inplace=True)
spanrank_med_rate['rank']=spanrank_med_rate['index']+1
spanrank_med_rate=spanrank_med_rate.rename(columns={'rank':'순위', 'exams':'검사명', 'monexp_span':'중간시간초과_건수', 'monexp_rate':'중간시간초과_비율'})
spanrank_med_rate=spanrank_med_rate[['순위', '검사명', '중간시간초과_비율', '중간시간초과_건수']]

spanrank_med_num=spanrank_med.sort_values('monexp_span', ascending=False)
spanrank_med_num['monexp_rate']=spanrank_med_num['monexp_rate'].astype(str)+'%'
spanrank_med_num.reset_index(inplace=True); spanrank_med_num.reset_index(inplace=True)
spanrank_med_num['rank']=spanrank_med_num['index']+1
spanrank_med_num=spanrank_med_num.rename(columns={'rank':'순위', 'exams':'검사명', 'monexp_span':'중간시간초과_건수', 'monexp_rate':'중간시간초과_비율'})
spanrank_med_num=spanrank_med_num[['순위', '검사명', '중간시간초과_비율', '중간시간초과_건수']]





spanrankreal = SedSub_d_exe[SedSub_d_exe['exams'].str.contains(', noexam, noexam, noexam, noexam')]
spanrankreal = spanrankreal[['exams', 'mon_span']]
spanrankreal['exams']=spanrankreal['exams'].str.replace(', noexam, noexam, noexam, noexam', '')

#
spanrankreal120n=spanrankreal.where(spanrankreal.mon_span>120).groupby(['exams']).agg({'mon_span': pd.Series.count})
spanrankreal120r=spanrankreal.groupby(['exams']).agg({'mon_span': pd.Series.count})
spanrankreal120r=spanrankreal120r.rename(columns={'mon_span':'mon_rate'})
spanrankreal120=spanrankreal120n.merge(spanrankreal120r, on='exams', how='left')
spanrankreal120['mon_rate']=((spanrankreal120['mon_span']/spanrankreal120['mon_rate'])*100).round(2)

spanrankreal120_rate=spanrankreal120.sort_values('mon_rate', ascending=False)
spanrankreal120_rate['mon_rate']=spanrankreal120_rate['mon_rate'].astype(str)+'%'
spanrankreal120_rate.reset_index(inplace=True); spanrankreal120_rate.reset_index(inplace=True)
spanrankreal120_rate['rank']=spanrankreal120_rate['index']+1
spanrankreal120_rate=spanrankreal120_rate.rename(columns={'rank':'순위', 'exams':'검사명', 'mon_span':'120분시간초과_건수', 'mon_rate':'120분시간초과_비율'})
spanrankreal120_rate=spanrankreal120_rate[['순위', '검사명', '120분시간초과_비율', '120분시간초과_건수']]

spanrankreal120_num=spanrankreal120.sort_values('mon_span', ascending=False)
spanrankreal120_num['mon_rate']=spanrankreal120_num['mon_rate'].astype(str)+'%'
spanrankreal120_num.reset_index(inplace=True); spanrankreal120_num.reset_index(inplace=True)
spanrankreal120_num['rank']=spanrankreal120_num['index']+1
spanrankreal120_num=spanrankreal120_num.rename(columns={'rank':'순위', 'exams':'검사명', 'mon_span':'120분시간초과_건수', 'mon_rate':'120분시간초과_비율'})
spanrankreal120_num=spanrankreal120_num[['순위', '검사명', '120분시간초과_비율', '120분시간초과_건수']]

##
spanrankreal_avgn=spanrankreal.where(spanrankreal.mon_span>totspan_avg).groupby(['exams']).agg({'mon_span': pd.Series.count})
spanrankreal_avgr=spanrankreal.groupby(['exams']).agg({'mon_span': pd.Series.count})
spanrankreal_avgr=spanrankreal_avgr.rename(columns={'mon_span':'mon_rate'})
spanrankreal_avg=spanrankreal_avgn.merge(spanrankreal_avgr, on='exams', how='left')
spanrankreal_avg['mon_rate']=((spanrankreal_avg['mon_span']/spanrankreal_avg['mon_rate'])*100).round(2)

spanrankreal_avg_rate=spanrankreal_avg.sort_values('mon_rate', ascending=False)
spanrankreal_avg_rate['mon_rate']=spanrankreal_avg_rate['mon_rate'].astype(str)+'%'
spanrankreal_avg_rate.reset_index(inplace=True); spanrankreal_avg_rate.reset_index(inplace=True)
spanrankreal_avg_rate['rank']=spanrankreal_avg_rate['index']+1
spanrankreal_avg_rate=spanrankreal_avg_rate.rename(columns={'rank':'순위', 'exams':'검사명', 'mon_span':'평균시간초과_건수', 'mon_rate':'평균시간초과_비율'})
spanrankreal_avg_rate=spanrankreal_avg_rate[['순위', '검사명', '평균시간초과_비율', '평균시간초과_건수']]

spanrankreal_avg_num=spanrankreal_avg.sort_values('mon_span', ascending=False)
spanrankreal_avg_num['mon_rate']=spanrankreal_avg_num['mon_rate'].astype(str)+'%'
spanrankreal_avg_num.reset_index(inplace=True); spanrankreal_avg_num.reset_index(inplace=True)
spanrankreal_avg_num['rank']=spanrankreal_avg_num['index']+1
spanrankreal_avg_num=spanrankreal_avg_num.rename(columns={'rank':'순위', 'exams':'검사명', 'mon_span':'평균시간초과_건수', 'mon_rate':'평균시간초과_비율'})
spanrankreal_avg_num=spanrankreal_avg_num[['순위', '검사명', '평균시간초과_비율', '평균시간초과_건수']]

###
spanrankreal_medn=spanrankreal.where(spanrankreal.mon_span>totspan_med).groupby(['exams']).agg({'mon_span': pd.Series.count})
spanrankreal_medr=spanrankreal.groupby(['exams']).agg({'mon_span': pd.Series.count})
spanrankreal_medr=spanrankreal_medr.rename(columns={'mon_span':'mon_rate'})
spanrankreal_med=spanrankreal_medn.merge(spanrankreal_medr, on='exams', how='left')
spanrankreal_med['mon_rate']=((spanrankreal_med['mon_span']/spanrankreal_med['mon_rate'])*100).round(2)

spanrankreal_med_rate=spanrankreal_med.sort_values('mon_rate', ascending=False)
spanrankreal_med_rate['mon_rate']=spanrankreal_med_rate['mon_rate'].astype(str)+'%'
spanrankreal_med_rate.reset_index(inplace=True); spanrankreal_med_rate.reset_index(inplace=True)
spanrankreal_med_rate['rank']=spanrankreal_med_rate['index']+1
spanrankreal_med_rate=spanrankreal_med_rate.rename(columns={'rank':'순위', 'exams':'검사명', 'mon_span':'중간시간초과_건수', 'mon_rate':'중간시간초과_비율'})
spanrankreal_med_rate=spanrankreal_med_rate[['순위', '검사명', '중간시간초과_비율', '중간시간초과_건수']]

spanrankreal_med_num=spanrankreal_med.sort_values('mon_span', ascending=False)
spanrankreal_med_num['mon_rate']=spanrankreal_med_num['mon_rate'].astype(str)+'%'
spanrankreal_med_num.reset_index(inplace=True); spanrankreal_med_num.reset_index(inplace=True)
spanrankreal_med_num['rank']=spanrankreal_med_num['index']+1
spanrankreal_med_num=spanrankreal_med_num.rename(columns={'rank':'순위', 'exams':'검사명', 'mon_span':'중간시간초과_건수', 'mon_rate':'중간시간초과_비율'})
spanrankreal_med_num=spanrankreal_med_num[['순위', '검사명', '중간시간초과_비율', '중간시간초과_건수']]





SedSub_d_exe_under120 = SedSub_d[SedSub_d.monexp_span<=120]
under120rate = ((SedSub_d_exe_under120['p_id'].count()/SedSub_d['p_id'].count())*100).round(2).astype(str) + '%'
print(under120rate)

SedSub_d_exe_underavg = SedSub_d[SedSub_d.monexp_span<=totspan_avg]
underavgrate = ((SedSub_d_exe_underavg['p_id'].count()/SedSub_d['p_id'].count())*100).round(2).astype(str) + '%'
print(underavgrate)

SedSub_d_exe_undermed = SedSub_d[SedSub_d.monexp_span<=totspan_med]
undermedrate = ((SedSub_d_exe_undermed['p_id'].count()/SedSub_d['p_id'].count())*100).round(2).astype(str) + '%'
print(undermedrate)


spanrank120_rate.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\spanrank120_rate.csv', index=True, encoding='utf-8-sig')
spanrank_avg_rate.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\spanrank_avg_rate.csv', index=True, encoding='utf-8-sig')
spanrank_med_rate.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\spanrank_med_rate.csv', index=True, encoding='utf-8-sig')
spanrank120_num.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\spanrank120_num.csv', index=True, encoding='utf-8-sig')
spanrank_avg_num.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\spanrank_avg_num.csv', index=True, encoding='utf-8-sig')
spanrank_med_num.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\spanrank_med_num.csv', index=True, encoding='utf-8-sig')
spanrankreal120_rate.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\spanrankreal120_rate.csv', index=True, encoding='utf-8-sig')
spanrankreal_avg_rate.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\spanrankreal_avg_rate.csv', index=True, encoding='utf-8-sig')
spanrankreal_med_rate.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\spanrankreal_med_rate.csv', index=True, encoding='utf-8-sig')
spanrankreal120_num.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\spanrankreal120_num.csv', index=True, encoding='utf-8-sig')
spanrankreal_avg_num.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\spanrankreal_avg_num.csv', index=True, encoding='utf-8-sig')
spanrankreal_med_num.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\spanrankreal_med_num.csv', index=True, encoding='utf-8-sig')


SedSub_d.to_csv('D:\\★☆DATA_ANALYSIS☆★\\WorkStuff\\Result\\20200721_SedScene\\SedSub_d_aud.csv', index=True, encoding='utf-8-sig')









