import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import seaborn as sns
csv_path_h = r'D:\htfg_all\dice_list_h.csv'
csv_path_t1 = r'D:\htfg_all\dice_list_h_transfer_1.csv'

csv_file = csv.reader(open(csv_path_h,'r'))
csv_list = list(csv_file)

csv_file_t1 = csv.reader(open(csv_path_t1,'r'))
csv_list_t1 = list(csv_file_t1)
RN_list = []
SN_list = []
STN_list = []
RN_list_t1 = []
SN_list_t1 = []
STN_list_t1 = []
for j in range(1,37):
    RN = csv_list[j][1]
    RN_list.append(RN)
    SN = csv_list[j][2]
    SN_list.append(SN)
    STN = csv_list[j][3]
    STN_list.append(STN)

    RN_t1 = csv_list_t1[j][1]
    RN_list_t1.append(RN_t1)
    SN_t1 = csv_list_t1[j][2]
    SN_list_t1.append(SN_t1)
    STN_t1 = csv_list_t1[j][3]
    STN_list_t1.append(STN_t1)
#
print(RN_list)
print(SN_list)
# print(STN_list)

data = {
'RN': RN_list,
'RN_t': RN_list_t1,
'SN': SN_list,
'SN_t': SN_list_t1,
'STN': STN_list,
'STN_t': STN_list_t1,
}
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14}
df = pd.DataFrame(data).astype(float)
print(df)
f = df.boxplot(patch_artist=True, return_type='dict',sym='r*', positions=[1,1.55,3,3.55,5,5.55])
# f_t = df.boxplot(patch_artist=True, return_type='dict',sym='r*')
# print(f['boxes'])
color = ['g', 'deepskyblue', 'g', 'deepskyblue','g', 'deepskyblue']  # 有多少box就对应设置多少颜色
# color_t = ['deepskyblue', 'deepskyblue', 'deepskyblue', ]
for box, c in zip(f['boxes'], color):
    # 箱体边框颜色
    box.set(color=c, linewidth=2)
    # 箱体内部填充颜色
    box.set(facecolor='w')
plt.ylabel('DSC',font1)
plt.xlabel('Model2 and TransferFixNone', font1)
plt.grid(linestyle="--", alpha=0.3)
plt.show()

# df.plot.box()
# plt.ylabel('DSC',font1)
# plt.xlabel('TransferFixZeroUpLayer',font1)
# plt.grid(linestyle="--", alpha=0.3)
# plt.show()