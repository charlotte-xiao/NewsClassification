import random
import xlrd
import xlwt
import numpy as np

#主要目的：把download文件分割成训练集和测试集
data_path = "./Data/download.xls"
train_file = "./Data/train.xls"
test_file = "./Data/test.xls"

#定义一个函数把一个excel文件转换为一个字典，把value保存到一个excel文件中
def write_excel_xls_append(path, value):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet(u"sheet1", cell_overwrite_ok=True)  # 创建sheet
    for i in range(0, index):
        for j in range(0, len(value[i])):
            worksheet.write(i, j, value[i][j])
    workbook.save(path)  # 保存工作簿


data = xlrd.open_workbook(data_path)#打开文件获得数据
sheet = data.sheet_by_index(0)#读取第一个表
list = []
for i in range(sheet.nrows):
    list.append(i)#对所有行进行初始化，加到list中
test_rows = random.sample(list, int(sheet.nrows * 0.2))#0.3是一个参数，如果是1000个，800个是训练集，200个测试集合

for item in test_rows:
    list.remove(item)#把测试集的序列移出去
train_rows = list#得到一个训练集

mat_train = np.zeros([len(train_rows), sheet.ncols], np.str).tolist()#初始化value
mat_test = np.zeros([len(test_rows), sheet.ncols], np.str).tolist()#初始化value

num_test = 0#测试集的数据初始化为0
num_train = 0#训练集的数据初始化为0

# 去除标题行
for row in range(1, sheet.nrows):
    if row in test_rows:#如果行在测试序列中
        mat_test[num_test] = sheet.row_values(row)#放到测试集中
        num_test += 1
    else:#如果行在训练集中
        mat_train[num_train] = sheet.row_values(row)#放到训练集中
        num_train += 1

write_excel_xls_append(train_file, mat_train)
write_excel_xls_append(test_file, mat_test)
