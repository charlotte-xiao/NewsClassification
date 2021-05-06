import random
import xlrd
import xlwt
import numpy as np


data_path = "./Test/download.xlsx"
train_file = "./Test/train.xls"
test_file = "./Test/test.xls"


def write_excel_xls_append(path, value):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet(u"sheet1", cell_overwrite_ok=True)  # 创建sheet
    for i in range(0, index):
        for j in range(0, len(value[i])):
            worksheet.write(i, j, value[i][j])
    workbook.save(path)  # 保存工作簿


data = xlrd.open_workbook(data_path)
sheet = data.sheet_by_index(0)
list = []
for i in range(sheet.nrows):
    list.append(i)
test_rows = random.sample(list, int(sheet.nrows * 0.1))

for item in test_rows:
    list.remove(item)
train_rows = list

mat_train = np.zeros([len(train_rows), sheet.ncols], np.str).tolist()
mat_test = np.zeros([len(test_rows), sheet.ncols], np.str).tolist()

num_test = 0
num_train = 0

# 去除标题行
for row in range(1, sheet.nrows):
    if row in test_rows:
        mat_test[num_test] = sheet.row_values(row)
        num_test += 1
    else:
        mat_train[num_train] = sheet.row_values(row)
        num_train += 1

write_excel_xls_append(train_file, mat_train)
write_excel_xls_append(test_file, mat_test)
