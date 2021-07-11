import xlwt
import xlrd
from progressbar import *


# 合并脚本
def merge(data_path1, data_path2, outoutPaht):
    # 获取需要写入数据的行数
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet(u"sheet1", cell_overwrite_ok=True)
    progress = ProgressBar()
    data1 = xlrd.open_workbook(data_path1) 
    sheet1 = data1.sheet_by_index(0) 
    data2 = xlrd.open_workbook(data_path2) 
    sheet2 = data2.sheet_by_index(0) 
    for i in progress(range(sheet1.nrows+sheet2.nrows)):
        if i < sheet1.nrows:
            value = sheet1.row_values(i)
            for j in range(2):
                worksheet.write(i, j, value[j])
        else:
            value = sheet2.row_values(i-sheet1.nrows)
            for j in range(2):
                worksheet.write(i, j, value[j])
    workbook.save(outoutPaht)  # 保存工作簿
