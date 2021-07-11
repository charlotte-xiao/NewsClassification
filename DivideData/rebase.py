import xlwt
import xlrd
from progressbar import *
import numpy as np

def rebase(inputpath, outputpath, name):
    # 获取需要写入数据的行数
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet(u"sheet1", cell_overwrite_ok=True)
    progress = ProgressBar()
    data = xlrd.open_workbook(inputpath) 
    sheet = data.sheet_by_index(0) 
    for i in progress(range(sheet.nrows)):
        [title , content] = sheet.row_values(i)
        value = [title, content, name]
        for j in range(3):
            worksheet.write(i, j, value[j])
    workbook.save(outputpath)  # 保存工作簿



rebase("./军事-158数据.xlsx", "./预处理/军事158.xls", "军事")


