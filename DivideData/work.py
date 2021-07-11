import xlwt
from progressbar import *

# THUCNews 新闻处理
def work(path, start, end, name):
    # 获取需要写入数据的行数
    index = end - start + 1  
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet(u"sheet1", cell_overwrite_ok=True)
    progress = ProgressBar()
    for i in progress(range(index)):
        file = open(name + str(start + i) +".txt","r",encoding='utf-8',errors='ignore')
        title = file.readline()
        content = ""
        while 1:
            line = file.readline().strip()
            if not line:
                break
            content += line
        file.close()
        value = [title , content]
        for j in range(2):
            worksheet.write(i, j, value[j])
    workbook.save(path)  # 保存工作簿



work("./星座.xls", 403001, 403900, "./星座/")


