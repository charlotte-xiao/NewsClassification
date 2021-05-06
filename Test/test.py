# import xlrd
# import csv
# import codecs


# def xlsx_to_csv(xlsxFilename, csvFilename):
#     workbook = xlrd.open_workbook(xlsxFilename)

#     table = workbook.sheet_by_index(0)
#     for row_num in range(table.nrows):
#         if row_num == 0:
#             continue
#         # 内容,标签,标题
#         content, label, title = table.row_values(row_num)

# workbook = xlrd.open_workbook(xlsxFilename)
# table = workbook.sheet_by_index(0)
# with codecs.open(csvFilename, "w", encoding="utf-8") as f:
#     write = csv.writer(f)
#     print(table.nrows)
#     for row_num in range(table.nrows):
#         if(row_num == 0):
#             continue
#         row_value = table.row_values(row_num)
#         write.writerow(row_value)


# if __name__ == "__main__":
#     xlsx_to_csv("download.xlsx", "download.csv")
import numpy as np
mat_train = np.zeros([5, 3], dtype = np.str)
print(mat_train)