import torch
import xlwt


def evaluate_single(config, model, data):  # 单条新闻预测函数
    model.eval()
    with torch.no_grad():
        texts = (
            torch.LongTensor([data[0]]).to(config.device),
            torch.LongTensor([data[1]]).to(config.device)
            )
        outputs = torch.max(model(texts).data, 1)[1].cpu().numpy()
        predic = config.class_list[outputs.tolist()[0]]
        # print("预测值" + predic)
        return predic


def evaluate_multi(config, model, predic_data, savePath):  # 多条新闻预测函数
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet(u"类别", cell_overwrite_ok=True)  # 创建sheet
    len = 4
    sheetHead = ["编号", "channelName", "title", "content"]
    for i in range(len):
        worksheet.write(0, i, sheetHead[i])
    model.eval()
    with torch.no_grad():
        for datas in predic_data:
            texts = (
                torch.LongTensor([datas[0]]).to(config.device),
                torch.LongTensor([datas[1]]).to(config.device)
                )
            index, title, content = datas[2]
            outputs = torch.max(model(texts).data, 1)[1].cpu().numpy()
            predic = config.class_list[outputs.tolist()[0]]
            # print("预测值" + predic)
            temp = [index, predic, title, content]
            for i in range(len):
                worksheet.write(int(index), i, temp[i])
    workbook.save(savePath)
