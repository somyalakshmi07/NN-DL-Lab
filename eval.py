# eval.py
def evaluate(model, loader):
    model.eval()
    mae, mse = 0, 0
    with torch.no_grad():
        for img, gt in loader:
            img = img.cuda()
            pred = model(img)
            count_pred = pred.sum().item()
            count_gt = gt.sum().item()
            mae += abs(count_pred - count_gt)
            mse += (count_pred - count_gt) ** 2
    mae /= len(loader)
    mse = (mse / len(loader)) ** 0.5
    return mae, mse