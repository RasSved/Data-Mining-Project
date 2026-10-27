import torch 
import numpy as np
from sklearn.metrics import confusion_matrix



def eval_model(model, loader, device):
    model.eval()
    with torch.no_grad():
        
        preds = []
        truth = []

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            predictions = logits.argmax(dim=1).cpu().numpy()
            preds.append(predictions)
            truth.append(y.cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(truth)

    return y_pred, y_true


def scorer(y_true, y_pred):
    matching = np.sum(y_true == y_pred)
    accuracy = matching / len(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    f1 = np.nan_to_num(f1)
    macro_f1 = f1.mean()

    return {"accuracy": accuracy, "macro_f1": macro_f1}
