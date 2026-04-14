import copy
import torch
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from util.label import remap_prediction



def training(model, train_loader, criterion, optimizer, epoch, num_epochs, device):
    model.train()
    epoch_loss = 0.0

    for feat, true in train_loader:
        optimizer.zero_grad(set_to_none=True)
        feat, true = feat.to(device), true.to(device)

        out = model(feat)
        loss = criterion(out, true)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], '
        f'Loss: {epoch_loss / len(train_loader):.4f}, '
        f'LR: {optimizer.param_groups[0]['lr']:.6f}'
    )


@torch.inference_mode()
def validation(model, val_loader, epoch, device):
    model.eval()
    val_true = []
    val_pred = []

    for feat, true in val_loader:
        feat, true = feat.to(device), true.to(device)
        out = model(feat)

        _, pred = torch.max(out, 1)
        val_true.extend(true.cpu().numpy())
        val_pred.extend(pred.cpu().numpy())

    val_oa = accuracy_score(val_true, val_pred)
    val_mf1 = f1_score(val_true, val_pred, average='macro')
    val_wf1 = f1_score(val_true, val_pred, average='weighted')
    print(f'Validation Accuracy after Epoch {epoch + 1}: OA {val_oa:.4f}, mF1 {val_mf1:.4f}')
    return val_true, val_pred, val_oa, val_mf1, val_wf1


def trainer(model, criterion, optimizer, train_loader, val_loader, test_loader, num_epochs, device, crop_label):
    best_val_oa = 0.0
    best_val_mf1 = 0.0
    best_val_wf1 = 0.0
    best_model_state = None
    for epoch in range(num_epochs):
        training(model, train_loader, criterion, optimizer, epoch, num_epochs, device)
        _, _, val_oa, val_mf1, val_wf1 = validation(model, val_loader, epoch, device)

        if val_mf1 > best_val_mf1:
            best_val_oa = val_oa
            best_val_mf1 = val_mf1
            best_val_wf1 = val_wf1
            best_model_state = copy.deepcopy(model.state_dict())
    print(f'Val Accuracy (%): {best_val_oa * 100:.2f}, Val mF1 (%): {best_val_mf1 * 100:.2f}')

    model.load_state_dict(best_model_state)
    model.eval()

    test_oa, test_mf1, test_wf1, test_cm = {}, {}, {}, {}
    for target in test_loader.keys():
        test_true, test_pred, _, _, _ = validation(model, test_loader[target], epoch, device)
        test_pred = remap_prediction(test_pred, target, crop_label)

        test_oa[target] = accuracy_score(test_true, test_pred)
        test_mf1[target] = f1_score(test_true, test_pred, average='macro', labels=np.unique(test_true).tolist())
        test_wf1[target] = f1_score(test_true, test_pred, average='weighted')
        test_cm[target] = confusion_matrix(test_true, test_pred, labels=list(crop_label.values()))

        print(f'{target} Test Accuracy (%): {test_oa[target] * 100:.2f}, Test mF1 (%): {test_mf1[target] * 100:.2f}')
    return best_val_oa, best_val_mf1, best_val_wf1, test_oa, test_mf1, test_wf1, test_cm, best_model_state
