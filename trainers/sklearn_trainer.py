import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from util.label import remap_prediction



def build_models(random_state: int):
    models = {
        'RF': RandomForestClassifier(
            n_estimators=500,
            n_jobs=-1,
            random_state=random_state,
            class_weight='balanced'
        ),
    }

    use_scaler = {
        'RF':  False,
    }
    return models, use_scaler


def classifier_test(model, targets, TG_datasets, crop_label):
    test_oa, test_mf1, test_wf1, test_cm = {}, {}, {}, {}
    for target in targets:
        test_fea, test_lbl = TG_datasets[target].tensors
        test_fea = test_fea.flatten(1).numpy()
        test_lbl = test_lbl.numpy()

        test_pred = model.predict(test_fea)
        test_pred = remap_prediction(test_pred, target, crop_label)

        test_oa[target] = accuracy_score(test_lbl, test_pred)
        test_mf1[target] = f1_score(test_lbl, test_pred, average='macro', labels=np.unique(test_lbl))
        test_wf1[target] = f1_score(test_lbl, test_pred, average='weighted')
        test_cm[target] = confusion_matrix(test_lbl, test_pred, labels=list(crop_label.values()))

        print(f'{target} Test Accuracy (%): {test_oa[target] * 100:.2f}, Test mF1 (%): {test_mf1[target] * 100:.2f}')
    return test_oa, test_mf1, test_wf1, test_cm


def classifier_trainer(args, train_fea, val_fea, train_lbl, val_lbl, seed):
    models, use_scaler = build_models(seed)
    cls_head = models[args.classifier]
    if use_scaler[args.classifier]:
        model = Pipeline([('scaler', StandardScaler()), ('clf', cls_head)])
    else:
        model = cls_head

    model.fit(train_fea, train_lbl)
    val_out = model.predict(val_fea)

    val_oa = accuracy_score(val_lbl, val_out)
    val_mf1 = f1_score(val_lbl, val_out, average='macro')
    val_wf1 = f1_score(val_lbl, val_out, average='weighted')
    print(f'Validation Accuracy: OA {val_oa:.4f}, mF1 {val_mf1:.4f}')

    return val_oa, val_mf1, val_wf1, model
