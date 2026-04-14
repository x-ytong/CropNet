import os
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from typing import cast
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset, random_split

from models.MLP import MLPClassifier
from data.loading import data_load
from data.augment import CropGlobeDataset
from trainers.torch_trainer import trainer
from trainers.sklearn_trainer import classifier_trainer, classifier_test
from util.label import CROP_INDEX
from util.seed import seed_everything
from util.writer import ensure_dir, write_txt, append_txt

sys.path.append(os.path.abspath("."))






def get_args():
    parser = argparse.ArgumentParser(description="Compare representations with RF/MLP.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--fea_type", type=str, required=True,
                        choices=["Harmonic", "Median", "Presto", "Presto_MayNov", "AlphaEarth"])
    parser.add_argument("--classifier", type=str, required=True, choices=["RF", "MLP"])

    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--targets", nargs="+", required=True)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--base_seed", type=int, default=81)

    return parser.parse_args()


def run(args):
    ensure_dir(args.output_dir)

    SR_dataset, TG_datasets, train_index, info_str = data_load(
        folder=args.data_dir,
        featype=args.fea_type,
        source=args.source,
        targets=args.targets,
        crop_index=CROP_INDEX,
    )

    output_file = f'{args.output_dir}/{args.source}_{args.fea_type}_{args.classifier}.txt'
    write_txt(info_str, output_file)

    if args.classifier == "RF":
        sr_fea, sr_lbl = SR_dataset.tensors
        sr_fea = sr_fea.flatten(1).numpy()
        sr_lbl = sr_lbl.numpy()
    elif args.classifier == "MLP":
        x, y = SR_dataset.tensors
        x = x.flatten(1)
        sr_dataset = TensorDataset(x, y)
        tg_datasets = {}
        for target in args.targets:
            x, y = TG_datasets[target].tensors
            x = x.flatten(1)
            tg_datasets[target] = TensorDataset(x, y)
    else:
        raise ValueError(f"Unsupported classifier: {args.classifier}.")

    VAL_OA, VAL_mF1, VAL_wF1 = [], [], []
    TEST_OA, TEST_mF1, TEST_wF1, TEST_CM = {}, {}, {}, {}
    for target in args.targets:
        TEST_OA[target], TEST_mF1[target], TEST_wF1[target] = [], [], []
        TEST_CM[target] = np.zeros((len(train_index), len(train_index)))

    for run_id in range(1, args.num_runs + 1):
        seed = args.base_seed + run_id
        generator = seed_everything(seed)

        if args.classifier == "RF":
            train_fea, val_fea, train_lbl, val_lbl = train_test_split(
                sr_fea, sr_lbl,
                test_size=args.val_ratio,
                stratify=sr_lbl,
                random_state=seed,
            )

            val_oa, val_mf1, val_wf1, model = classifier_trainer(args, train_fea, val_fea, train_lbl, val_lbl, seed)
            test_oa, test_mf1, test_wf1, test_cm = classifier_test(model, args.targets, TG_datasets, train_index)

        elif args.classifier == "MLP":
            x, _ = sr_dataset[0]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = MLPClassifier(in_dim=x.shape[0], num_classes=len(train_index)).to(device)

            val_size = int(args.val_ratio * len(sr_dataset))
            train_size = len(sr_dataset) - val_size
            train, val = random_split(sr_dataset, [train_size, val_size], generator=generator)

            train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False)

            test_loader = {}
            for target in args.targets:
                test_loader[target] = DataLoader(tg_datasets[target], batch_size=args.batch_size, shuffle=False)

            train_labels = sr_dataset.tensors[1].numpy()
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(train_labels),
                y=np.array(train_labels)
            )
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            val_oa, val_mf1, val_wf1, test_oa, test_mf1, test_wf1, test_cm, _ = (
                trainer(model, criterion, optimizer, train_loader, val_loader, test_loader,
                        args.num_epochs, device, train_index))

        VAL_OA.append(val_oa)
        VAL_mF1.append(val_mf1)
        VAL_wF1.append(val_wf1)
        for target in args.targets:
            TEST_OA[target].append(test_oa[target])
            TEST_mF1[target].append(test_mf1[target])
            TEST_wF1[target].append(test_wf1[target])
            TEST_CM[target] = TEST_CM[target] + test_cm[target]

        print(f'Round {run_id} completed.')

    print(f'{args.source}→{args.targets}.')
    print(f'Val OA (%): {np.mean(np.array(VAL_OA)) * 100:.2f} ± {np.std(np.array(VAL_OA)) * 100:.2f}, '
          f'Val mF1 (%): {np.mean(np.array(VAL_mF1)) * 100:.2f} ± {np.std(np.array(VAL_mF1)) * 100:.2f}, '
          f'Val wF1 (%): {np.mean(np.array(VAL_wF1)) * 100:.2f} ± {np.std(np.array(VAL_wF1)) * 100:.2f}.')
    for target in args.targets:
        print(
            f'Test {target} OA (%): {np.mean(np.array(TEST_OA[target])) * 100:.2f} ± {np.std(np.array(TEST_OA[target])) * 100:.2f}, '
            f'Test {target} mF1 (%): {np.mean(np.array(TEST_mF1[target])) * 100:.2f} ± {np.std(np.array(TEST_mF1[target])) * 100:.2f}, '
            f'Test {target} wF1 (%): {np.mean(np.array(TEST_wF1[target])) * 100:.2f} ± {np.std(np.array(TEST_wF1[target])) * 100:.2f}.')

    result_str = '\n' + '-' * 100 + '\n' + f'{args.source}→{args.targets}.'
    result_str += '\n' + (
        f'Val OA (%): {np.mean(np.array(VAL_OA)) * 100:.2f} ± {np.std(np.array(VAL_OA)) * 100:.2f}, '
        f'Val mF1 (%): {np.mean(np.array(VAL_mF1)) * 100:.2f} ± {np.std(np.array(VAL_mF1)) * 100:.2f}, '
        f'Val wF1 (%): {np.mean(np.array(VAL_wF1)) * 100:.2f} ± {np.std(np.array(VAL_wF1)) * 100:.2f}.'
    )
    for target in args.targets:
        result_str += '\n' + (
            f'Test {target} OA (%): {np.mean(np.array(TEST_OA[target])) * 100:.2f} ± {np.std(np.array(TEST_OA[target])) * 100:.2f}, '
            f'Test {target} mF1 (%): {np.mean(np.array(TEST_mF1[target])) * 100:.2f} ± {np.std(np.array(TEST_mF1[target])) * 100:.2f}, '
            f'Test {target} wF1 (%): {np.mean(np.array(TEST_wF1[target])) * 100:.2f} ± {np.std(np.array(TEST_wF1[target])) * 100:.2f}.'
        )
        result_str += '\n' + np.array2string(TEST_CM[target], separator=', ')
        result_str += '\n' + '-' * 100

    append_txt(result_str, output_file)


def main():
    args = get_args()
    run(args)


if __name__ == "__main__":
    main()