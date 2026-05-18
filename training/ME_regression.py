import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cancerxpress import task_model, data_normalizer


@tf.function
def printbar():
    ts = tf.timestamp()
    today_ts = ts % (24 * 60 * 60)
    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minute = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)
    def timeformat(m):
        return tf.strings.format('0{}', m) if tf.strings.length(tf.strings.format('{}', m)) == 1 else tf.strings.format('{}', m)
    timestring = tf.strings.join([timeformat(hour), timeformat(minute), timeformat(second)], separator=':')
    tf.print('==========' * 10, end='')
    print('=========' * 10, end='')
    tf.print(timestring)
    print(timestring)


def load_expression_and_me_targets(data_path: str, label_path: str):
    data = pd.read_csv(data_path, index_col=0, sep='\t')
    label = pd.read_csv(label_path, index_col=0)
    label = label.drop(columns=['MEgrey'], errors='ignore')
    common = data.index.intersection(label.index)
    return data.loc[common], label.loc[common]


def train_me_regressor(data, label, batch_size, lr, epochs, ckpt_dir, trainable_layer='none'):
    x = data_normalizer.gene2img(data, dtype='TPM')
    scaler = StandardScaler()
    y = scaler.fit_transform(label)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).shuffle(1000)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).shuffle(1000)
    classifier = task_model.make_finetune_model(y.shape[-1], activation=None, trainable_layer=trainable_layer, pretrain='v2')
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=500, decay_rate=0.96, staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    ckpt = tf.train.Checkpoint(classifier=classifier)
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=10)
    os.makedirs(ckpt_dir, exist_ok=True)
    patience = 0
    min_test_loss = 1e8
    for epoch in range(epochs):
        train_loss.reset_states()
        test_loss.reset_states()
        for train_x, train_y in train_dataset:
            train_step(classifier, optimizer, train_x, train_y, train_loss)
        for test_x, test_y in test_dataset:
            test_step(classifier, test_x, test_y, test_loss)
        printbar()
        print(f'Module Eigenpathway regressor, EPOCH {epoch + 1}')
        print(f'Train loss: {train_loss.result()}, Test loss: {test_loss.result()}')
        new_test_loss = test_loss.result()
        if new_test_loss < min_test_loss:
            min_test_loss = new_test_loss
            patience = 0
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')
        else:
            patience += 1
        if patience == 5:
            print('Test loss has not decreased during the last 5 epochs, early stopping')
            break


def train_step(model, optimizer, features, labels, train_loss):
    mse = tf.keras.losses.MeanSquaredError()
    with tf.GradientTape() as tape:
        preds = model(features)
        loss = mse(labels, preds)
    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    train_loss(loss)


def test_step(model, features, labels, test_loss):
    mse = tf.keras.losses.MeanSquaredError()
    preds = model(features)
    loss = mse(labels, preds)
    test_loss(loss)


def main():
    parser = argparse.ArgumentParser(description='Train the CancerXpress Module Eigenpathway regressor')
    parser.add_argument('--data', required=True, help='Expression matrix TSV with samples in rows and genes in columns')
    parser.add_argument('--label', required=True, help='Module Eigenpathway target table (CSV or TSV)')
    parser.add_argument('--sep', default='auto', choices=['auto', 'comma', 'tab'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--ckpt-dir', default=str(PROJECT_ROOT / 'training_checkpoints' / 'ME'))
    args = parser.parse_args()

    if args.sep == 'tab' or args.label.endswith('.tsv'):
        label = pd.read_csv(args.label, index_col=0, sep='\t')
        data = pd.read_csv(args.data, index_col=0, sep='\t')
        label = label.drop(columns=['MEgrey'], errors='ignore')
        common = data.index.intersection(label.index)
        data, label = data.loc[common], label.loc[common]
    else:
        data, label = load_expression_and_me_targets(args.data, args.label)
    train_me_regressor(data, label, args.batch_size, args.lr, args.epochs, args.ckpt_dir)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_visible_devices([gpus[0]], 'GPU')
    main()
