import argparse
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cancerxpress import task_model, data_normalizer
from cancerxpress.losses import focal_loss
from cancerxpress.metrics import micro_average


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


def train_label_classifier(data, label, encoder_path, batch_size, lr, epochs, ckpt_dir, class_type='multiclass', trainable_layer='none'):
    data = data.loc[label.index]
    x = data_normalizer.gene2img(data, dtype='TPM')
    encoder = joblib.load(encoder_path)
    target_name = label.columns[0]
    y = encoder.transform(label[[target_name]]).astype('float32')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).shuffle(1000)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).shuffle(1000)
    activation = 'softmax' if class_type == 'multiclass' else 'sigmoid'
    classifier = task_model.make_finetune_model(y.shape[-1], activation=activation, trainable_layer=trainable_layer, pretrain='v2')
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=500, decay_rate=0.96, staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    train_micro = micro_average(name='train_micro_average')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    test_micro = micro_average(name='test_micro_average')
    ckpt = tf.train.Checkpoint(classifier=classifier)
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=10)
    os.makedirs(ckpt_dir, exist_ok=True)
    patience = 0
    min_test_loss = 1e8
    for epoch in range(epochs):
        for metric in [train_loss, train_accuracy, train_micro, test_loss, test_accuracy, test_micro]:
            metric.reset_states()
        for train_x, train_y in train_dataset:
            train_step(classifier, optimizer, train_x, train_y, train_loss, train_accuracy, train_micro)
        for test_x, test_y in test_dataset:
            test_step(classifier, test_x, test_y, test_loss, test_accuracy, test_micro)
        printbar()
        print(f'{target_name} classifier, EPOCH {epoch + 1}')
        print(f'Train loss: {train_loss.result()}, Test loss: {test_loss.result()}')
        print(f'Train accuracy: {train_accuracy.result()}, Test accuracy: {test_accuracy.result()}')
        print(f'Train F1: {train_micro.result()[2]}, Test F1: {test_micro.result()[2]}')
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


def train_step(model, optimizer, features, labels, train_loss, accuracy, micro_avg):
    with tf.GradientTape() as tape:
        preds = model(features)
        loss = focal_loss(labels, preds)
    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    train_loss(loss)
    accuracy.update_state(labels, preds)
    micro_avg.update_state(labels, preds)


def test_step(model, features, labels, test_loss, accuracy, micro_avg):
    preds = model(features)
    loss = focal_loss(labels, preds)
    test_loss(loss)
    accuracy.update_state(labels, preds)
    micro_avg.update_state(labels, preds)


def main():
    parser = argparse.ArgumentParser(description='Train a CancerXpress classifier for categorical sample labels')
    parser.add_argument('--data', required=True, help='Expression matrix TSV with samples in rows and genes in columns')
    parser.add_argument('--label', required=True, help='Single-column label table TSV')
    parser.add_argument('--encoder', required=True, help='Fitted one-hot encoder for the target label')
    parser.add_argument('--class-type', default='multiclass', choices=['multiclass', 'multilabel'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--ckpt-dir', required=True)
    parser.add_argument('--trainable-layer', default='none')
    args = parser.parse_args()
    data = pd.read_csv(args.data, index_col=0, sep='\t')
    label = pd.read_csv(args.label, index_col=0, sep='\t')
    common = data.index.intersection(label.index)
    train_label_classifier(data.loc[common], label.loc[common], args.encoder, args.batch_size, args.lr, args.epochs, args.ckpt_dir, args.class_type, args.trainable_layer)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_visible_devices([gpus[0]], 'GPU')
    main()
