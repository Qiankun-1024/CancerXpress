#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the CancerXpress survival risk model."""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cancerxpress import task_model, data_normalizer
from cancerxpress.losses import cox_loss
from cancerxpress.resources import ModelPaths

MODEL_PATHS = ModelPaths()
DEFAULT_DATA_PATH = PROJECT_ROOT / "dataset" / "expression_matrix.tsv"
DEFAULT_LABEL_PATH = PROJECT_ROOT / "dataset" / "survival_labels.tsv"


@tf.function
def printbar():
    """Print a timestamped progress bar."""
    ts = tf.timestamp()
    today_ts = ts % (24 * 60 * 60)
    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minute = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return tf.strings.format("0{}", m)
        return tf.strings.format("{}", m)

    timestring = tf.strings.join([timeformat(hour), timeformat(minute), timeformat(second)], separator=":")
    tf.print("==========" * 8, end="")
    tf.print(timestring)


def calculate_c_index(time, event, risk_score):
    """Compute the concordance index."""
    n = len(time)
    concordant = 0
    permissible = 0

    for i in range(n):
        if event[i] == 1:
            for j in range(n):
                if i != j and time[j] >= time[i]:
                    permissible += 1
                    if risk_score[i] > risk_score[j]:
                        concordant += 1
                    elif risk_score[i] == risk_score[j]:
                        concordant += 0.5

    return concordant / permissible if permissible > 0 else 0


@tf.function
def cox_loss_with_same_cancer_weight(
    labels,
    preds,
    cancer_idx,
    same_cancer_weight=0.35,
    cancer_group_weights=None
):
    """Weighted Cox loss to emphasize within-cancer ranking and hard cancer types."""
    global_loss = cox_loss(labels, preds)

    cancer_idx = tf.cast(tf.reshape(cancer_idx, [-1]), tf.int32)
    unique_cancers, _ = tf.unique(cancer_idx)

    if cancer_group_weights is None:
        cancer_group_weights = tf.ones((tf.reduce_max(cancer_idx) + 1,), dtype=tf.float32)
    else:
        cancer_group_weights = tf.cast(cancer_group_weights, tf.float32)

    def group_loss(cidx):
        mask = tf.equal(cancer_idx, cidx)
        g_labels = tf.boolean_mask(labels, mask)
        g_preds = tf.boolean_mask(preds, mask)

        n = tf.shape(g_labels)[0]
        event_n = tf.reduce_sum(g_labels[:, 1])
        valid = tf.logical_and(n >= 4, event_n >= 1.0)

        loss_val = tf.cond(
            valid,
            lambda: cox_loss(g_labels, g_preds),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )
        group_weight = tf.gather(cancer_group_weights, cidx)
        weighted_loss = loss_val * group_weight
        valid_float = tf.cast(valid, tf.float32)
        return weighted_loss, valid_float, group_weight * valid_float

    group_losses, group_valid, valid_group_weights = tf.map_fn(
        group_loss,
        unique_cancers,
        fn_output_signature=(tf.float32, tf.float32, tf.float32)
    )

    valid_weight_sum = tf.reduce_sum(valid_group_weights)
    within_loss = tf.cond(
        valid_weight_sum > 0,
        lambda: tf.reduce_sum(group_losses) / valid_weight_sum,
        lambda: global_loss
    )

    same_cancer_weight = tf.cast(same_cancer_weight, tf.float32)
    return (1.0 - same_cancer_weight) * global_loss + same_cancer_weight * within_loss


@tf.function
def improved_train_step(model, optimizer, features, labels):
    """训练步"""
    with tf.GradientTape() as tape:
        preds = model(features, training=True)
        loss = cox_loss(labels, preds)

        if hasattr(model, 'losses') and len(model.losses) > 0:
            reg_loss = tf.add_n(model.losses)
        else:
            reg_loss = 0.0
        total_loss = loss + reg_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    grad_var_pairs = [(g, v) for g, v in zip(gradients, model.trainable_variables) if g is not None]
    grad_var_pairs = [(tf.clip_by_value(g, -1.0, 1.0), v) for g, v in grad_var_pairs]
    optimizer.apply_gradients(grad_var_pairs)
    return loss


@tf.function
def improved_test_step(model, features, labels):
    """测试步"""
    preds = model(features, training=False)
    loss = cox_loss(labels, preds)
    return loss


class SurvivalModelWithCancerType(tf.keras.Model):
    """带癌症类型协变量的生存预测模型"""

    def __init__(self, encoder_layer, n_cancer_types, output_dim=1):
        super().__init__()
        self.encoder = encoder_layer

        # 癌症类型 embedding
        self.cancer_embedding = tf.keras.layers.Embedding(
            input_dim=n_cancer_types + 1,  # +1 for unknown
            output_dim=16,
            name='cancer_type_embedding'
        )

        # FCN layers - 输入维度需要加上 cancer embedding
        self.fc_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64),
            tf.keras.layers.LeakyReLU(alpha=0.1)
        ])
        self.out_layer = tf.keras.layers.Dense(output_dim, activation=None)

    def call(self, inputs, training=False):
        x, cancer_idx = inputs  # x: gene image, cancer_idx: cancer type index

        # Encoder 提取基因特征
        mean, logvar = tf.split(self.encoder(x, training=training), num_or_size_splits=2, axis=1)
        gene_features = mean  # 使用 mean 作为特征

        # 癌症类型 embedding
        cancer_feat = self.cancer_embedding(cancer_idx)  # [batch, 16]

        # 拼接特征
        combined = tf.concat([gene_features, cancer_feat], axis=1)

        # FCN
        logits = self.fc_layer(combined, training=training)
        output = self.out_layer(logits)
        return output


def train_improved_model(
    cancer_types=None,
    test_batches=("CPTAC",),
    holdout_cancer_types=None,
    batch_size=64,
    lr=0.000005,
    EPOCHS=200,
    same_cancer_loss_weight=0.35,
    cancer_difficulty_lambda=0.3,
    target_val_cindex=0.7
):
    """
    使用除测试集外的数据训练，癌症类型作为协变量。
    测试集支持：
    1) 指定一个或多个外部批次（test_batches）
    2) 指定完全不纳入训练的癌症类型（holdout_cancer_types）
    """
    if isinstance(test_batches, str):
        test_batches = [test_batches]
    else:
        test_batches = list(test_batches)
    if holdout_cancer_types is None:
        holdout_cancer_types = []
    else:
        holdout_cancer_types = list(holdout_cancer_types)

    if cancer_types is None:
        print(f"Training with all batches except holdout test sets, cancer type as covariate")
    else:
        print(f"Training for cancer types: {cancer_types}")
    print(f"test_batches: {test_batches}")
    print(f"holdout_cancer_types: {holdout_cancer_types}")
    print(f"same_cancer_loss_weight: {same_cancer_loss_weight}")
    print(f"cancer_difficulty_lambda: {cancer_difficulty_lambda}")
    print(f"target_val_cindex: {target_val_cindex}")

    # 加载所有数据
    data = pd.read_csv(DEFAULT_DATA_PATH, index_col=0, sep='\t')
    label = pd.read_csv(DEFAULT_LABEL_PATH, index_col=0, sep='\t')
    label = label[label['cancer_type'] != 'Normal']
    label.loc[label['cancer_type'] == 'LAML', 'cancer_type'] = 'AML'
    label = label[['OS_time_months', 'OS_Status', 'batch', 'cancer_type']]
    label = label.dropna(how='any')
    label = label[label['OS_time_months'] > 0]

    # 如果指定癌症类型，过滤数据
    if cancer_types:
        label = label[label['cancer_type'].isin(cancer_types)]

    # 过滤掉样本数太少的癌症类型（至少要有30个样本）
    cancer_counts = label['cancer_type'].value_counts()
    sufficient_cancers = cancer_counts[cancer_counts >= 30].index
    label = label[label['cancer_type'].isin(sufficient_cancers)]

    data = data.loc[label.index]

    # === 构建 cancer type 编码 ===
    cancer_types_list = sorted(label['cancer_type'].unique())
    cancer_type2idx = {ct: i for i, ct in enumerate(cancer_types_list)}
    n_cancer_types = len(cancer_types_list)
    print(f"Cancer types ({n_cancer_types}): {cancer_types_list}")

    # 添加 cancer_idx 列
    label['cancer_idx'] = label['cancer_type'].map(cancer_type2idx)

    # === 固定测试集（测试批次 + 排除癌种），其余数据做训练/验证 ===
    from sklearn.model_selection import train_test_split

    test_mask = label["batch"].isin(test_batches)
    if len(holdout_cancer_types) > 0:
        test_mask = test_mask | label["cancer_type"].isin(holdout_cancer_types)

    test_idx = label.index[test_mask].tolist()
    train_val_idx = label.index[~test_mask].tolist()

    if len(test_idx) == 0:
        raise ValueError(
            f"No samples found for test set (test_batches={test_batches}, holdout_cancer_types={holdout_cancer_types})"
        )
    if len(train_val_idx) == 0:
        raise ValueError(
            f"No train/val samples left after holdout (test_batches={test_batches}, holdout_cancer_types={holdout_cancer_types})"
        )

    # 提示测试集中训练未见癌种（保留这些样本用于更严格泛化评估）
    train_cancer_set = set(label.loc[train_val_idx, "cancer_type"].unique())
    test_cancer_set = set(label.loc[test_idx, "cancer_type"].unique())
    unseen_test_cancers = sorted(test_cancer_set - train_cancer_set)
    if len(unseen_test_cancers) > 0:
        print(f"Warning: test set contains unseen cancers in training: {unseen_test_cancers}")

    # 训练/验证按 癌种+事件 分层，必要时回退到癌种
    stratify_key_tv = (
        label.loc[train_val_idx, "cancer_type"].astype(str)
        + "__"
        + label.loc[train_val_idx, "OS_Status"].astype(int).astype(str)
    )
    if (stratify_key_tv.value_counts() < 2).any():
        print("Warning: Some train/val cancer+event strata have <2 samples; fallback to cancer_type stratification.")
        stratify_key_tv = label.loc[train_val_idx, "cancer_type"].astype(str)

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.15/0.85,
        random_state=42,
        stratify=stratify_key_tv,
    )

    print(f"\nDataset split (holdout batches={test_batches}, holdout cancers={holdout_cancer_types}):")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")

    # 各集的 batch 分布
    for split_name, split_idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        batch_dist = label.loc[split_idx, "batch"].value_counts()
        print(f"\n{split_name} batch distribution:")
        for b, c in batch_dist.items():
            print(f"  {b}: {c}")

    # === 预处理数据 ===
    def preprocess_data(indices):
        sub_data = data.loc[indices]
        sub_label = label.loc[indices]
        x = data_normalizer.gene2img(sub_data, dtype='TPM')
        y = sub_label[['OS_time_months', 'OS_Status']].to_numpy().astype(np.float32)
        cancer_idx = sub_label['cancer_idx'].to_numpy().astype(np.int32)
        return x, y, cancer_idx

    x_train, y_train, cancer_train = preprocess_data(train_idx)
    x_val, y_val, cancer_val = preprocess_data(val_idx)
    x_test, y_test, cancer_test = preprocess_data(test_idx)

    # === 构建模型 ===
    encoder_layer = task_model.get_encoder_layer(trainable_layer=3,
                                                  ckpt_path=str(MODEL_PATHS.batch_correction))
    model = SurvivalModelWithCancerType(
        encoder_layer=encoder_layer,
        n_cancer_types=n_cancer_types,
        output_dim=1
    )

    # 学习率调度
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        lr, decay_steps=100, decay_rate=0.96, staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # === 新的训练步函数（带 cancer_idx）===
    @tf.function
    def train_step(model, optimizer, features, cancer_idx, labels, cancer_group_weights):
        with tf.GradientTape() as tape:
            preds = model((features, cancer_idx), training=True)
            loss = cox_loss_with_same_cancer_weight(
                labels,
                preds,
                cancer_idx,
                same_cancer_weight=same_cancer_loss_weight,
                cancer_group_weights=cancer_group_weights
            )
            if hasattr(model, 'losses') and len(model.losses) > 0:
                reg_loss = tf.add_n(model.losses)
            else:
                reg_loss = 0.0
            total_loss = loss + reg_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        grad_var_pairs = [(g, v) for g, v in zip(gradients, model.trainable_variables) if g is not None]
        grad_var_pairs = [(tf.clip_by_value(g, -1.0, 1.0), v) for g, v in grad_var_pairs]
        optimizer.apply_gradients(grad_var_pairs)
        return loss

    # 数据集 - 包含 cancer_idx
    # 训练集采用按癌种平衡重采样，缓解高频癌种主导训练
    train_cancer_counts = pd.Series(cancer_train).value_counts().sort_index()
    print("\nTrain cancer distribution before balancing:")
    for cid, cnt in train_cancer_counts.items():
        cname = cancer_types_list[int(cid)]
        print(f"  {cname}: {cnt}")

    # inverse-sqrt频率，避免过度上采样极少类
    sample_weights = np.array([1.0 / np.sqrt(train_cancer_counts[c]) for c in cancer_train], dtype=np.float64)
    sample_weights = sample_weights / (sample_weights.sum() + 1e-12)
    base_group_weights = np.array(
        [1.0 / np.sqrt(train_cancer_counts.get(cid, 1.0)) for cid in range(n_cancer_types)],
        dtype=np.float32
    )
    base_group_weights = base_group_weights / (base_group_weights.mean() + 1e-12)
    dynamic_group_weights = base_group_weights.copy()
    print("\nInitial cancer group weights:")
    for cid, weight in enumerate(dynamic_group_weights):
        print(f"  {cancer_types_list[cid]}: {weight:.3f}")

    val_dataset = tf.data.Dataset.from_tensor_slices(
        ((x_val, cancer_val), y_val)
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        ((x_test, cancer_test), y_test)
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 设置检查点
    checkpoint_dir = os.path.join('training_checkpoints', 'improved_survival_all_batches')
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=10)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 训练循环
    patience = 0
    best_val_cindex = 0.0
    best_val_focus_macro = 0.0
    best_val_obj = -1.0
    best_epoch = 0

    # 关注癌种（用于组合早停目标）
    focus_cancers = ['UCEC', 'KIRC', 'LUAD', 'PAAD', 'HNSC', 'LUSC', 'GBM']
    val_label_eval = label.loc[val_idx]
    val_indices_map = {idx: i for i, idx in enumerate(val_idx)}

    def compute_val_focus_metrics(current_val_risk):
        current_focus = {}
        for cancer_type in focus_cancers:
            if cancer_type not in set(val_label_eval['cancer_type']):
                continue

            cancer_mask = val_label_eval['cancer_type'] == cancer_type
            cancer_indices = cancer_mask[cancer_mask].index.tolist()
            valid_indices = [val_indices_map[idx] for idx in cancer_indices if idx in val_indices_map]

            if len(valid_indices) > 10:
                cancer_times = y_val[valid_indices, 0]
                cancer_events = y_val[valid_indices, 1]
                cancer_risks = current_val_risk[valid_indices]
                cancer_cindex = calculate_c_index(cancer_times, cancer_events, cancer_risks)
                current_focus[cancer_type] = (cancer_cindex, len(valid_indices))

        if len(current_focus) > 0:
            macro = float(np.mean([v[0] for v in current_focus.values()]))
        else:
            macro = 0.0
        return current_focus, macro

    def compute_val_cindex_by_cancer(current_val_risk):
        current_metrics = {}
        for cancer_type in sorted(val_label_eval['cancer_type'].unique()):
            cancer_mask = val_label_eval['cancer_type'] == cancer_type
            cancer_indices = cancer_mask[cancer_mask].index.tolist()
            valid_indices = [val_indices_map[idx] for idx in cancer_indices if idx in val_indices_map]

            if len(valid_indices) > 10:
                cancer_times = y_val[valid_indices, 0]
                cancer_events = y_val[valid_indices, 1]
                cancer_risks = current_val_risk[valid_indices]
                cancer_cindex = calculate_c_index(cancer_times, cancer_events, cancer_risks)
                current_metrics[cancer_type] = (cancer_cindex, len(valid_indices))

        return current_metrics

    for epoch in range(EPOCHS):
        cancer_group_weights_tensor = tf.convert_to_tensor(dynamic_group_weights, dtype=tf.float32)

        # 训练阶段（按癌种平衡重采样）
        rng = np.random.default_rng(42 + epoch)
        sampled_indices = rng.choice(
            np.arange(len(cancer_train)),
            size=len(cancer_train),
            replace=True,
            p=sample_weights,
        )

        epoch_train_dataset = tf.data.Dataset.from_tensor_slices(
            ((x_train[sampled_indices], cancer_train[sampled_indices]), y_train[sampled_indices])
        ).shuffle(min(1000, len(sampled_indices)), seed=42 + epoch).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        train_losses = []
        for (batch_x, batch_cancer), batch_y in epoch_train_dataset:
            loss = train_step(model, optimizer, batch_x, batch_cancer, batch_y, cancer_group_weights_tensor)
            train_losses.append(loss)

        # 验证阶段
        val_risk_batches = []
        for (batch_x_val, batch_cancer_val), batch_y_val in tf.data.Dataset.from_tensor_slices(
            ((x_val, cancer_val), y_val)
        ).batch(32):
            batch_risk = model((batch_x_val, batch_cancer_val), training=False).numpy()
            val_risk_batches.extend(batch_risk.flatten())
        val_risk = np.array(val_risk_batches)
        val_cindex = calculate_c_index(y_val[:, 0], y_val[:, 1], val_risk)

        # 验证集损失
        val_losses = []
        for (batch_x, batch_cancer), batch_y in val_dataset:
            preds = model((batch_x, batch_cancer), training=False)
            val_loss = cox_loss_with_same_cancer_weight(
                batch_y,
                preds,
                batch_cancer,
                same_cancer_weight=same_cancer_loss_weight,
                cancer_group_weights=cancer_group_weights_tensor
            )
            val_losses.append(val_loss.numpy())
        avg_val_loss = np.mean(val_losses)

        val_cindex_focus_epoch, val_focus_macro = compute_val_focus_metrics(val_risk)
        val_cindex_by_cancer_epoch = compute_val_cindex_by_cancer(val_risk)
        val_obj = 0.6 * val_cindex + 0.4 * val_focus_macro

        updated_group_weights = base_group_weights.copy()
        for cid, cancer_type in enumerate(cancer_types_list):
            if cancer_type in val_cindex_by_cancer_epoch:
                cancer_val_cindex, _ = val_cindex_by_cancer_epoch[cancer_type]
                difficulty = max(0.0, target_val_cindex - cancer_val_cindex)
                updated_group_weights[cid] = base_group_weights[cid] * (1.0 + cancer_difficulty_lambda * difficulty)
        dynamic_group_weights = updated_group_weights / (updated_group_weights.mean() + 1e-12)

        printbar()
        print(f'EPOCH {epoch+1}/{EPOCHS}')
        print(f'Train loss: {np.mean(train_losses):.4f}, Val loss: {avg_val_loss:.4f}')
        print(f'Val C-index: {val_cindex:.4f}, Val Focus Macro C-index: {val_focus_macro:.4f}, Val Obj: {val_obj:.4f}')
        hardest_cancers = sorted(
            [(c, float(dynamic_group_weights[cancer_type2idx[c]])) for c in val_cindex_by_cancer_epoch.keys()],
            key=lambda x: -x[1]
        )[:5]
        print(f'Dynamic cancer weights (top 5): {[(c, round(w, 3)) for c, w in hardest_cancers]}')

        # 早停逻辑（组合目标）
        if val_obj > best_val_obj:
            best_val_obj = val_obj
            best_val_cindex = val_cindex
            best_val_focus_macro = val_focus_macro
            best_epoch = epoch
            patience = 0
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')
            print(f'Best Val Obj improved to: {best_val_obj:.4f} (overall={best_val_cindex:.4f}, focus_macro={best_val_focus_macro:.4f})')
        else:
            patience += 1

        if patience >= 15:
            print(f'Early stopping triggered at epoch {epoch+1}. Best epoch: {best_epoch+1}')
            break
    # 恢复最佳验证性能对应的checkpoint后再做最终评估
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print(f"Restored best checkpoint: {ckpt_manager.latest_checkpoint}")

    # 最终评估
    val_risk_batches = []
    for (batch_x_val, batch_cancer_val), batch_y_val in tf.data.Dataset.from_tensor_slices(
        ((x_val, cancer_val), y_val)
    ).batch(32):
        batch_risk = model((batch_x_val, batch_cancer_val), training=False).numpy()
        val_risk_batches.extend(batch_risk.flatten())
    val_risk = np.array(val_risk_batches)
    val_cindex_final = calculate_c_index(y_val[:, 0], y_val[:, 1], val_risk)

    train_risk_batches = []
    for (batch_x_train, batch_cancer_train), batch_y_train in tf.data.Dataset.from_tensor_slices(
        ((x_train, cancer_train), y_train)
    ).batch(32):
        batch_risk = model((batch_x_train, batch_cancer_train), training=False).numpy()
        train_risk_batches.extend(batch_risk.flatten())
    train_risk = np.array(train_risk_batches)
    train_cindex_final = calculate_c_index(y_train[:, 0], y_train[:, 1], train_risk)

    test_risk_batches = []
    for (batch_x_test, batch_cancer_test), batch_y_test in tf.data.Dataset.from_tensor_slices(
        ((x_test, cancer_test), y_test)
    ).batch(32):
        batch_risk = model((batch_x_test, batch_cancer_test), training=False).numpy()
        test_risk_batches.extend(batch_risk.flatten())
    test_risk = np.array(test_risk_batches)
    test_cindex = calculate_c_index(y_test[:, 0], y_test[:, 1], test_risk)
    # 验证集按关注癌症类型计算C-index
    val_cindex_focus, val_focus_macro_final = compute_val_focus_metrics(val_risk)

    # 按癌症类型计算测试集的c-index
    test_cindex_by_cancer = {}
    test_label = label.loc[test_idx]
    test_indices_map = {idx: i for i, idx in enumerate(test_idx)}

    for cancer_type in test_label['cancer_type'].unique():
        cancer_mask = test_label['cancer_type'] == cancer_type
        cancer_indices = cancer_mask[cancer_mask].index.tolist()

        valid_indices = [test_indices_map[idx] for idx in cancer_indices if idx in test_indices_map]

        if len(valid_indices) > 10:  # 至少10个样本才计算
            cancer_times = y_test[valid_indices, 0]
            cancer_events = y_test[valid_indices, 1]
            cancer_risks = test_risk[valid_indices]
            cancer_cindex = calculate_c_index(cancer_times, cancer_events, cancer_risks)
            test_cindex_by_cancer[cancer_type] = (cancer_cindex, len(valid_indices))

    # 按批次计算测试集的c-index
    test_cindex_by_batch = {}
    for batch in test_label['batch'].unique():
        batch_mask = test_label['batch'] == batch
        batch_indices = batch_mask[batch_mask].index.tolist()

        valid_indices = [test_indices_map[idx] for idx in batch_indices if idx in test_indices_map]

        if len(valid_indices) > 10:
            batch_times = y_test[valid_indices, 0]
            batch_events = y_test[valid_indices, 1]
            batch_risks = test_risk[valid_indices]
            batch_cindex = calculate_c_index(batch_times, batch_events, batch_risks)
            test_cindex_by_batch[batch] = (batch_cindex, len(valid_indices))

    train_val_gap = abs(train_cindex_final - val_cindex_final)
    train_test_gap = abs(train_cindex_final - test_cindex)

    print(f'\nFinal Results:')
    print(f'Best Val C-index: {best_val_cindex:.4f} at epoch {best_epoch+1}')
    print(f'Best Val Focus Macro C-index: {best_val_focus_macro:.4f}')
    print(f'Best Val Obj (0.6*overall + 0.4*focus_macro): {best_val_obj:.4f}')
    print(f'Train C-index: {train_cindex_final:.4f}')
    print(f'Val C-index: {val_cindex_final:.4f}')
    print(f'Val Focus Macro C-index: {val_focus_macro_final:.4f}')
    print(f'Test C-index: {test_cindex:.4f}')
    print(f'Train-Val Gap: {train_val_gap:.4f}')
    print(f'Train-Test Gap: {train_test_gap:.4f}')

    print(f'\nTest C-index by Cancer Type:')
    for cancer_type, (cindex, n) in sorted(test_cindex_by_cancer.items(), key=lambda x: -x[1][1]):
        print(f'  {cancer_type}: {cindex:.4f} (n={n})')

    print(f'\nVal C-index by Focus Cancer Type:')
    for cancer_type in focus_cancers:
        if cancer_type in val_cindex_focus:
            cindex, n = val_cindex_focus[cancer_type]
            print(f'  {cancer_type}: {cindex:.4f} (n={n})')
        else:
            print(f'  {cancer_type}: NA (n<=10 or absent)')

    print(f'\nTest C-index by Batch:')
    for batch, (cindex, n) in sorted(test_cindex_by_batch.items(), key=lambda x: -x[1][1]):
        print(f'  {batch}: {cindex:.4f} (n={n})')

    return model, test_cindex


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train the CancerXpress survival risk model")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Expression matrix TSV with samples in rows and genes in columns")
    parser.add_argument("--label", default=str(DEFAULT_LABEL_PATH), help="Survival label TSV with time, event, batch, and cancer-type columns")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.000005)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--test-batches", nargs="+", default=["META-PRISM", "MMRF", "CPTAC"], help="External cohorts to keep out of training")
    parser.add_argument("--holdout-cancer-types", nargs="*", default=["MM"], help="Cancer types to exclude from training and reserve for evaluation")
    parser.add_argument("--same-cancer-loss-weight", type=float, default=0.35)
    parser.add_argument("--cancer-difficulty-lambda", type=float, default=0.3)
    parser.add_argument("--target-val-cindex", type=float, default=0.7)
    parser.add_argument("--log-file", default=None, help="Optional log file path")
    args = parser.parse_args()

    globals()["DEFAULT_DATA_PATH"] = Path(args.data)
    globals()["DEFAULT_LABEL_PATH"] = Path(args.label)
    if args.log_file:
        sys.stdout = open(args.log_file, mode="w", encoding="utf-8")

    print("Training with multi-holdout external test sets, cancer type as covariate...")
    print("Starting improved risk prediction model training...")

    # 训练模型：META-PRISM + MMRF作为外部测试集，且MM癌种不纳入训练
    model, test_cindex = train_improved_model(
        cancer_types=None,
        test_batches=args.test_batches,
        holdout_cancer_types=args.holdout_cancer_types,
        batch_size=args.batch_size,
        lr=args.lr,
        EPOCHS=args.epochs,
        same_cancer_loss_weight=args.same_cancer_loss_weight,
        cancer_difficulty_lambda=args.cancer_difficulty_lambda,
        target_val_cindex=args.target_val_cindex
    )

    print(f"\nModel training completed with final test C-index: {test_cindex:.4f}")


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        gpu = gpus[0]
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices([gpu], "GPU")

    main()
