import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


'''cross entropy loss
def focal loss and cross entropy loss 
'''
def cross_entropy(labels, probs):
    #mask NA label
    zero = tf.constant(0, dtype=tf.float32)
    probs = tf.where(tf.math.is_nan(labels), zero, probs)
    labels = tf.where(tf.math.is_nan(labels), zero, labels)
    
    cross_loss = tf.add(tf.math.log(1e-10 + probs) * labels, tf.math.log(1e-10 + (1 - probs)) * (1 - labels))
    loss = tf.negative(tf.reduce_mean(tf.reduce_sum(cross_loss, axis=1)))
    return loss


def focal_loss(labels, probs, alpha=0.25, gamma=2):
    #mask NA label
    zero = tf.constant(0, dtype=tf.float32)
    probs = tf.where(tf.math.is_nan(labels), zero, probs)
    labels = tf.where(tf.math.is_nan(labels), zero, labels)
    
    cross_loss = tf.add(alpha * ((1- probs) ** gamma) * tf.math.log(1e-10 + probs) * labels, (1-alpha) * (probs ** gamma)* tf.math.log(1e-10 + (1 - probs)) * (1 - labels))
    loss = tf.negative(tf.reduce_mean(tf.reduce_sum(cross_loss, axis=1)))
    return loss


def missing_label_focal_loss(labels, probs, alpha=0.25, gamma=2):
    loss = 0.0
    i, cond = 0, tf.constant(1)
    probs = tf.cast(probs, tf.float32)
    while cond == 1:
        cond = tf.cond(i >= tf.shape(labels)[0] - 1, lambda: tf.constant(0), lambda: tf.constant(1))
        probs_, Y_ = tf.slice(probs, [i, 0], [1, tf.shape(labels)[1]]), tf.slice(labels, [i, 0],[1, tf.shape(labels)[1]])
        one = tf.constant(1, dtype=tf.float32)
        probs_, Y_ = tf.expand_dims(tf.gather_nd(probs_, tf.where(Y_ <= one)), 0), tf.expand_dims(tf.gather_nd(Y_, tf.where(Y_ <= one)), 0)
        cross_loss = tf.add(alpha * ((1- probs_) ** gamma) * tf.math.log(1e-10 + probs_) * Y_, (1-alpha) * (probs_ ** gamma)* tf.math.log(1e-10 + (1 - probs_)) * (1 - Y_))
        loss += tf.negative(tf.reduce_mean(tf.reduce_sum(cross_loss, axis=1)))
        i += 1
    return 0.1 * loss


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_mean(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


'''loss function'''
def conditional_VAE_loss(x, zr_mean, zr_logvar, zr, zp_mean, zp_logvar, zp, x_logit):
    logpx_bc = -tf.reduce_mean(tf.square(x - tf.math.sigmoid(x_logit)))
    logpc = log_normal_pdf(zr, 0., 0.)
    logqc_x = log_normal_pdf(zr, zr_mean, zr_logvar)
    logpb = log_normal_pdf(zp, 0., 0.)
    logqb_x = log_normal_pdf(zp, zp_mean, zp_logvar)
    return -tf.reduce_mean(logpx_bc + logpc - logqc_x + logpb - logqb_x)


def L2_loss(x, xr):
    loss = tf.reduce_mean(tf.square(x - xr))
    return loss


def L1_loss(x, xr):
    loss = tf.reduce_mean(tf.abs(x - xr))
    return loss


def feature_matching_loss(x_feature, xr_feature):
    loss = tf.reduce_mean(tf.square(x_feature - xr_feature))
    return loss


def latent_batch_discriminator_loss(batch_from_zp, batch_from_zr, batch):
    zp_loss = focal_loss(batch, batch_from_zp)
    zr_loss = focal_loss(batch, batch_from_zr)
    return tf.reduce_mean(zp_loss + zr_loss)


def latent_bio_discriminator_loss(bio_from_zp, bio_from_zr, biology):
    zp_loss = focal_loss(biology, bio_from_zp)
    zr_loss = focal_loss(biology, bio_from_zr)
    return tf.reduce_mean(zp_loss + zr_loss)


def zp_generator_loss(batch_from_zp, bio_from_zp, batch, biology):
    # zp_bio_loss = focal_loss(tf.zeros_like(biology), bio_from_zp)
    zp_bio_loss = focal_loss(0.5 * tf.ones_like(biology), bio_from_zp)
    zp_batch_loss = focal_loss(batch, batch_from_zp)
    return tf.reduce_mean(zp_bio_loss + zp_batch_loss)


def zr_generator_loss(batch_from_zr, bio_from_zr, batch, biology):
    # zr_batch_loss = focal_loss(tf.zeros_like(batch), batch_from_zr)
    zr_batch_loss = focal_loss(0.5 * tf.ones_like(batch), batch_from_zr)
    zr_bio_loss = focal_loss(biology, bio_from_zr)
    return tf.reduce_mean(zr_batch_loss + zr_bio_loss)


def observed_batch_discriminator_loss(x_output, xr_output, batch):
    x_loss = focal_loss(batch, x_output)
    xr_loss = focal_loss(batch, xr_output)
    return tf.reduce_mean(x_loss + xr_loss)


def x_generator_loss(batch_from_xr, xr_bio, biology):
    # batch_loss = cross_entropy(tf.zeros_like(batch_from_xr), batch_from_xr)
    batch_loss = focal_loss(0.5 * tf.ones_like(batch_from_xr), batch_from_xr)
    bio_loss = focal_loss(biology, xr_bio)
    return tf.reduce_mean(batch_loss + bio_loss)


def observed_bio_discriminator_loss(x_output, biology):    
    x_loss = focal_loss(biology, x_output)
    return x_loss


# def unsupervised_discriminator_loss(biology, pred, class_num=None):
#     sup_idx = []
    
#     if class_num is None:    
#         labeled = tf.reduce_all(tf.math.is_nan(biology), axis=1)
#         labeled = tf.where(labeled[:, tf.newaxis],
#                           tf.zeros([biology.shape[0], 1]),
#                           tf.ones([biology.shape[0], 1]))
#         sup_idx = labeled
#     else:
#         start = 0
#         for n in class_num:
#             end = start + n
#             labeled = tf.reduce_all(tf.math.is_nan(biology[:,start:end]), axis=1)
#             labeled = tf.where(labeled[:, tf.newaxis],
#                               tf.zeros([biology.shape[0], 1]),
#                               tf.ones([biology.shape[0], 1]))
#             sup_idx.append(labeled)
#             start = end
#         sup_idx = tf.concat(sup_idx, axis=1)     

#     loss = focal_loss(sup_idx, pred)
#     return loss


# def unsupervised_generator_loss(biology, pred, class_num=None):
#     unsup_idx = []
    
#     if class_num is None:    
#         labeled = tf.reduce_all(tf.math.is_nan(biology), axis=1)
#         labeled = tf.where(labeled[:, tf.newaxis],
#                           tf.ones([biology.shape[0], 1]),
#                           tf.zeros([biology.shape[0], 1]))
#         unsup_idx = labeled
#     else:
#         start = 0
#         for n in class_num:
#             end = start + n
#             labeled = tf.reduce_all(tf.math.is_nan(biology[:,start:end]), axis=1)
#             labeled = tf.where(labeled[:, tf.newaxis],
#                               tf.ones([biology.shape[0], 1]),
#                               tf.zeros([biology.shape[0], 1]))
#             unsup_idx.append(labeled)
#             start = end
#         unsup_idx = tf.concat(unsup_idx, axis=1)    

#     pred = tf.multiply(pred, unsup_idx)
#     loss = focal_loss(unsup_idx, pred)
#     return loss


def unsupervised_generator_loss(biology, pred, class_num=None):
    if class_num is None:
        # 单任务情况 - 原有逻辑
        unlabeled_mask = tf.reduce_all(tf.math.is_nan(biology), axis=1)
        unlabeled_mask = tf.cast(unlabeled_mask, tf.float32)
        
        # 只对无标签样本计算熵最小化损失
        entropy = -tf.reduce_sum(pred * tf.math.log(pred + 1e-8), axis=1)
        unsupervised_loss = tf.reduce_mean(entropy * unlabeled_mask)
        
        return unsupervised_loss
    else:
        # 多任务情况
        total_loss = 0
        start_idx = 0
        
        for i, n_classes in enumerate(class_num):
            end_idx = start_idx + n_classes
            
            # 提取当前任务的标签和预测
            task_biology = biology[:, start_idx:end_idx]
            task_pred = pred[:, start_idx:end_idx]
            
            # 检测当前任务的无标签样本
            task_unlabeled_mask = tf.reduce_all(tf.math.is_nan(task_biology), axis=1)
            task_unlabeled_mask = tf.cast(task_unlabeled_mask, tf.float32)
            
            # 计算当前任务的熵损失
            task_entropy = -tf.reduce_sum(task_pred * tf.math.log(task_pred + 1e-8), axis=1)
            task_loss = tf.reduce_mean(task_entropy * task_unlabeled_mask)
            
            total_loss += task_loss
            start_idx = end_idx
        
        # 返回所有任务损失的平均值
        return total_loss / len(class_num)
    


def unsupervised_generator_loss_with_balance(biology, pred, class_num=None, alpha=1.0):
    """
    添加平衡损失的无监督生成器损失
    """
    if class_num is None:
        # 单任务情况
        unlabeled_mask = tf.reduce_all(tf.math.is_nan(biology), axis=1)
        unlabeled_mask = tf.cast(unlabeled_mask, tf.float32)
        
        # 最小熵损失
        entropy = -tf.reduce_sum(pred * tf.math.log(pred + 1e-8), axis=1)
        min_entropy_loss = tf.reduce_mean(entropy * unlabeled_mask)
        
        # 平衡损失 - 防止模型坍塌
        balance_loss = 0.0
        if tf.reduce_sum(unlabeled_mask) > 0:
            unlabeled_pred = tf.boolean_mask(pred, tf.cast(unlabeled_mask, tf.bool))
            avg_pred = tf.reduce_mean(unlabeled_pred, axis=0)  # 平均预测分布
            target_dist = tf.ones_like(avg_pred) / tf.cast(tf.shape(avg_pred)[0], tf.float32)
            
            # 使用KL散度或MSE鼓励预测分布接近均匀分布
            balance_loss = tf.keras.losses.kld(target_dist, avg_pred)
            # 或者使用MSE: balance_loss = tf.reduce_mean(tf.square(avg_pred - target_dist))
        
        return min_entropy_loss + alpha * balance_loss
        
    else:
        # 多任务情况
        total_loss = 0
        start_idx = 0
        
        for i, n_classes in enumerate(class_num):
            end_idx = start_idx + n_classes
            
            # 提取当前任务的标签和预测
            task_biology = biology[:, start_idx:end_idx]
            task_pred = pred[:, start_idx:end_idx]
            
            # 检测当前任务的无标签样本
            task_unlabeled_mask = tf.reduce_all(tf.math.is_nan(task_biology), axis=1)
            task_unlabeled_mask = tf.cast(task_unlabeled_mask, tf.float32)
            
            # 最小熵损失
            task_entropy = -tf.reduce_sum(task_pred * tf.math.log(task_pred + 1e-8), axis=1)
            task_entropy_loss = tf.reduce_mean(task_entropy * task_unlabeled_mask)
            
            # 平衡损失（每个任务独立）
            task_balance_loss = 0.0
            if tf.reduce_sum(task_unlabeled_mask) > 0:
                task_unlabeled_pred = tf.boolean_mask(task_pred, tf.cast(task_unlabeled_mask, tf.bool))
                task_avg_pred = tf.reduce_mean(task_unlabeled_pred, axis=0)
                task_target_dist = tf.ones_like(task_avg_pred) / tf.cast(n_classes, tf.float32)
                
                task_balance_loss = tf.keras.losses.kld(task_target_dist, task_avg_pred)
            
            total_loss += task_entropy_loss + alpha * task_balance_loss
            start_idx = end_idx
        
        return total_loss / len(class_num)



def triplet_loss(y_true, embeddings, margin=1.0, alpha=2.0):
    # 获取样本的癌症类型和亚型标签
    cancer_type, subtype = y_true[:, 0], y_true[:, 1]
    
    # 随机选择三元组（Anchor, Positive, Negative）
    anchor_idx = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(embeddings)[0], dtype=tf.int32)
    anchor_cancer = cancer_type[anchor_idx]
    anchor_subtype = subtype[anchor_idx]
    
    # 正样本：同亚型
    positive_mask = (subtype == anchor_subtype)
    positive_idx = tf.random.shuffle(tf.where(positive_mask))[0]
    
    # 负样本：同癌症不同亚型（更高惩罚）或其他癌症
    negative_cancer_mask = (cancer_type == anchor_cancer) & (subtype != anchor_subtype)
    negative_other_mask = (cancer_type != anchor_cancer)
    negative_idx = tf.cond(
        tf.reduce_any(negative_cancer_mask),
        lambda: tf.random.shuffle(tf.where(negative_cancer_mask))[0],
        lambda: tf.random.shuffle(tf.where(negative_other_mask))[0]
    )
    
    # 计算距离
    anchor = embeddings[anchor_idx]
    positive = embeddings[positive_idx]
    negative = embeddings[negative_idx]
    
    pos_dist = tf.reduce_sum(tf.square(anchor - positive))
    neg_dist = tf.reduce_sum(tf.square(anchor - negative))
    
    # 动态调整margin：同癌症不同亚型的负样本施加更大margin
    loss = tf.maximum(pos_dist - neg_dist + (alpha if negative_cancer_mask else margin), 0.0)
    return loss


#Survival model loss function
def cox_loss(y_true, y_pred):
    """
    Optimized Cox Proportional Hazards Loss for TensorFlow.
    Supports mini-batch training, handles few events, and is numerically stable.

    Args:
        y_true: Tensor of shape [batch_size, 2]
                y_true[:,0] = time (months)
                y_true[:,1] = event (1=event, 0=censor)
        y_pred: Tensor of shape [batch_size, 1] risk score (can be negative)

    Returns:
        scalar loss
    """

    # squeeze risk_score to [batch]
    risk_score = tf.squeeze(y_pred, axis=-1)
    time = y_true[:, 0]
    event = y_true[:, 1]

    # --- Step 1: sort by descending time ---
    # ensures risk set is all samples with t >= t_i
    order = tf.argsort(time, direction='DESCENDING')
    sorted_risk = tf.gather(risk_score, order)
    sorted_event = tf.gather(event, order)

    # --- Step 2: numerical stability ---
    # shift risk so max=0 to avoid exp overflow
    sorted_risk = sorted_risk - tf.reduce_max(sorted_risk)

    # --- Step 3: compute cumulative sum of exp(risk) over risk set ---
    exp_risk = tf.exp(sorted_risk)
    cumsum_exp_risk = tf.cumsum(exp_risk, axis=0)  # descending order → forward cumsum

    # --- Step 4: compute partial likelihood ---
    loss_per_sample = sorted_risk - tf.math.log(cumsum_exp_risk + 1e-8)

    # --- Step 5: mask only event=1 samples ---
    mask = tf.cast(sorted_event, tf.bool)
    n_events = tf.reduce_sum(tf.cast(mask, tf.float32))

    # --- Step 6: compute normalized loss ---
    loss = -tf.reduce_sum(tf.boolean_mask(loss_per_sample, mask)) / (n_events + 1e-8)

    return loss


def ranking_loss(y_true, y_pred):
    time = y_true[:, 0]
    event = y_true[:, 1]
    risk = y_pred[:, 0]

    # 构造可比较对
    ti = tf.expand_dims(time, 1)
    tj = tf.expand_dims(time, 0)

    ei = tf.expand_dims(event, 1)

    comparable = tf.logical_and(ti < tj, ei > 0)

    ri = tf.expand_dims(risk, 1)
    rj = tf.expand_dims(risk, 0)

    loss = tf.nn.softplus(-(ri - rj))  # log(1 + exp(-(ri - rj)))

    loss = tf.boolean_mask(loss, comparable)

    return tf.reduce_mean(loss)