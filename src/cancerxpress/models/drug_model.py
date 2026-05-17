import tensorflow as tf
    
class GCNConv(tf.keras.layers.Layer):
    def __init__(self, units, activation='relu', use_bias=True, **kwargs):
        super(GCNConv, self).__init__(**kwargs)
        self.units = units              # 输出特征维度
        self.activation = tf.keras.activations.get(activation)  # 激活函数
        self.use_bias = use_bias        # 是否使用偏置项

    def build(self, input_shapes):
        """构建层权重"""
        input_dim = input_shapes[0][-1]

        # 添加权重矩阵（核）
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dim, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # 添加偏置项
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True
            )
        else:
            self.bias = None

        super(GCNConv, self).build(input_shapes)

    def call(self, inputs):
        """前向传播逻辑"""
        # 解包输入：节点特征和邻接矩阵
        features, adjacency = inputs

        # 特征线性变换 (XW)
        transformed = tf.matmul(features, self.kernel)
        
        # 邻接矩阵传播 (A·XW)
        if isinstance(adjacency, tf.SparseTensor):
            # 稀疏矩阵乘法
            propagated = tf.sparse.sparse_dense_matmul(adjacency, transformed)
        else:
            # 密集矩阵乘法（支持批量）
            propagated = tf.matmul(adjacency, transformed)

        # 添加偏置
        if self.use_bias:
            propagated = tf.nn.bias_add(propagated, self.bias)

        # 应用激活函数
        return self.activation(propagated)

    def get_config(self):
        """序列化配置（用于模型保存）"""
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias
        })
        return config



class GlobalMaxPooling(tf.keras.layers.Layer):
    def call(self, node_features):
        return tf.squeeze(tf.reduce_max(node_features, axis=1, keepdims=True), axis=1)


    

class GCNModel(tf.keras.Model):
    def __init__(self):
        super(GCNModel, self).__init__()
        
        self.gcn1 = GCNConv(256, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.gcn2 = GCNConv(256, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.gcn3 = GCNConv(256, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(0.2)
        self.gcn4 = GCNConv(100, activation='relu')
        self.dropout4 = tf.keras.layers.Dropout(0.2)
        self.pool = tf.keras.layers.GlobalMaxPooling1D()
    
    def call(self, inputs):
        features, adjacency = inputs
        
        x = self.gcn1([features, adjacency])
        x = self.dropout1(x)
        x = self.gcn2([x, adjacency])
        x = self.dropout2(x)
        x = self.gcn3([x, adjacency])
        x = self.dropout3(x)
        x = self.gcn4([x, adjacency])
        x = self.dropout4(x)
        output = self.pool(x)
        
        return output


class drug_FCN(tf.keras.Model):
    
    def __init__(self, activation):
        super(drug_FCN, self).__init__()
        self.fc_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.Dense(384),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128),
            tf.keras.layers.ELU(),             
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64),
            tf.keras.layers.LeakyReLU(alpha=0.1)
        ])
        self.concat = tf.keras.layers.Concatenate()
        if activation == 'sigmoid':
            self.out_layer = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(1, activation = 'sigmoid')
                ]
            )
        elif activation == 'softmax':
            self.out_layer = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(1, activation = 'softmax')
                ]
            )
        elif activation == None:
            self.out_layer = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(1, activation = None)
                ]
            )
            

    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def call(self, x_exp, x_drug):
        mean, logvar = tf.split(x_exp, num_or_size_splits=2, axis=1)
        # z = mean
        z = self.reparameterize(mean, logvar)
        z = self.concat([x_drug, z])
        z = self.fc_layer(z)
        logits = self.out_layer(z)
        return logits
