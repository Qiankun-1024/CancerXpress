import tensorflow as tf
import tensorflow_addons as tfa


class Bio_Generator(tf.keras.Model):

    def __init__(self, input_shape, z_dim):
        super(Bio_Generator, self).__init__()
        self.z_dim = z_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_shape[1], input_shape[2], 1)),
                tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same'),
                tfa.layers.GroupNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same'),
                tfa.layers.GroupNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1000),
                tf.keras.layers.LeakyReLU(),
                # No activation
                tf.keras.layers.Dense((self.z_dim + self.z_dim)),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=(self.z_dim,)),
            tf.keras.layers.Dense(units=16*32*64),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape(target_shape=(16, 32, 64)),
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same'),
            tfa.layers.GroupNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same'),
            tfa.layers.GroupNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same'),
            tfa.layers.GroupNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=5, strides=1, padding='same'),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.z_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x, reparameterize=False):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        if reparameterize:
            z = self.reparameterize(mean, logvar)
            return z
        else:
            return mean, logvar
    

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=True):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def call(self, x, apply_sigmoid=True):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        z = self.reparameterize(mean, logvar)
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


class Noise_Encoder(tf.keras.Model):
    
    def __init__(self, input_shape, z_dim):
        super(Noise_Encoder, self).__init__()
        self.z_dim = z_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_shape[1], input_shape[2], 1)),
                tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2),
                tfa.layers.GroupNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=2),
                tfa.layers.GroupNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1000),
                tf.keras.layers.LeakyReLU(),
                # No activation
                tf.keras.layers.Dense(z_dim + z_dim)
            ]
        )
    
    @tf.function
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def call(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        z = self.reparameterize(mean, logvar)
        return z


class GroupNormalization(tf.keras.layers.Layer):

    def __init__(self, N=16, group=2, epsilon=1e-5):
        super(GroupNormalization, self).__init__()
        self.group = group
        self.epsilon = epsilon
        self.N = N

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=input_shape[-1:],
            initializer='ones',
            trainable=True)

        self.beta = self.add_weight(
            name='beta',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        _, C, H, W = x.shape
        x = tf.reshape(x, [self.N, self.group, C // self.group, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True) 
        x = (x - mean) / tf.sqrt(var + self.epsilon)
        x = tf.reshape(x, [self.N, C, H, W]) 
        return self.gamma * x + self.beta


class Reconstructor(tf.keras.Model):

    def __init__(self, input_shape, z_dim):
        super(Reconstructor, self).__init__()
        self.z_dim = z_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_shape[1], input_shape[2], 1)),
                tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same'),
                tfa.layers.GroupNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same'),
                tfa.layers.GroupNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1000),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(self.z_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=(self.z_dim * 2,)),
            tf.keras.layers.Dense(units=16*32*64),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape(target_shape=(16, 32, 64)),
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same'),
            tfa.layers.GroupNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same'),
            tfa.layers.GroupNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same'),
            tfa.layers.GroupNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=5, strides=1, padding='same'),
            ]
        )
        self.concat = tf.keras.layers.Concatenate()

    @tf.function
    def call(self, x, z, apply_sigmoid=False):
        x = self.encoder(x)
        x = self.concat([x,z])
        x = self.decoder(x)
        if apply_sigmoid:
            x = tf.sigmoid(x)
            return x
        return x


# def latent_batch_Discriminator(output_dim, activation='softmax'):
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.LeakyReLU())
#     model.add(tf.keras.layers.Dropout(0.3))
#     model.add(tf.keras.layers.Dense(output_dim, activation=activation))
#     return model


def latent_batch_Discriminator(output_dim, activation='softmax'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(384))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(output_dim, activation=activation))
    return model



def latent_bio_Discriminator(output_dim, activation='sigmoid'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(384))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(output_dim, activation=activation))
    return model


def recon_Discriminator(input_shape, output_dim, activation='sigmoid'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, 5, strides=2, padding='same', input_shape=[input_shape[1], input_shape[2], 1]))
    model.add(tfa.layers.GroupNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Conv2D(64, 5, strides=2, padding='same'))
    model.add(tfa.layers.GroupNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Conv2D(128, 5, strides=2, padding='same'))
    model.add(tfa.layers.GroupNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1000))
    model.add(tf.keras.layers.LeakyReLU(name='feature_matching_layer'))

    if activation=='sigmoid':
        model.add(tf.keras.layers.Dense(output_dim, activation='sigmoid'))
    elif activation=='softmax':
        model.add(tf.keras.layers.Dense(output_dim, activation='softmax'))
    return model


def unsupervised_Discriminator(Discriminator, label_type='multiclass', class_num=None):
    model = tf.keras.models.Sequential(Discriminator.layers[:-1])
    
    def predict(x):
        prediction = 1.0 - (1.0 / (tf.reduce_sum(tf.exp(x), axis=-1, keepdims=True) + 1.0))
        return prediction
    
    if label_type == 'multiclass':
        if class_num is not None:
            raise ValueError('class_num only supports for multilabel classification')

        model.add(tf.keras.layers.Lambda(predict))
    
    elif label_type == 'multilabel':
        
        def multilabel_predict(x):
            multilabel_pred = []
            start=0
            for n in class_num:
                end = start + n
                multilabel_pred.append(predict(x[:,start:end]))
                start = end
            return tf.concat(multilabel_pred, axis=1)

        model.add(tf.keras.layers.Lambda(multilabel_predict))

    return model


