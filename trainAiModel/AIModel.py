import open3d as o3d
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

L2_WEIGHT_DECAY = 1e-5 # 增加一個全局的 L2 正則化因子

def conv_bn_relu(x, filters, kernel_size=1, strides=1, activation='relu', name_prefix='', use_l2=False):
    x = layers.Conv1D(filters, kernel_size=kernel_size, strides=strides, padding='valid',
                      kernel_initializer='glorot_uniform', kernel_regularizer=keras.regularizers.L2(L2_WEIGHT_DECAY) if use_l2 else None, name=f'{name_prefix}_conv')(x)
    x = layers.BatchNormalization(momentum=0.0, name=f'{name_prefix}_bn')(x) # PointNet 論文建議 momentum=0.0 或較小值
    if activation:
        x = layers.Activation(activation, name=f'{name_prefix}_act')(x)
    return x

# Define a named function for quaternion normalization
@keras.utils.register_keras_serializable()
def l2_normalize_quaternion(x):
    return tf.linalg.l2_normalize(x, axis=-1)

def dense_bn_relu(x, units, activation='relu', name_prefix='', use_l2=False):
    x = layers.Dense(units, kernel_initializer='glorot_uniform', kernel_regularizer=keras.regularizers.L2(L2_WEIGHT_DECAY) if use_l2 else None, name=f'{name_prefix}_dense')(x)
    x = layers.BatchNormalization(momentum=0.0, name=f'{name_prefix}_bn')(x)
    if activation:
        x = layers.Activation(activation, name=f'{name_prefix}_act')(x)
    return x

# T-Net (輸入對齊網路)
def tnet(inputs, num_features, name):
    # inputs shape: (batch, num_points, num_features)
    x = conv_bn_relu(inputs, 64, name_prefix=f'{name}_conv1', use_l2=True)
    x = conv_bn_relu(x, 128, name_prefix=f'{name}_conv2', use_l2=True)
    x = conv_bn_relu(x, 1024, name_prefix=f'{name}_conv3', use_l2=True)
    x = layers.GlobalMaxPooling1D(name=f'{name}_globalmaxpool')(x)
    x = dense_bn_relu(x, 512, name_prefix=f'{name}_fc1', use_l2=True)
    x = dense_bn_relu(x, 256, name_prefix=f'{name}_fc2', use_l2=True)

    # 初始化為單位矩陣的偏置
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    transform_matrix = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=keras.regularizers.L2(L2_WEIGHT_DECAY), # 添加正則化防止矩陣退化
        name=f'{name}_transform_matrix'
    )(x)
    transform_matrix = layers.Reshape((num_features, num_features), name=f'{name}_reshape')(transform_matrix)
    return transform_matrix

def create_pointnet_regression(num_points, num_output_features):
    point_cloud_input = keras.Input(shape=(num_points, 3), name="point_cloud_input")

    # 輸入 T-Net (K=3)
    input_transform = tnet(point_cloud_input, 3, name="input_tnet")
    x = layers.Dot(axes=(2, 1), name="input_transform_dot")([point_cloud_input, input_transform])

    # 點特徵提取
    x = conv_bn_relu(x, 64, name_prefix="feature_conv1", use_l2=True)
    x = conv_bn_relu(x, 64, name_prefix="feature_conv2", use_l2=True) # PointNet++ 中這裡的 channel 數可能不同

    # 特徵 T-Net (K=64)
    feature_transform_input = x
    feature_transform = tnet(feature_transform_input, 64, name="feature_tnet")
    x = layers.Dot(axes=(2, 1), name="feature_transform_dot")([feature_transform_input, feature_transform])

    # 更多特徵提取
    x = conv_bn_relu(x, 64, name_prefix="feature_conv3", use_l2=True)
    x = conv_bn_relu(x, 128, name_prefix="feature_conv4", use_l2=True)
    x = conv_bn_relu(x, 1024, name_prefix="feature_conv5_globalfeat", use_l2=True) # 全局特徵前的最後一層

    # 對稱函數：最大池化
    global_feature = layers.GlobalMaxPooling1D(name="global_max_pool")(x)

    # 全連接層進行回歸
    x = dense_bn_relu(global_feature, 512, name_prefix="reg_fc1", use_l2=True)
    x = layers.Dropout(0.3, name="reg_dropout1")(x)
    x = dense_bn_relu(x, 256, name_prefix="reg_fc2", use_l2=True)
    x = layers.Dropout(0.3, name="reg_dropout2")(x)

    # 輸出層，預測展平後的方向矩陣 (9個值)
    # 這裡的激活函數取決於你的矩陣元素範圍，如果沒有特殊約束，可以是線性 'linear'
    outputs = layers.Dense(num_output_features, activation='linear', name="output_matrix")(x)
    outputs = layers.Lambda(
          l2_normalize_quaternion, # 使用具名函數
          output_shape=(num_output_features,),
          name="output_normalization"
        )(outputs) # 歸一化四元數

    model = keras.Model(inputs=point_cloud_input, outputs=outputs, name="pointnet_regression")
    return model

@keras.utils.register_keras_serializable()
def quaternion_loss(y_true, y_pred):
    dot_product = tf.reduce_sum(y_true * y_pred, axis=-1)
    loss = 1.0 - tf.square(dot_product)
    return tf.reduce_mean(loss)

optimizer = keras.optimizers.Adam(learning_rate=0.001)