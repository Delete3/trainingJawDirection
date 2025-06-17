
import tensorflow as tf # 確保 tensorflow 被匯入
import tensorflow as tf
print(tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
print(gpus)