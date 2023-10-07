import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Currently, we allow TensorFlow to use only one GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    print("GPU is available", gpus)
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
else:
  print("No GPU available", tf.__version__)