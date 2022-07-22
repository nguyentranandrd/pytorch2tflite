import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(
    "/content/drive/My Drive/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model")
tflite_model = converter.convert()

with open('model_tf.tflite', 'wb') as f:
    f.write(tflite_model)
