from tensorflow.lite.python.lite import TFLiteConverter
converter = TFLiteConverter.from_saved_model("deploy_model")
tflite_model = converter.convert()
with open('inception_resnet_v1.tflite', 'wb') as f:
    f.write(tflite_model)
