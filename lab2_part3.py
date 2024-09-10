## for ECE479 ICC Lab2 Part3

'''
*Main Student Script*
'''

# Your works start here

# Import packages you need here
from inception_resnet import InceptionResNetV1Norm
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_lfw_people
# import tensorflow_model_optimization as tfmot

# def Pruning():
#     model = InceptionResNetV1Norm()
#     pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
#                                                             final_sparsity=0.50,
#                                                             begin_step=0,
#                                                             end_step=1000)
#     prune_low_mag = tfmot.sparsity.keras.prune_low_magnitude
#     return prune_low_mag(model, pruning_schedule=pruning_schedule)



lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
# Create a model
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(lfw_people).batch(1).take(100):
    yield [tf.dtypes.cast(input_value, tf.float32)]

# # PRUNING OPTIMIZATIONS 
# # model = Pruning()
# # model.compile(optimizer='adam',
# #               loss='categorical_crossentropy',
# #               metrics=['accuracy'])
# # callbacks = [
# #     tfmot.sparsity.keras.UpdatePruningStep(),
# #     tfmot.sparsity.keras.PruningSummaries(log_dir='./pruning_logs'),
# # ]

# # converter = tf.lite.TFLiteConverter.from_keras_model(model)
# # converter.optimizations = [tf.lite.Optimize.DEFAULT]
# model = InceptionResNetV1Norm()

# # Verify the model and load the weights into the net
# print(model.summary())
# print(len(model.layers))
# model.load_weights("./weights/inception_keras_weights.h5")  # Has been translated from checkpoint

# model.save("quantization_model")
# from tensorflow.lite.python.lite import TFLiteConverter
# converter = TFLiteConverter.from_saved_model("quantization_model")
converter = tf.lite.TFLiteConverter.from_saved_model("quantization_model")
converter.representative_dataset = representative_data_gen
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# tflite_model_quant = converter.convert()
tflite_model = converter.convert()
with open('inception_resnet_v1.tflite', 'wb') as f:
    f.write(tflite_model)



