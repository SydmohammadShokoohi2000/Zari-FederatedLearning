import os
import numpy as np
import tensorflow as tf
keras = tf.keras
Sequential = keras.models.Sequential
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Flatten = keras.layers.Flatten
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D

class ZariTFLite:
    def __init__(self, input_size=13, num_classes=2, use_image_input=False,
                 resource_profile="medium", img_height=64, img_width=64, img_channels=3):
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.use_image_input = use_image_input
        self.resource_profile = resource_profile
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        
        self.model = None
        self.tflite_model = None
        self.interpreter = None

        if use_image_input:
            self._build_image_model()
        else:
            self._build_feature_model()

        print(f"[ZariTFLite] Keras model initialized (profile: {self.resource_profile})")
        if self.model:
            print(f"[ZariTFLite] Keras model params: {self.model.count_params()}")
        print(f"[ZariTFLite] Using {'image' if use_image_input else 'feature'}-based input")

    def _build_feature_model(self):
        if self.resource_profile == "low":
            model = Sequential([
                keras.Input(shape=(self.input_size,), name='input_layer'),
                Dense(10, activation='relu', name='dense_1'),
                Dropout(0.1),
                Dense(5, activation='relu', name='dense_2'),
                Dense(self.num_classes, activation='softmax', name='output_layer')
            ])
        elif self.resource_profile == "medium":
            model = Sequential([
                keras.Input(shape=(self.input_size,), name='input_layer'),
                Dense(16, activation='relu', name='dense_1'),
                Dropout(0.2),
                Dense(8, activation='relu', name='dense_2'),
                Dense(self.num_classes, activation='softmax', name='output_layer')
            ])
        elif self.resource_profile == "high":
            model = Sequential([
                keras.Input(shape=(self.input_size,), name='input_layer'),
                Dense(32, activation='relu', name='dense_1'),
                Dropout(0.3),
                Dense(16, activation='relu', name='dense_2'),
                Dropout(0.2),
                Dense(8, activation='relu', name='dense_3'),
                Dense(self.num_classes, activation='softmax', name='output_layer')
            ])
        else:
            model = Sequential([
                keras.Input(shape=(self.input_size,), name='input_layer'),
                Dense(8, activation='relu', name='dense_1'),
                Dense(self.num_classes, activation='softmax', name='output_layer')
            ])
        
        self.model = model
        self.compile_model()

    def _build_image_model(self):
        input_shape = (self.img_height, self.img_width, self.img_channels)

        if self.resource_profile == "low":
            model = Sequential([
                keras.Input(shape=input_shape, name='input_layer'),
                Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_1'),
                MaxPooling2D(2, 2),
                Flatten(),
                Dense(10, activation='relu', name='dense_1'),
                Dropout(0.2),
                Dense(self.num_classes, activation='softmax', name='output_layer')
            ])
        elif self.resource_profile == "medium":
            model = Sequential([
                keras.Input(shape=input_shape, name='input_layer'),
                Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_1'),
                MaxPooling2D(2, 2),
                Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_2'),
                MaxPooling2D(2, 2),
                Flatten(),
                Dense(20, activation='relu', name='dense_1'),
                Dropout(0.25),
                Dense(self.num_classes, activation='softmax', name='output_layer')
            ])
        elif self.resource_profile == "high":
            model = Sequential([
                keras.Input(shape=input_shape, name='input_layer'),
                Conv2D(32, (5, 5), activation='relu', padding='same', name='conv_1'),
                MaxPooling2D(2, 2),
                Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2'),
                MaxPooling2D(2, 2),
                Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_3'),
                MaxPooling2D(2, 2),
                Flatten(),
                Dense(64, activation='relu', name='dense_1'),
                Dropout(0.3),
                Dense(self.num_classes, activation='softmax', name='output_layer')
            ])
        else: 
            model = Sequential([
                keras.Input(shape=input_shape, name='input_layer'),
                Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_1'),
                MaxPooling2D(2, 2),
                Flatten(),
                Dense(10, activation='relu', name='dense_1'),
                Dense(self.num_classes, activation='softmax', name='output_layer')
            ])

        self.model = model
        self.compile_model()

    def compile_model(self):
        if self.model:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )


    def convert_to_tflite(self, quantize=True):
        if self.model is None:
            raise ValueError("Keras model not initialized. Build model first.")

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if self.resource_profile == "low" or self.device_type == "resource_constrained":
                print("[ZariTFLite] Applying INT8 quantization.")
                
                def representative_dataset_gen():
                    num_calibration_steps = 100
                    if self.use_image_input:
                        input_shape = self.model.input_shape
                        for _ in range(num_calibration_steps):
                            yield [np.random.uniform(0.0, 1.0, size=(1, input_shape[1], input_shape[2], input_shape[3])).astype(np.float32)]
                    else:
                        for _ in range(num_calibration_steps):
                            yield [np.random.uniform(0.0, 1.0, size=(1, self.input_size)).astype(np.float32)]

                converter.representative_dataset = representative_dataset_gen
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        
        self.tflite_model = converter.convert()
        model_size_kb = len(self.tflite_model) / 1024
        print(f"[ZariTFLite] Converted Keras model to TFLite. Profile: {self.resource_profile}, Quantized: {quantize}, Size: {model_size_kb:.2f} KB")
        return self.tflite_model
    
    def save_tflite_model(self, file_path="model.tflite"):
        if self.tflite_model is None:
            self.convert_to_tflite()
            
        with open(file_path, 'wb') as f:
            f.write(self.tflite_model)
        
        print(f"[ZariTFLite] TFLite model saved to {file_path}")
    
    def load_tflite_model(self, model_content=None, file_path=None):
        if model_content is not None:
            self.tflite_model = model_content
        elif file_path is not None:
            with open(file_path, 'rb') as f:
                self.tflite_model = f.read()
        else:
            raise ValueError("Either model_content or file_path must be provided")

        self.interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"[ZariTFLite] TFLite model loaded with input shape: {self.input_details[0]['shape']}")
    
    def predict(self, input_data):
        if self.interpreter is None:
            raise ValueError("TFLite model not loaded. Call load_tflite_model first.")

        input_shape = self.input_details[0]['shape']
        if input_data.shape != tuple(input_shape):
             input_data = np.reshape(input_data, input_shape)

        input_dtype = self.input_details[0]['dtype']
        if input_data.dtype != input_dtype:
            input_data = input_data.astype(input_dtype)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return output_data
    
    def evaluate(self, x_test, y_test):
        if self.interpreter is None:
            raise ValueError("TFLite model not loaded. Call load_tflite_model first.")
        
        y_pred = []
        for i in range(len(x_test)):
            sample = np.expand_dims(x_test[i], axis=0)
            output = self.predict(sample)
            prediction = np.argmax(output, axis=1)[0]
            y_pred.append(prediction)

        y_pred = np.array(y_pred)
        accuracy = np.mean(y_pred == y_test)

        true_positives = np.sum((y_test == 1) & (y_pred == 1))
        false_positives = np.sum((y_test == 0) & (y_pred == 1))

        metrics = {
            'accuracy': accuracy * 100,
            'true_positives': true_positives,
            'false_positives': false_positives
        }
        
        return metrics