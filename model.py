import tensorflow as tf
import numpy as np


pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = False
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

class NoisyImage() : 
    def __init__(self,image,pretrained_model,decode_predictions): 
        self.image = image 
        self.pretrained_model = pretrained_model
        self.decode_predictions = decode_predictions
    # Helper function to preprocess the image so that it can be inputted in MobileNetV2
    def preprocess(self,image):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (224, 224))
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = image[None, ...]
        return image

    # Helper function to extract labels from probability vector
    def get_imagenet_label(self,image,pretrained_model):
        probs = pretrained_model.predict(image)
        image_index = np.argmax(probs)
        label = tf.one_hot(image_index, probs.shape[-1])
        label = tf.reshape(label, (1, probs.shape[-1]))
        return label

    def pretrained__model(self) : 
        return self.pretrained_model,self.decode_predictions

    def create_adversarial_pattern(self,image):
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        image = self.preprocess(image)
        MobileNet_model,_ = self.pretrained__model()
        label = self.get_imagenet_label(image, MobileNet_model)
        with tf.GradientTape() as tape:
            tape.watch(image)
            MobileNet_model,_ = self.pretrained__model()
            prediction = MobileNet_model(image)
            loss = loss_object(label, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, image)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        return signed_grad
    def hacked_image_defined_eps(self) : 
        ## we give epsilon the value 0.1, 0.01, 0.5
        eps = 0.1
        print(self.preprocess(self.image).shape)
        print(self.create_adversarial_pattern(self.image).shape)
        adv_x = self.preprocess(self.image) + eps*self.create_adversarial_pattern(self.image)
        MobileNet_model,decode_prediction = self.pretrained__model()
        hackedImage_prediction = MobileNet_model.predict(adv_x)
        normalImage_prediction = MobileNet_model.predict(self.preprocess(self.image))
        decoded_hacked_prediction = decode_prediction(hackedImage_prediction, top=1)[0][0]
        decoded_prediction = decode_prediction(normalImage_prediction, top=1)[0][0]
        return decoded_hacked_prediction, decoded_prediction
        
    def hacked_image(self,eps) : 
        ## we give epsilon the value 0.01
        print(self.preprocess(self.image).shape)
        print(self.create_adversarial_pattern(self.image).shape)
        adv_x = self.preprocess(self.image) + eps*self.create_adversarial_pattern(self.image)
        MobileNet_model,decode_prediction = self.pretrained__model()
        hackedImage_prediction = MobileNet_model.predict(adv_x)
        normalImage_prediction = MobileNet_model.predict(self.preprocess(self.image))
        decoded_hacked_prediction = decode_prediction(hackedImage_prediction, top=1)[0][0]
        decoded_prediction = decode_prediction(normalImage_prediction, top=1)[0][0]
        return decoded_hacked_prediction, decoded_prediction

# if __name__ == "__main__":
#     pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,
#                                                        weights='imagenet')
#     # image = cv2.imread('/home/oussa/Desktop/Projet_milouda/gettyimages-475636556-612x612.jpg')
#     pretrained_model.trainable = False
#     decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
#     # hack = NoisyImage(image,pretrained_model,decode_predictions).hacked_image()
#     # print(hack)