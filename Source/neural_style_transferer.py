from keras import backend as K
from keras.layers import Input
from keras.optimizers import Adam
from keras.models import Model
from keras.regularizers import Regularizer

from layers import InputNormalize, Denormalize, conv_bn_relu, res_conv, dconv_bn_nolinear
import numpy as np
import cv2
import sys


class TVRegularizer(Regularizer):
    """ Enforces smoothness in image output. """

    def __init__(self, weight=1.0):
        self.weight = weight
        self.uses_learning_phase = False
        super(TVRegularizer, self).__init__()

    def __call__(self, x):
        assert K.ndim(x.output) == 4
        x_out = x.output

        shape = K.shape(x_out)
        img_width, img_height, channel = (shape[1], shape[2], shape[3])
        size = img_width * img_height * channel
        if K.image_dim_ordering() == 'th':
            a = K.square(x_out[:, :, :img_width - 1, :img_height - 1] - x_out[:, :, 1:, :img_height - 1])
            b = K.square(x_out[:, :, :img_width - 1, :img_height - 1] - x_out[:, :, :img_width - 1, 1:])
        else:
            a = K.square(x_out[:, :img_width - 1, :img_height - 1, :] - x_out[:, 1:, :img_height - 1, :])
            b = K.square(x_out[:, :img_width - 1, :img_height - 1, :] - x_out[:, :img_width - 1, 1:, :])
        loss = self.weight * K.sum(K.pow(a + b, 1.25))
        return loss


def add_total_variation_loss(transform_output_layer, weight):
    # Total Variation Regularization
    layer = transform_output_layer  # Output layer
    tv_regularizer = TVRegularizer(weight)(layer)
    layer.add_loss(tv_regularizer)


def image_transform_net_simple(img_width, img_height, tv_weight=1):
    x = Input(shape=(img_width, img_height, 3))
    a = InputNormalize()(x)

    a = conv_bn_relu(8, 9, 9, stride=(1, 1))(a)
    a = conv_bn_relu(16, 3, 3, stride=(2, 2))(a)
    a = conv_bn_relu(32, 3, 3, stride=(2, 2))(a)
    for i in range(2):
        a = res_conv(32, 3, 3)(a)
    a = dconv_bn_nolinear(16, 3, 3)(a)
    a = dconv_bn_nolinear(8, 3, 3)(a)
    a = dconv_bn_nolinear(3, 9, 9, stride=(1, 1), activation="tanh")(a)

    # Scale output to range [0, 255] via custom Denormalize layer
    y = Denormalize(name='transform_output')(a)

    model = Model(inputs=x, outputs=y)

    if tv_weight > 0:
        add_total_variation_loss(model.layers[-1], tv_weight)

    return model


def load_model(model_path, image_width, image_height, channels=3):
    # Important: flip width and height here. Because OpenCV assumes (width, height), while
    # numpy assumes (height, width)
    model = image_transform_net_simple(image_height, image_width, channels)
    model.compile(Adam(), loss='mse')
    model.load_weights(model_path)
    return model


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage: %s [video_file_path] [output_file_path] [model_file_path]" % sys.argv[0])
        exit(0)

    video_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    model_file_path = sys.argv[3]
    input_video = cv2.VideoCapture(video_file_path)

    if not input_video.isOpened():
        print("Can't open input file %s" % video_file_path)
        exit(0)
    else:
        # get vcap propertys
        width = int(input_video.get(cv2.cv2.CAP_PROP_FRAME_WIDTH))
        height = int(input_video.get(cv2.cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(input_video.get(cv2.cv2.CAP_PROP_FRAME_COUNT))
        frame_size = (width, height)
        fourcc = int(input_video.get(cv2.cv2.CAP_PROP_FOURCC))
        fps = int(input_video.get(cv2.cv2.CAP_PROP_FPS))

        pos_frame = input_video.get(cv2.cv2.CAP_PROP_POS_FRAMES)

        output_video = cv2.VideoWriter()

        print("Starting to load model")
        model = load_model(model_file_path, width, height)

        print("Starting to neural style transfer")

        while input_video.get(cv2.cv2.CAP_PROP_POS_FRAMES) < input_video.get(cv2.cv2.CAP_PROP_FRAME_COUNT):
            print("\rTransferring frame %d/%d" % (int(pos_frame), frame_count), end="")
            _, frame = input_video.read()  # get the frame

            # store the current frame in as a numpy array
            transferred_frame = np.expand_dims(np.copy(frame), axis=0)
            transferred_frame = model.predict(transferred_frame)[0] # TODO: batch-process and batch-write?
            transferred_frame = cv2.cvtColor(transferred_frame, cv2.COLOR_BGR2RGB)

            if not output_video.isOpened():
                output_video.open(output_file_path, fourcc, fps, (transferred_frame.shape[1], transferred_frame.shape[0]), True)

            output_video.write(transferred_frame.astype(np.uint8))
            pos_frame = input_video.get(cv2.cv2.CAP_PROP_POS_FRAMES)

    print("Finished neural style transfer")

    input_video.release()
    output_video.release()
