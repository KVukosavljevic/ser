from torchvision import transforms as tf

def transforms():
    return tf.Compose(
            [tf.ToTensor(), tf.Normalize((0.5,), (0.5,))]
        )