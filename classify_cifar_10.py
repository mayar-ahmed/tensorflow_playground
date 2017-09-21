"""
This file will contain Models to classify cifar-10 dataset
Keep Calm and be ready to classify
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm


def read_args():
    """
    Read the arguments of the process.
    :return _args: the arguments of the process
    """
    tf.app.flags.DEFINE_string('model', "", """ MODEL NAME """)
    tf.app.flags.DEFINE_integer('n_epochs', "0", """ n_epochs """)
    tf.app.flags.DEFINE_integer('batch_size', "0", """ batch_size """)
    tf.app.flags.DEFINE_float('learning_rate', "0.0", """ learning_rate """)
    tf.app.flags.DEFINE_string('data_dir', "", """ Data dir """)
    tf.app.flags.DEFINE_string('exp_dir', "", """ Experiment dir to store ckpt & summaries """)
    tf.app.flags.DEFINE_boolean('train_n_test', False, """ Finish the train with the number of epochs then test """)
    tf.app.flags.DEFINE_boolean('is_train', True, """ Whether it is a training or testing""")

    _args = tf.app.flags.FLAGS

    print("\nUsing this arguments check it\n")
    for key, value in sorted(vars(_args).items()):
        if value is not None:
            print("{} -- {} --".format(key, value))
    print("\n\n")

    return _args


class BasicModel:
    def __init__(self, config):
        pass

    @staticmethod
    def conv_bn_relu_pool(X,W,b,train):
        conv=tf.nn.conv2d(X,W,[1,1,1,1],padding='SAME')+b
        bn=tf.layers.batch_normalization(conv,axis=3,training=train)
        relu=tf.nn.relu(bn)
        dropout=tf.nn.dropout(relu,0.5)
        max1=tf.nn.max_pool(relu,[1,2,2,1],[1,2,2,1],'SAME')

        return max1

    @staticmethod
    def affine_bn_relu(X,w,b,train):
        affine=tf.matmul(X,w)+b
        bn=tf.layers.batch_normalization(affine,axis=1,training=train)
        relu=tf.nn.relu(bn)
        return relu


class Train:
    def __init__(self, sess, model, config):
        pass

    def train(self):
        pass

    def validate(self):
        pass

    def test(self):
        pass


def main():
    args = read_args()

    # Reset the graph
    tf.reset_default_graph()

    # Create the sess
    sess = tf.Session()

    # Create Model class and build it
    model = None
    if args.model == "Basic":
        model = BasicModel(config=args)
    else:
        print("ERROR model provided is not defined")
        exit(-1)

    # Create the operator
    operator = Train(sess=sess, model=model, config=args)

    if args.mode == 'train_n_test':
        operator.train()
        operator.save()
        operator.test()
    elif args.mode == 'train':
        operator.train()
        operator.save()
    else:
        operator.test()


if __name__ == '__main__':
    main()
