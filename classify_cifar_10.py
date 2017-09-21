"""
This file will contain Models to classify cifar-10 dataset
Keep Calm and be ready to classify
"""

import tensorflow as tf
import numpy as np
import os
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


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def create_exp_dirs(args):
    """
    Create experiment and out dirs
    :param args: Arguments of the program
    :return: args , The new one which contains all needed dirs
    """
    args.data_dir = os.path.realpath(os.getcwd()) + "/data/" + args.data_dir + "/"
    args.exp_dir = os.path.realpath(os.getcwd()) + "/experiments/" + args.exp_dir + "/"
    args.summary_dir = args.exp_dir + 'summaries/'
    args.checkpoint_dir = args.exp_dir + 'checkpoints/'
    args.checkpoint_best_dir = args.exp_dir + 'checkpoints/best/'

    dirs_to_be_created = [args.checkpoint_dir,
                          args.checkpoint_best_dir,
                          args.summary_dir]
    # Create the dirs if it is not exist
    create_dirs(dirs_to_be_created)

    return args


class BasicModel:
    def __init__(self, config):
        pass

    def build(self):
        pass

    @staticmethod
    def conv_bn_relu_pool_drop(x, num_filters, train_flag, filter_size=(3, 3)):
        conv = tf.layers.conv2d(
            x,
            num_filters,
            filter_size,
            strides=(1, 1),
            padding='same',
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(),

        )
        bn = tf.layers.batch_normalization(conv, training=train_flag)
        maxpool = tf.layers.max_pooling2d(bn, (2, 2), (2, 2), 'same')
        out = tf.nn.relu(maxpool)

        return out

    @staticmethod
    def affine_bn_relu(X, out_size, dropout_prob, train_flag):
        affine=tf.layers.dense(
            X,
            out_size,
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(),
        )

        bn = tf.layers.batch_normalization(affine, training=train_flag)
        relu = tf.nn.relu(bn)
        out = tf.layers.dropout(relu, rate=dropout_prob, training=train_flag)
        return out


class Train:
    def __init__(self, sess, model, config):
        print("\nTraining is initializing itself\n")

        self.config = config
        self.sess = sess
        self.model = model

        print("Initializing the variables of the model")
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)
        print("Initialization finished")

        # Create a saver object
        self.saver = tf.train.Saver(max_to_keep=1,
                                    save_relative_paths=True)

        self.saver_best = tf.train.Saver(max_to_keep=1,
                                         save_relative_paths=True)

        # Load from latest checkpoint if found
        self.load_model()

        ##################################################################################
        # Init summaries

        # Summary variables
        self.scalar_summary_tags = ['train-loss-per-epoch', 'val-loss-per-epoch',
                                    'train-acc-per-epoch', 'val-acc-per-epoch']
        self.summary_tags = []
        self.summary_placeholders = {}
        self.summary_ops = {}
        # init summaries and it's operators
        self.init_summaries()
        # Create summary writer
        self.summary_writer = tf.summary.FileWriter(self.config.summary_dir, self.sess.graph)

    def save_model(self):
        """
        Save Model Checkpoint
        :return:
        """
        print("saving a checkpoint")
        self.saver.save(self.sess, self.config.checkpoint_dir, self.model.global_step_tensor)
        print("Saved a checkpoint")

    def save_best_model(self):
        """
        Save Model Checkpoint
        :return:
        """
        print("saving a checkpoint for the best model")
        self.saver_best.save(self.sess, self.config.checkpoint_best_dir, self.model.global_step_tensor)
        print("Saved a checkpoint for the best model")

    def load_model(self):
        """
        Load the latest checkpoint
        :return:
        """
        print("Searching for a checkpoint")
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("Model loaded from the latest checkpoint\n")
        else:
            print("\n.. No ckpt, SO First time to train :D ..\n")

    def load_best_model(self):
        """
        Load the latest checkpoint
        :return:
        """
        print("Searching for the best checkpoint")
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_best_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver_best.restore(self.sess, latest_checkpoint)
            print("Model loaded from the latest checkpoint\n")
        else:
            print("\n.. ERROR No BEST MODEL! ..\n")
            exit(-1)

    def load_data(self):
        pass

    def init_summaries(self):
        """
        Create the summary part of the graph
        :return:
        """
        with tf.variable_scope('train-summary-per-epoch'):
            for tag in self.scalar_summary_tags:
                self.summary_tags += tag
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])

    def add_summary(self, step, summaries_dict=None, summaries_merged=None):
        """
        Add the summaries to tensorboard
        :param step:
        :param summaries_dict:
        :param summaries_merged:
        :return:
        """
        if summaries_dict is not None:
            summary_list = self.sess.run([self.summary_ops[tag] for tag in summaries_dict.keys()],
                                         {self.summary_placeholders[tag]: value for tag, value in
                                          summaries_dict.items()})
            for summary in summary_list:
                self.summary_writer.add_summary(summary, step)
        if summaries_merged is not None:
            self.summary_writer.add_summary(summaries_merged, step)

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

    # build the model and Create the operator
    model.build()
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
