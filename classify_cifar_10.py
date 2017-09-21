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
    tf.app.flags.DEFINE_integer('num_epochs', "0", """ n_epochs """)
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
        self.config = config

        with tf.variable_scope('global_epoch'):
            self.global_epoch_tensor = tf.Variable(-1, trainable=False, name='global_epoch')
            self.global_epoch_input = tf.placeholder('int32', None, name='global_epoch_input')
            self.global_epoch_assign_op = self.global_epoch_tensor.assign(self.global_epoch_input)

        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

        with tf.variable_scope('best_iou'):
            # Save the best iou on validation
            self.best_iou_tensor = tf.Variable(0, trainable=False, name='best_iou')
            self.best_iou_input = tf.placeholder('float32', None, name='best_iou_input')
            self.best_iou_assign_op = self.best_iou_tensor.assign(self.best_iou_input)

    def build(self):
        self.X= tf.placeholder(tf.float32,[None,32,32,3])
        self.y=tf.placeholder(tf.uint8,[None])
        self.training = tf.placeholder(tf.bool)
        self.reg=tf.placeholder(tf.float32)

        self.conv1=BasicModel.conv_bn_relu_pool(self.X,16,self.straining,(3,3),self.reg)
        self.conv2=BasicModel.conv_bn_relu_pool(self.conv1,32,self.training,(3,3),self.reg)
        self.conv3=BasicModel.conv_bn_relu_pool(self.conv2,32,self.training(3,3),self.reg)
        self.flatten=tf.reshape(self.conv3,shape=[-1,512])
        self.fc1=BasicModel.affine_bn_relu(self.flatten,256,0.3,self.training,self.reg)
        self.fc2=BasicModel.affine_bn_relu(self.fc1,128,0.3,self.training,sef.reg)
        self.scores=BasicModel.affine_bn_relu(self.fc2,10,self.training,self.reg)

        self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.scores))
        self.optimizer=tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)



    @staticmethod
    def conv_bn_relu_pool(x, num_filters, train_flag, filter_size=(3, 3),reg=0):
        conv = tf.layers.conv2d(
            x,
            num_filters,
            filter_size,
            strides=(1, 1),
            padding='same',
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),

        )
        bn = tf.layers.batch_normalization(conv, training=train_flag)
        maxpool = tf.layers.max_pooling2d(bn, (2, 2), (2, 2), 'same')
        out = tf.nn.relu(maxpool)
        return out

    @staticmethod
    def affine_bn_relu(x, out_size, dropout_prob, train_flag,reg=0):
        affine = tf.layers.dense(
            x,
            out_size,
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
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
        #####################################################################################
        self.load_data()

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
        print("Loading Training data..")
        self.train_data = {'X': np.load(self.config.data_dir + "X_train.npy"),
                           'Y': np.load(self.config.data_dir + "y_train.npy")}
        self.train_data_len = self.train_data['X'].shape[0] - self.train_data['X'].shape[0] % self.config.batch_size
        self.num_iterations_training_per_epoch = (
                                                     self.train_data_len + self.config.batch_size - 1) // self.config.batch_size
        print("Train-shape-x -- " + str(self.train_data['X'].shape) + " " + str(self.train_data_len))
        print("Train-shape-y -- " + str(self.train_data['Y'].shape))
        print("Num of iterations on training data in one epoch -- " + str(self.num_iterations_training_per_epoch))
        print("Training data is loaded")

        print("Loading Validation data..")
        self.val_data = {'X': np.load(self.config.data_dir + "X_val.npy"),
                         'Y': np.load(self.config.data_dir + "y_val.npy")}
        self.val_data_len = self.val_data['X'].shape[0] - self.val_data['X'].shape[0] % self.config.batch_size
        self.num_iterations_validation_per_epoch = (
                                                       self.val_data_len + self.config.batch_size - 1) // self.config.batch_size
        print("Val-shape-x -- " + str(self.val_data['X'].shape) + " " + str(self.val_data_len))
        print("Val-shape-y -- " + str(self.val_data['Y'].shape))
        print("Num of iterations on validation data in one epoch -- " + str(self.num_iterations_validation_per_epoch))
        print("Validation data is loaded")

        print("Loading Testing data..")
        self.test_data = {'X': np.load(self.config.data_dir + "X_test.npy"),
                          'Y': np.load(self.config.data_dir + "y_test.npy")}
        self.test_data_len = self.test_data['X'].shape[0] - self.test_data['X'].shape[0] % self.config.batch_size
        self.num_iterations_test_per_epoch = (
                                                 self.test_data_len + self.config.batch_size - 1) // self.config.batch_size
        print("test-shape-x -- " + str(self.test_data['X'].shape) + " " + str(self.test_data_len))
        print("test-shape-y -- " + str(self.test_data['Y'].shape))
        print("Num of iterations on test data in one epoch -- " + str(self.num_iterations_test_per_epoch))
        print("test data is loaded")

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

    def generator(self, _data, _len):
        start = 0
        data = _data
        len = _len
        new_epoch_flag = True
        idx = None
        while True:
            # init index array if it is a new_epoch
            if new_epoch_flag:
                idx = np.random.choice(len, len, replace=False)
                new_epoch_flag = False

            # select the mini_batches
            mask = idx[start:start + self.config.batch_size]
            x_batch = data['X'][mask]
            y_batch = data['Y'][mask]

            # update start idx
            start += self.config.batch_size

            if start >= len:
                start = 0
                new_epoch_flag = True

            yield x_batch, y_batch

    def train(self):
        print("Training mode will begin NOW ..")
        for cur_epoch in range(self.model.global_epoch_tensor.eval(self.sess) + 1, self.config.num_epochs + 1, 1):

            # init tqdm and get the epoch value
            tt = tqdm(self.generator(self.train_data, self.train_data_len),
                      total=self.num_iterations_training_per_epoch,
                      desc="epoch-" + str(cur_epoch) + "-")

            # init acc and loss lists
            loss_list = []
            acc_list = []

            summaries_merged = None

            # loop by the number of iterations
            for x_batch, y_batch in tt:
                # get the cur_it for the summary
                cur_it = self.model.global_step_tensor.eval(self.sess)

                # Feed this variables to the network
                # TODO
                feed_dict = {}

                # TODO revise it
                # run the feed_forward
                _, loss, acc, summaries_merged = self.sess.run(
                    [self.model.train_op, self.model.loss, self.model.accuracy, self.model.merged_summaries],
                    feed_dict=feed_dict)

                # log loss and acc
                loss_list += [loss]
                acc_list += [acc]
                # summarize
                self.add_summary(cur_it, summaries_merged=summaries_merged)

                # Update the Global step
                self.model.global_step_assign_op.eval(session=self.sess,
                                                      feed_dict={self.model.global_step_input: cur_it + 1})

            # log loss and acc
            total_loss = np.mean(loss_list)
            total_acc = np.mean(acc_list)
            # summarize
            summaries_dict = dict()
            summaries_dict['train-loss-per-epoch'] = total_loss
            summaries_dict['train-acc-per-epoch'] = total_acc
            self.add_summary(cur_it, summaries_dict=summaries_dict, summaries_merged=summaries_merged)

            # Update the Global step
            self.model.global_step_assign_op.eval(session=self.sess,
                                                  feed_dict={self.model.global_step_input: cur_it + 1})

            # Update the Cur Epoch tensor
            # it is the last thing because if it is interrupted it repeat this
            self.model.global_epoch_assign_op.eval(session=self.sess,
                                                   feed_dict={self.model.global_epoch_input: cur_epoch + 1})

            # print in console
            tt.close()
            print("epoch-" + str(cur_epoch) + "-" + "loss:" + str(total_loss) + "-" + " acc:" + str(total_acc)[
                                                                                                :6])

            self.save_model()

            # val the model on validation
            if cur_epoch % 2 == 0:
                self.val(step=self.model.global_step_tensor.eval(self.sess),
                         epoch=self.model.global_epoch_tensor.eval(self.sess))

        print("Training Finished")

    def val(self, step, epoch):
        print("Validation at step:" + str(step) + " at epoch:" + str(epoch) + " ..")

        # init tqdm and get the epoch value
        tt = tqdm(range(self.num_iterations_validation_per_epoch), total=self.num_iterations_validation_per_epoch,
                  desc="Val-epoch-" + str(epoch) + "-")

        # init acc and loss lists
        loss_list = []
        acc_list = []

        # idx of minibatch
        idx = 0

        # reset metrics
        self.metrics.reset()

        # get the maximum iou to compare with and save the best model
        max_iou = self.model.best_iou.tensor.eval(self.sess)

        # loop by the number of iterations
        for cur_iteration in tt:
            # load minibatches
            x_batch = self.val_data['X'][idx:idx + self.args.batch_size]
            y_batch = self.val_data['Y'][idx:idx + self.args.batch_size]

            # update idx of minibatch
            idx += self.args.batch_size

            # Feed this variables to the network
            feed_dict = {self.model.x_pl: x_batch,
                         self.model.y_pl: y_batch,
                         self.model.is_training: False
                         }

            # Run the feed forward but the last iteration finalize what you want to do
            if cur_iteration < self.num_iterations_validation_per_epoch - 1:

                # run the feed_forward
                out_argmax, loss, acc, summaries_merged = self.sess.run(
                    [self.model.out_argmax, self.model.loss, self.model.accuracy, self.model.merged_summaries],
                    feed_dict=feed_dict)
                # log loss and acc
                loss_list += [loss]
                acc_list += [acc]
                # log metrics
                self.metrics.update_metrics_batch(out_argmax, y_batch)

            else:

                # run the feed_forward
                out_argmax, loss, acc, summaries_merged, segmented_imgs = self.sess.run(
                    [self.model.out_argmax, self.model.loss, self.model.accuracy,
                     self.model.merged_summaries, self.model.segmented_summary],
                    feed_dict=feed_dict)
                # log loss and acc
                loss_list += [loss]
                acc_list += [acc]
                # log metrics
                self.metrics.update_metrics_batch(out_argmax, y_batch)
                # mean over batches
                total_loss = np.mean(loss_list)
                total_acc = np.mean(acc_list)
                mean_iou = self.metrics.compute_final_metrics(self.num_iterations_validation_per_epoch)
                # summarize
                summaries_dict = dict()
                summaries_dict['val-loss-per-epoch'] = total_loss
                summaries_dict['val-acc-per-epoch'] = total_acc
                summaries_dict['mean_iou_on_val'] = mean_iou
                summaries_dict['val_prediction_sample'] = segmented_imgs
                self.add_summary(step, summaries_dict=summaries_dict, summaries_merged=summaries_merged)

                # print in console
                tt.close()
                print("Val-epoch-" + str(epoch) + "-" + "loss:" + str(total_loss) + "-" +
                      "acc:" + str(total_acc)[:6] + "-mean_iou:" + str(mean_iou))
                if mean_iou > max_iou:
                    print("This validation got a new best iou. so we will save this one")
                    # save the best model
                    self.save_best_model()
                    # Set the new maximum
                    self.model.best_iou_assign_op.eval(session=self.sess,
                                                       feed_dict={self.model.best_iou_input: mean_iou})

                # Break the loop to finalize this epoch
                break

    def test(self):
        print("Testing mode will begin NOW..")

        # init tqdm and get the epoch value
        tt = tqdm(range(self.test_data_len))

        # init acc and loss lists
        loss_list = []
        acc_list = []
        img_list = []

        # idx of image
        idx = 0

        # reset metrics
        self.metrics.reset()

        # loop by the number of iterations
        for cur_iteration in tt:
            # load mini_batches
            x_batch = self.test_data['X'][idx:idx + 1]
            y_batch = self.test_data['Y'][idx:idx + 1]

            # update idx of mini_batch
            idx += 1

            # Feed this variables to the network
            feed_dict = {self.model.x_pl: x_batch,
                         self.model.y_pl: y_batch,
                         self.model.is_training: False
                         }

            # run the feed_forward
            out_argmax, loss, acc, summaries_merged, segmented_imgs = self.sess.run(
                [self.model.out_argmax, self.model.loss, self.model.accuracy,
                 self.model.merged_summaries, self.model.segmented_summary],
                feed_dict=feed_dict)

            # log loss and acc
            loss_list += [loss]
            acc_list += [acc]
            img_list += [segmented_imgs[0]]

            # log metrics
            self.metrics.update_metrics(out_argmax[0], y_batch[0], 0, 0)

        # mean over batches
        total_loss = np.mean(loss_list)
        total_acc = np.mean(acc_list)
        mean_iou = self.metrics.compute_final_metrics(self.test_data_len)

        # print in console
        tt.close()
        print("Here the statistics")
        print("Total_loss: " + str(total_loss))
        print("Total_acc: " + str(total_acc)[:6])
        print("mean_iou: " + str(mean_iou))

        print("Plotting imgs")
        for i in range(len(img_list)):
            plt.imsave(self.args.imgs_dir + 'test_' + str(i) + '.png', img_list[i])


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
        operator.save_model()
        operator.test()
    elif args.mode == 'train':
        operator.train()
        operator.save_model()
    else:
        operator.test()


if __name__ == '__main__':
    main()
