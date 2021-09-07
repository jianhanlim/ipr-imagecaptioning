import tensorflow as tf
import numpy as np

from base_model import BaseModel
from utils import resnet_v1, inception_v4


class CaptionGenerator(BaseModel):
    def build(self):
        """ Build the model. """
        self.build_cnn()
        self.build_rnn()
        if self.is_train:
            self.build_optimizer()
            self.build_summary()
        if self.config.cnn == 'resnet101':
            self.saver_cnn = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_v1_101/'))

        if self.config.cnn == 'inceptionv4':
            self.saver_cnn = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='InceptionV4/'))

        self.saver = tf.train.Saver(max_to_keep=10)

    def build_cnn(self):
        """ Build the CNN. """
        print("Building the CNN...")
        if self.config.cnn == 'vgg16':
            self.build_vgg16()
        elif self.config.cnn == 'resnet50':
            self.build_resnet50()
        elif self.config.cnn == 'resnet101':
            self.build_resnet101()
        elif self.config.cnn == 'inceptionv4':
            self.build_inceptionv4()
        else:
            print('Incorrect CNN is selected')
            exit(-1)
        print("CNN built.")

    def build_vgg16(self):
        """ Build the VGG16 net. """
        config = self.config

        images = tf.placeholder(
            dtype = tf.float32,
            shape = [config.batch_size] + self.image_shape)

        conv1_1_feats = self.nn.conv2d(images, 64, name = 'conv1_1')
        conv1_2_feats = self.nn.conv2d(conv1_1_feats, 64, name = 'conv1_2')
        pool1_feats = self.nn.max_pool2d(conv1_2_feats, name = 'pool1')

        conv2_1_feats = self.nn.conv2d(pool1_feats, 128, name = 'conv2_1')
        conv2_2_feats = self.nn.conv2d(conv2_1_feats, 128, name = 'conv2_2')
        pool2_feats = self.nn.max_pool2d(conv2_2_feats, name = 'pool2')

        conv3_1_feats = self.nn.conv2d(pool2_feats, 256, name = 'conv3_1')
        conv3_2_feats = self.nn.conv2d(conv3_1_feats, 256, name = 'conv3_2')
        conv3_3_feats = self.nn.conv2d(conv3_2_feats, 256, name = 'conv3_3')
        pool3_feats = self.nn.max_pool2d(conv3_3_feats, name = 'pool3')

        conv4_1_feats = self.nn.conv2d(pool3_feats, 512, name = 'conv4_1')
        conv4_2_feats = self.nn.conv2d(conv4_1_feats, 512, name = 'conv4_2')
        conv4_3_feats = self.nn.conv2d(conv4_2_feats, 512, name = 'conv4_3')
        pool4_feats = self.nn.max_pool2d(conv4_3_feats, name = 'pool4')

        conv5_1_feats = self.nn.conv2d(pool4_feats, 512, name = 'conv5_1')
        conv5_2_feats = self.nn.conv2d(conv5_1_feats, 512, name = 'conv5_2')
        conv5_3_feats = self.nn.conv2d(conv5_2_feats, 512, name = 'conv5_3')

        reshaped_conv5_3_feats = tf.reshape(conv5_3_feats,
                                            [config.batch_size, 196, 512])

        self.conv_feats = reshaped_conv5_3_feats
        self.num_ctx = 196
        self.dim_ctx = 512
        self.images = images

    def build_resnet50(self):
        """ Build the ResNet50. """
        config = self.config

        images = tf.placeholder(
            dtype = tf.float32,
            shape = [config.batch_size] + self.image_shape)

        conv1_feats = self.nn.conv2d(images,
                                  filters = 64,
                                  kernel_size = (7, 7),
                                  strides = (2, 2),
                                  activation = None,
                                  name = 'conv1')
        conv1_feats = self.nn.batch_norm(conv1_feats, 'bn_conv1')
        conv1_feats = tf.nn.relu(conv1_feats)
        pool1_feats = self.nn.max_pool2d(conv1_feats,
                                      pool_size = (3, 3),
                                      strides = (2, 2),
                                      name = 'pool1')

        res2a_feats = self.resnet_block(pool1_feats, 'res2a', 'bn2a', 64, 1)
        res2b_feats = self.resnet_block2(res2a_feats, 'res2b', 'bn2b', 64)
        res2c_feats = self.resnet_block2(res2b_feats, 'res2c', 'bn2c', 64)

        res3a_feats = self.resnet_block(res2c_feats, 'res3a', 'bn3a', 128)
        res3b_feats = self.resnet_block2(res3a_feats, 'res3b', 'bn3b', 128)
        res3c_feats = self.resnet_block2(res3b_feats, 'res3c', 'bn3c', 128)
        res3d_feats = self.resnet_block2(res3c_feats, 'res3d', 'bn3d', 128)

        res4a_feats = self.resnet_block(res3d_feats, 'res4a', 'bn4a', 256)
        res4b_feats = self.resnet_block2(res4a_feats, 'res4b', 'bn4b', 256)
        res4c_feats = self.resnet_block2(res4b_feats, 'res4c', 'bn4c', 256)
        res4d_feats = self.resnet_block2(res4c_feats, 'res4d', 'bn4d', 256)
        res4e_feats = self.resnet_block2(res4d_feats, 'res4e', 'bn4e', 256)
        res4f_feats = self.resnet_block2(res4e_feats, 'res4f', 'bn4f', 256)

        res5a_feats = self.resnet_block(res4f_feats, 'res5a', 'bn5a', 512)
        res5b_feats = self.resnet_block2(res5a_feats, 'res5b', 'bn5b', 512)
        res5c_feats = self.resnet_block2(res5b_feats, 'res5c', 'bn5c', 512)

        reshaped_res5c_feats = tf.reshape(res5c_feats,
                                         [config.batch_size, 49, 2048])

        self.conv_feats = reshaped_res5c_feats
        self.num_ctx = 49
        self.dim_ctx = 2048
        self.images = images

    def resnet_block(self, inputs, name1, name2, c, s=2):
        """ A basic block of ResNet. """
        branch1_feats = self.nn.conv2d(inputs,
                                    filters = 4*c,
                                    kernel_size = (1, 1),
                                    strides = (s, s),
                                    activation = None,
                                    use_bias = False,
                                    name = name1+'_branch1')
        branch1_feats = self.nn.batch_norm(branch1_feats, name2+'_branch1')

        branch2a_feats = self.nn.conv2d(inputs,
                                     filters = c,
                                     kernel_size = (1, 1),
                                     strides = (s, s),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2a')
        branch2a_feats = self.nn.batch_norm(branch2a_feats, name2+'_branch2a')
        branch2a_feats = tf.nn.relu(branch2a_feats)

        branch2b_feats = self.nn.conv2d(branch2a_feats,
                                     filters = c,
                                     kernel_size = (3, 3),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2b')
        branch2b_feats = self.nn.batch_norm(branch2b_feats, name2+'_branch2b')
        branch2b_feats = tf.nn.relu(branch2b_feats)

        branch2c_feats = self.nn.conv2d(branch2b_feats,
                                     filters = 4*c,
                                     kernel_size = (1, 1),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2c')
        branch2c_feats = self.nn.batch_norm(branch2c_feats, name2+'_branch2c')

        outputs = branch1_feats + branch2c_feats
        outputs = tf.nn.relu(outputs)
        return outputs

    def resnet_block2(self, inputs, name1, name2, c):
        """ Another basic block of ResNet. """
        branch2a_feats = self.nn.conv2d(inputs,
                                     filters = c,
                                     kernel_size = (1, 1),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2a')
        branch2a_feats = self.nn.batch_norm(branch2a_feats, name2+'_branch2a')
        branch2a_feats = tf.nn.relu(branch2a_feats)

        branch2b_feats = self.nn.conv2d(branch2a_feats,
                                     filters = c,
                                     kernel_size = (3, 3),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2b')
        branch2b_feats = self.nn.batch_norm(branch2b_feats, name2+'_branch2b')
        branch2b_feats = tf.nn.relu(branch2b_feats)

        branch2c_feats = self.nn.conv2d(branch2b_feats,
                                     filters = 4*c,
                                     kernel_size = (1, 1),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2c')
        branch2c_feats = self.nn.batch_norm(branch2c_feats, name2+'_branch2c')

        outputs = inputs + branch2c_feats
        outputs = tf.nn.relu(outputs)
        return outputs

    def build_resnet101(self):
        """ Build the ResNet101. """
        config = self.config

        global_pool = False
        num_neu = 49
        if config.globalpool == 'yes':
            global_pool = True
            num_neu = 1

        images = tf.placeholder(
            dtype = tf.float32,
            shape = [config.batch_size] + self.image_shape)

        arg_scope = resnet_v1.resnet_arg_scope()
        with tf.contrib.slim.arg_scope(arg_scope):
            res_feats, end_points = resnet_v1.resnet_v1_101(images,
                                                is_training=self.train_cnn,
                                                global_pool=global_pool,
                                                scope='resnet_v1_101')

        reshaped_res5c_feats = tf.reshape(res_feats,
                                         [config.batch_size, num_neu, 2048])

        self.conv_feats = reshaped_res5c_feats
        self.num_ctx = num_neu
        self.dim_ctx = 2048
        self.images = images

    def build_inceptionv4(self):
        """ Build the inceptionv4. """
        config = self.config

        images = tf.placeholder(
            dtype = tf.float32,
            shape = [config.batch_size] + self.image_shape)

        arg_scope = inception_v4.inception_v4_arg_scope()
        with tf.contrib.slim.arg_scope(arg_scope):
            res_feats, end_points = inception_v4.inception_v4(images,
                                                num_classes=None,
                                                is_training=self.train_cnn,
                                                scope='InceptionV4')

        res_feats = end_points['Mixed_7d']	# take before avg pooling

        reshaped_res5c_feats = tf.reshape(res_feats,
                                         [config.batch_size, 64, 1536])

        self.conv_feats = reshaped_res5c_feats
        self.num_ctx = 64
        self.dim_ctx = 1536
        self.images = images

    def build_rnn(self):
        """ Build the RNN. """
        print("Building the RNN...")
        config = self.config

        # Setup the placeholders
        if self.is_train:
            contexts = self.conv_feats

            sentences = tf.placeholder(
                dtype = tf.int32,
                shape = [config.batch_size, config.max_caption_length])
            masks = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.max_caption_length])
            key = tf.placeholder(
                dtype = tf.float32,
                shape = [config.num_lstm_units])
            signkey = tf.placeholder(
                dtype = tf.float32,
                shape = [config.num_lstm_units])
        else:
            contexts = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, self.num_ctx, self.dim_ctx])
            last_memory = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.num_lstm_units])
            last_output_ori = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.num_lstm_units])
            last_word = tf.placeholder(
                dtype = tf.int32,
                shape = [config.batch_size])
            key = tf.placeholder(
                dtype = tf.float32,
                shape = [config.num_lstm_units])

        # Setup the word embedding
        with tf.variable_scope("word_embedding"):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [config.vocabulary_size, config.dim_embedding],
                initializer = self.nn.fc_kernel_initializer,
                regularizer = self.nn.fc_kernel_regularizer,
                trainable = self.is_train)

        # Setup the LSTM
        lstm = tf.nn.rnn_cell.LSTMCell(
            config.num_lstm_units,
            initializer = self.nn.fc_kernel_initializer)
        if self.is_train:
            lstm = tf.nn.rnn_cell.DropoutWrapper(
                lstm,
                input_keep_prob = 1.0-config.lstm_drop_rate,
                output_keep_prob = 1.0-config.lstm_drop_rate,
                state_keep_prob = 1.0-config.lstm_drop_rate)

        # Initialize the LSTM using the mean context
        with tf.variable_scope("initialize"):
            context_mean = tf.reduce_mean(self.conv_feats, axis = 1)
            initial_memory, initial_output = self.initialize(context_mean)
            initial_state = initial_memory, initial_output

        # Prepare to run
        predictions = []
        if self.is_train:
            alphas = []
            cross_entropies = []
            signs = []
            predictions_correct = []
            num_steps = config.max_caption_length
            last_output_ori = initial_output
            last_memory = initial_memory
            last_word = tf.zeros([config.batch_size], tf.int32)
        else:
            num_steps = 1

        # add key to hidden state
        last_output = tf.math.multiply(last_output_ori, key)
        last_state = last_memory, last_output

        # Generate the words one by one
        for idx in range(num_steps):
            # Attention mechanism
            with tf.variable_scope("attend"):
                # shape: (32, 49)
                alpha = self.attend(contexts, last_output)

                # shape: context - (32, 2048), contexts - (32, 49, 2048)
                context = tf.reduce_sum(contexts*tf.expand_dims(alpha, 2),
                                        axis = 1)

                if self.is_train:
                    tiled_masks = tf.tile(tf.expand_dims(masks[:, idx], 1),
                                         [1, self.num_ctx])
                    masked_alpha = alpha * tiled_masks
                    alphas.append(tf.reshape(masked_alpha, [-1]))

            # Embed the last word
            with tf.variable_scope("word_embedding"):
                word_embed = tf.nn.embedding_lookup(embedding_matrix,
                                                    last_word)
           # Apply the LSTM
            with tf.variable_scope("lstm"):
                current_input = tf.concat([context, word_embed], 1)
                output, state = lstm(current_input, last_state)
                memory, _ = state

            # Decode the expanded output of LSTM into a word
            with tf.variable_scope("decode"):
                expanded_output = tf.concat([output,
                                             context,
                                             word_embed],
                                             axis = 1,)
                logits = self.decode(expanded_output)
                probs = tf.nn.softmax(logits)
                prediction = tf.argmax(logits, 1)
                predictions.append(prediction)

            # Compute the loss for this step, if necessary
            if self.is_train:
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels = sentences[:, idx],
                    logits = logits)
                masked_cross_entropy = cross_entropy * masks[:, idx]
                cross_entropies.append(masked_cross_entropy)
                
                # sign loss, = (alpha * relu(-hiddenstate * signkey + 0.1)).sum()
                # hiddenstate (32x512), signkey(512), output (1)
                sign = tf.reduce_sum(config.sign_alpha * tf.nn.relu((tf.math.multiply(-output, signkey) + 0.1)))
                signs.append(sign)

                ground_truth = tf.cast(sentences[:, idx], tf.int64)
                prediction_correct = tf.where(
                    tf.equal(prediction, ground_truth),
                    tf.cast(masks[:, idx], tf.float32),
                    tf.cast(tf.zeros_like(prediction), tf.float32))
                predictions_correct.append(prediction_correct)

                # add key to hidden state
                last_output = tf.math.multiply(output, key)
                last_memory = memory
                last_state = memory, last_output
                last_word = sentences[:, idx]

            tf.get_variable_scope().reuse_variables()

        # Compute the final loss, if necessary
        if self.is_train:
            cross_entropies = tf.stack(cross_entropies, axis = 1)
            cross_entropy_loss = tf.reduce_sum(cross_entropies) \
                                 / tf.reduce_sum(masks)
            sign_loss = tf.reduce_sum(tf.stack(signs)) / (config.batch_size * num_steps)

            alphas = tf.stack(alphas, axis = 1)
            alphas = tf.reshape(alphas, [config.batch_size, self.num_ctx, -1])
            attentions = tf.reduce_sum(alphas, axis = 2)
            diffs = tf.ones_like(attentions) - attentions
            attention_loss = config.attention_loss_factor \
                             * tf.nn.l2_loss(diffs) \
                             / (config.batch_size * self.num_ctx)

            reg_loss = tf.losses.get_regularization_loss()

            total_loss = cross_entropy_loss + attention_loss + reg_loss + sign_loss

            predictions_correct = tf.stack(predictions_correct, axis = 1)
            accuracy = tf.reduce_sum(predictions_correct) \
                       / tf.reduce_sum(masks)

        self.contexts = contexts
        self.key = key
        if self.is_train:
            self.sentences = sentences
            self.masks = masks
            self.total_loss = total_loss
            self.cross_entropy_loss = cross_entropy_loss
            self.attention_loss = attention_loss
            self.reg_loss = reg_loss
            self.sign_loss = sign_loss
            self.accuracy = accuracy
            self.attentions = attentions
            self.signkey = signkey
        else:
            self.initial_memory = initial_memory
            self.initial_output = initial_output
            self.last_memory = last_memory
            self.last_output = last_output_ori
            self.last_word = last_word
            self.memory = memory
            self.output = output
            self.probs = probs

        print("RNN built.")

    def initialize(self, context_mean):
        """ Initialize the LSTM using the mean context. """
        config = self.config
        context_mean = self.nn.dropout(context_mean)
        if config.num_initalize_layers == 1:
            # use 1 fc layer to initialize
            memory = self.nn.dense(context_mean,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_a')
            output = self.nn.dense(context_mean,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_b')
        else:
            # use 2 fc layers to initialize
            temp1 = self.nn.dense(context_mean,
                                  units = config.dim_initalize_layer,
                                  activation = tf.tanh,
                                  name = 'fc_a1')
            temp1 = self.nn.dropout(temp1)
            memory = self.nn.dense(temp1,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_a2')

            temp2 = self.nn.dense(context_mean,
                                  units = config.dim_initalize_layer,
                                  activation = tf.tanh,
                                  name = 'fc_b1')
            temp2 = self.nn.dropout(temp2)
            output = self.nn.dense(temp2,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_b2')
        return memory, output

    def attend(self, contexts, output):
        """ Attention Mechanism. """
        config = self.config
        reshaped_contexts = tf.reshape(contexts, [-1, self.dim_ctx])
        reshaped_contexts = self.nn.dropout(reshaped_contexts)
        output = self.nn.dropout(output)
        if config.num_attend_layers == 1:
            # use 1 fc layer to attend
            logits1 = self.nn.dense(reshaped_contexts,
                                    units = 1,
                                    activation = None,
                                    use_bias = False,
                                    name = 'fc_a')
            logits1 = tf.reshape(logits1, [-1, self.num_ctx])
            logits2 = self.nn.dense(output,
                                    units = self.num_ctx,
                                    activation = None,
                                    use_bias = False,
                                    name = 'fc_b')
            logits = logits1 + logits2
        else:
            # use 2 fc layers to attend
            temp1 = self.nn.dense(reshaped_contexts,
                                  units = config.dim_attend_layer,
                                  activation = tf.tanh,
                                  name = 'fc_1a')
            temp2 = self.nn.dense(output,
                                  units = config.dim_attend_layer,
                                  activation = tf.tanh,
                                  name = 'fc_1b')
            temp2 = tf.tile(tf.expand_dims(temp2, 1), [1, self.num_ctx, 1])
            temp2 = tf.reshape(temp2, [-1, config.dim_attend_layer])
            temp = temp1 + temp2
            temp = self.nn.dropout(temp)
            logits = self.nn.dense(temp,
                                   units = 1,
                                   activation = None,
                                   use_bias = False,
                                   name = 'fc_2')
            logits = tf.reshape(logits, [-1, self.num_ctx])
        alpha = tf.nn.softmax(logits)
        return alpha

    def decode(self, expanded_output):
        """ Decode the expanded output of the LSTM into a word. """
        config = self.config
        expanded_output = self.nn.dropout(expanded_output)
        if config.num_decode_layers == 1:
            # use 1 fc layer to decode
            logits = self.nn.dense(expanded_output,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'fc')
        else:
            # use 2 fc layers to decode
            temp = self.nn.dense(expanded_output,
                                 units = config.dim_decode_layer,
                                 activation = tf.tanh,
                                 name = 'fc_1')
            temp = self.nn.dropout(temp)
            logits = self.nn.dense(temp,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'fc_2')
        return logits

    def build_optimizer(self):
        """ Setup the optimizer and training operation. """
        config = self.config

        # learning_rate = tf.constant(config.initial_learning_rate)
        # if config.learning_rate_decay_factor < 1.0:
        #     def _learning_rate_decay_fn(learning_rate, global_step):
        #         return tf.train.exponential_decay(
        #             learning_rate,
        #             global_step,
        #             decay_steps = config.num_steps_per_decay,
        #             decay_rate = config.learning_rate_decay_factor,
        #             staircase = True)
        #     learning_rate_decay_fn = _learning_rate_decay_fn
        # else:
        #     learning_rate_decay_fn = None
        learning_rate = tf.train.exponential_decay(config.initial_learning_rate, self.global_step,
                                                   config.num_steps_per_decay, config.learning_rate_decay_factor,
                                                   staircase=True)

        with tf.variable_scope('optimizer/OptimizeLoss', reuse = tf.AUTO_REUSE):
            if config.optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(
                    learning_rate = learning_rate,
                    beta1 = config.beta1,
                    beta2 = config.beta2,
                    epsilon = config.epsilon
                    )
            elif config.optimizer == 'RMSProp':
                optimizer = tf.train.RMSPropOptimizer(
                    learning_rate = learning_rate,
                    decay = config.decay,
                    momentum = config.momentum,
                    centered = config.centered,
                    epsilon = config.epsilon
                )
            elif config.optimizer == 'Momentum':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate = learning_rate,
                    momentum = config.momentum,
                    use_nesterov = config.use_nesterov
                )
            else:
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate = learning_rate
                )

            # opt_op = tf.contrib.layers.optimize_loss(
            #     loss = self.total_loss,
            #     global_step = self.global_step,
            #     learning_rate = learning_rate,
            #     optimizer = optimizer,
            #     clip_gradients = config.clip_gradients,
            #     learning_rate_decay_fn = learning_rate_decay_fn)

            # make sure loss is computed before compute gradient
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.total_loss = tf.identity(self.total_loss)

            # accumulate gradient
            # Retrieve all trainable variables you defined in your graph
            tvs = tf.trainable_variables()

            # Creation of a list of variables with the same shape as the trainable ones
            # initialized with 0s
            accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
            zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

            # Calls the compute_gradients function of the optimizer to obtain... the list of gradients
            gvs = optimizer.compute_gradients(self.total_loss, tvs)

            # Adds to each element from the list you initialized earlier with zeros its gradient (works because accum_vars and gvs are in the same order)
            increment_global_step = tf.assign(self.global_step, self.global_step+1)
            accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]

            # Define the training step (part with variable value update)
            accum_vars_clip, _ = tf.clip_by_global_norm(accum_vars, config.clip_gradients)
            train_opt = optimizer.apply_gradients([(accum_vars_clip[i], gv[1]) for i, gv in enumerate(gvs)])

            # ensure gradients are updated
            with tf.control_dependencies([train_opt]):
                opt_op = tf.identity(self.global_step)

        self.zero_ops = zero_ops
        self.increment_global_step = increment_global_step
        self.accum_ops = accum_ops
        self.opt_op = opt_op

    def build_summary(self):
        """ Build the summary (for TensorBoard visualization). """

        with tf.name_scope("metrics"):
            tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)
            tf.summary.scalar("attention_loss", self.attention_loss)
            tf.summary.scalar("reg_loss", self.reg_loss)
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.scalar("accuracy", self.accuracy)

        with tf.name_scope("attentions"):
            self.variable_summary(self.attentions)

        self.summary = tf.summary.merge_all()

    def variable_summary(self, var):
        """ Build the summary for a variable. """
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
