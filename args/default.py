TERMS_PER_CONTEXT = 50
DROPOUT_KEEP_PROB = 0.5
TEST_EVERY_K_EPOCH = 5
EPOCHS_COUNT = 150
BAGS_PER_MINIBATCH = 2
LEARNING_RATE = 0.1
BAG_SIZE = 1

# Considering to forcely terminate training process in case when
# training accuracy becomes greater than the limit value.
TRAIN_ACC_LIMIT = 0.98

# Specific of the particular experiment, therefore
# we disable such limitation by default.
TRAIN_F1_LIMIT = None