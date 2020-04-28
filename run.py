from config import config
from data import * 


train_examples = read_question_answer_examples(config.TRAINING_FILE)
dev_examples = read_question_answer_examples(config.DEV_FILE)
test_examples = read_question_answer_examples(config.TEST_FILE)