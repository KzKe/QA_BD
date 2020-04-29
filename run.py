from config import config
from data import * 


train_examples = read_question_answer_examples(config.TRAINING_FILE)
dev_examples = read_question_answer_examples(config.DEV_FILE)
test_examples = read_question_answer_examples(config.TEST_FILE)


train_features = convert_examples_to_features(train_examples, tokenizer=config.TOKENIZER)
dev_features = convert_examples_to_features(dev_examples, tokenizer=config.TOKENIZER)
test_features = convert_examples_to_features(test_examples, tokenizer=config.TOKENIZER)

# temp = Question_Answer_Dataset(train_features)

train_features = Question_Answer_Dataset(train_features)
dev_features = Question_Answer_Dataset(dev_features)
test_features = Question_Answer_Dataset(test_features)