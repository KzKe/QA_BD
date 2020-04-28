import transformers
from transformers import BertTokenizer

class config:
    LEARNING_RATE = 4e-5
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    EPOCHS = 3
    TRAINING_FILE = "../input/tweet-train-folds-v2/train_8folds.csv"
    # ROBERTA_PATH = "../input/roberta-base"
    # TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    #     vocab_file=f"{ROBERTA_PATH}/vocab.json", 
    #     merges_file=f"{ROBERTA_PATH}/merges.txt", 
    #     lowercase=True,
    #     add_prefix_space=True
    # )

    Bert_name = 'bert-base-chinese'
    OKENIZER = BertTokenizer.from_pretrained(Bert_name)
