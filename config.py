import transformers
from transformers import BertTokenizer

class config:
    LEARNING_RATE = 4e-5
    # warmup_proportion = 

    MAX_LEN = 512
    QUESTION_MAXLEN = 100
    CONTEXT_MAXLEN = 400

    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    hidden_size = 768

    EPOCHS = 3

    TRAINING_FILE = '/content/drive/My Drive/dureader_robust-data/train.json'
    DEV_FILE = '/content/drive/My Drive/dureader_robust-data/dev.json'
    TEST_FILE = '/content/drive/My Drive/Baidu _Reading_Comprehension/dureader_robust-test1/test1.json'
    # ROBERTA_PATH = "../input/roberta-base"
    # TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    #     vocab_file=f"{ROBERTA_PATH}/vocab.json", 
    #     merges_file=f"{ROBERTA_PATH}/merges.txt", 
    #     lowercase=True,
    #     add_prefix_space=True
    # )

    Bert_name = 'bert-base-chinese'

    ROBERTA_BASE = 'hfl/chinese-roberta-wwm-ext'
    ROBERTA_LARGE = 'hfl/chinese-roberta-wwm-ext-large'

    BERT_BASE = 'hfl/chinese-roberta-wwm-ext'
    BERT_LARGE = 'hfl/chinese-roberta-wwm-ext-large'

    TOKENIZER = BertTokenizer.from_pretrained(Bert_name, lowercase=False)


    


    

