from utils.tokenizer import call_tokenizer
from utils.config import TRAIN_PATH, TEST_PATH, VAL_PATH
import pandas as pd
df_test = pd.read_csv(TEST_PATH)


tokenizer = call_tokenizer()

