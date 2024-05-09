import pandas as pd

from utils.config import TEST_PATH
from utils.tokenizer import call_tokenizer

df_test = pd.read_csv(TEST_PATH)


tokenizer = call_tokenizer()

