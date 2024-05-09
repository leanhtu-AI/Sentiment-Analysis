from utils.tokenizer import call_tokenizer
from utils.config import TEST_PATH
import pandas as pd
df_test = pd.read_csv(TEST_PATH)


tokenizer = call_tokenizer()

