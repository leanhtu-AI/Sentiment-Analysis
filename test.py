from utils.config import BATCH_SIZE
from predict import reloaded_model, predict_test, replacements, categories, print_acsa_pred_test
from utils.variables import test_tf_dataset

y_pred = predict_test(reloaded_model, test_tf_dataset, BATCH_SIZE, verbose=1)
reloaded_model.evaluate(test_tf_dataset, batch_size=BATCH_SIZE, verbose=1)

print_acsa_pred_test(replacements, categories, y_pred[0])