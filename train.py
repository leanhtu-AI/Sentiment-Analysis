from transformers import TFAutoModel
from variables import train_tf_dataset, STEPS_PER_EPOCH, VALIDATION_STEPS, early_stop_callback, df_train, val_tf_dataset
from config import EPOCHS, MODEL_PATH
from utils.tokenizer import PRETRAINED_MODEL
from create_model import create_model

def main():
    pretrained_bert = TFAutoModel.from_pretrained(PRETRAINED_MODEL, output_hidden_states=True)
    model = create_model(pretrained_bert)

    # Assume val_tf_dataset is defined similarly to train_tf_dataset
    history = model.fit(
        train_tf_dataset,
        validation_data=val_tf_dataset,
        validation_steps=VALIDATION_STEPS,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        callbacks=[early_stop_callback],
        verbose=1
    )
    model.save_weights(f'{MODEL_PATH}/v2.weights.h5')

if __name__ == "__main__":
    main()