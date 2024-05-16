from transformers import TFAutoModel

from create_model import create_model
from utils.config import EPOCHS
from utils.tokenizer import PRETRAINED_MODEL
from utils.variables import STEPS_PER_EPOCH, VALIDATION_STEPS, checkpoint_callback, early_stopping, train_tf_dataset, val_tf_dataset


def main():
    pretrained_bert = TFAutoModel.from_pretrained(PRETRAINED_MODEL, output_hidden_states=True)
    model = create_model(pretrained_bert)
    
    model.fit(
        train_tf_dataset,
        validation_data=val_tf_dataset,
        validation_steps=VALIDATION_STEPS,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        callbacks=[early_stopping,checkpoint_callback], 
        verbose=1
    )
    
if __name__ == "__main__":
    main()