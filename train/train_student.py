
import sys

# adding Folder_2 to the system path
sys.path.insert(0, 'utils/')
from config import EPOCHS
from create_model import create_model_stu
from tokenizer import PRETRAINED_MODEL
from transformers import TFAutoModel
from variables_student import (
    STEPS_PER_EPOCH_STU,
    VALIDATION_STEPS_STU,
    checkpoint_callback,
    early_stopping,
    train_stu_dataset,
    val_stu_dataset,
)


def main():
    pretrained_bert = TFAutoModel.from_pretrained(PRETRAINED_MODEL, output_hidden_states=True)
    model = create_model_stu(pretrained_bert)
    
    model.fit(
        train_stu_dataset,
        validation_data=val_stu_dataset,
        validation_steps=VALIDATION_STEPS_STU,
        steps_per_epoch=STEPS_PER_EPOCH_STU,
        epochs=EPOCHS,
        callbacks=[early_stopping,checkpoint_callback], 
        verbose=1
    )
    
if __name__ == "__main__":
    main()