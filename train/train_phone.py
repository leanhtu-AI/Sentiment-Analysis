
import sys
# adding Folder_2 to the system path
sys.path.insert(0, 'utils/')
from transformers import TFAutoModel
from variables_phone import STEPS_PER_EPOCH_PHONE, VALIDATION_STEPS_PHONE, early_stopping, df_train_phone, checkpoint_callback,train_phone_dataset,val_phone_dataset
from config import EPOCHS
from tokenizer import PRETRAINED_MODEL
from create_model import create_model_phone

def main():
    pretrained_bert = TFAutoModel.from_pretrained(PRETRAINED_MODEL, output_hidden_states=True)
    model = create_model_phone(pretrained_bert)
    
    history = model.fit(
        train_phone_dataset,
        validation_data=val_phone_dataset,
        validation_steps=VALIDATION_STEPS_PHONE,
        steps_per_epoch=STEPS_PER_EPOCH_PHONE,
        epochs=EPOCHS,
        callbacks=[early_stopping,checkpoint_callback], 
        verbose=1
    )
    
if __name__ == "__main__":
    main()