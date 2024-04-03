from keras.models import Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.layers import Layer
from tensorflow.keras.optimizers import AdamW
from transformers import TFAutoModel
from variables import train_tf_dataset, STEPS_PER_EPOCH, VALIDATION_STEPS, early_stop_callback, df_train, val_tf_dataset
from config import MAX_SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS, MODEL_PATH
from utils.tokenizer import PRETRAINED_MODEL

class CustomLayer(Layer):
    def __init__(self, pretrained_bert, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.pretrained_bert = pretrained_bert

    def call(self, inputs):
        # Call the TensorFlow function with inputs using pretrained_bert
        return self.pretrained_bert(input_ids = inputs['input_ids'], token_type_ids=inputs['token_type_ids'], attention_mask=inputs['attention_mask']).hidden_states

def create_model(pretrained_bert):
    inputs = {
        'input_ids'     : Input((MAX_SEQUENCE_LENGTH,), dtype='int32', name='input_ids'),
        'token_type_ids': Input((MAX_SEQUENCE_LENGTH,), dtype='int32', name='token_type_ids'),
        'attention_mask': Input((MAX_SEQUENCE_LENGTH,), dtype='int32', name='attention_mask')
    }

    # Define a custom layer to handle TensorFlow function call
    custom_layer = CustomLayer(pretrained_bert)
    hidden_states = custom_layer(inputs)

    # Combine hidden states
    pooled_output = concatenate(
        tuple([hidden_states[i] for i in range(-4, 0)]),
        name = 'last_4_hidden_states',
        axis = -1
    )[:, 0, :]
    x = Dropout(0.2)(pooled_output)
    print(pooled_output)

    outputs = concatenate([
        Dense(
            units = 4,
            activation = 'softmax',
            name = label.replace('#', '-').replace('&', '_'),
        )(x) for label in df_train.columns[1:]
    ], axis = -1)

    optimizer = AdamW(learning_rate=1e-4, weight_decay=0.004)  # Set your learning rate and weight decay
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    return model
