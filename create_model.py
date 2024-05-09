from keras.layers import Dense, Dropout, Input, Layer, concatenate
from keras.models import Model
from tensorflow.keras.optimizers import AdamW

from utils.config import DROP_OUT, LEARNING_RATE, MAX_SEQUENCE_LENGTH, WEIGHT_DECAY
from utils.variables import df_train


class CustomLayer(Layer):
      def __init__(self, pretrained_bert, **kwargs):
            super(CustomLayer, self).__init__(**kwargs)
            self.pretrained_bert = pretrained_bert

      def call(self, inputs):
            # Call the TensorFlow function with inputs using pretrained_bert
            return self.pretrained_bert(input_ids=inputs['input_ids'], token_type_ids=inputs['token_type_ids'], attention_mask=inputs['attention_mask']).hidden_states

      def get_config(self):
            config = super().get_config()
            return config

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
        tuple([hidden_states[i] for i in range(-3, 0)]),
        name = 'last_3_hidden_states',
        axis = -1
    )[:, 0, :]
    x = Dropout(DROP_OUT)(pooled_output)

    outputs = concatenate([
        Dense(
            units = 4,
            activation = 'softmax',
            name = label.replace('#', '-').replace('&', '_'),
        )(x) for label in df_train.columns[1:]
    ], axis = -1)

    optimizer = AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)  # Set your learning rate and weight decay
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    return model


