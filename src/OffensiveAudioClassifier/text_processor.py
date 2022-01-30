import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

import pandas as pd
import numpy as np

class TextProcessor():

    def __init__(self, sentences, model_dir='OLID_ALBERTBase', tokenizer_model_name='albert-base-v2'):
        self.model_dir = model_dir
        self.tokenizer_model_name = tokenizer_model_name
        self.data_loader = self.load_data(sentences)
        
        self.model_path = '../OffensiveTextClassifier/models/' + self.model_dir

    def load_data(self, sentences, clean=False):
        tokenizer = AlbertTokenizer.from_pretrained(self.tokenizer_model_name)

        # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
        sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]

        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

        MAX_LEN = 128

        # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        # Pad our input tokens
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        # Create attention masks
        attention_masks = []

        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            seq_mask = [float(i>0) for i in seq]
            attention_masks.append(seq_mask) 

        prediction_inputs = torch.tensor(input_ids)
        prediction_masks = torch.tensor(attention_masks)
        
        batch_size = 2  

        prediction_data = TensorDataset(prediction_inputs, prediction_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
        return prediction_dataloader

    def get_sentence_embeddings(self, predict=False):
        model = AlbertForSequenceClassification.from_pretrained(self.model_path)
        
        # Put model in evaluation mode
        model.eval()
        predictions = []
        all_sentence_embeddings = np.array([[]])

        # Predict 
        for batch in self.data_loader:
        
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask = batch

            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, return_dict=True, output_hidden_states=True)
                logits = outputs.logits

                # The model returns all the hidden states of all layers (12) + output from embeddings
                hidden_states = outputs.hidden_states
                last_layer_hidden_states = hidden_states[-1]

                # Move data to CPU and convert tesnor to numpy 3D array
                last_layer_hidden_states = last_layer_hidden_states.detach().cpu().numpy()
                sentence_embeddings = last_layer_hidden_states[:,0,:]

                all_sentence_embeddings = np.vstack((all_sentence_embeddings,sentence_embeddings)) if all_sentence_embeddings.size else sentence_embeddings     

            # Move logits and labels to CPU
            logits = logits.numpy() #.detach().cpu()

            # Store predictions and true labels
            predictions.append(logits)
        all_sentence_embeddings = pd.DataFrame(all_sentence_embeddings)
        if predict:
            return (all_sentence_embeddings, np.argmax(predictions[0], axis=1)) 
        return all_sentence_embeddings
        
DATA_DIR = '../../Data'
METADATA_FILE_NAME = '/compiled_data.csv'
if __name__ == '__main__':
    # Create sentence and label lists
    # sentences = np.array(["hey ya bitch !!", "how are you"])
    df = pd.read_csv(DATA_DIR + METADATA_FILE_NAME)
    print("Data Loaded")
    print(df.shape)

    tp = TextProcessor(df['Text'])

    # tp = TextProcessor(sentences=sentences)
    (_, predictions) = tp.get_sentence_embeddings(predict=True)
    print(predictions.shape)
    print(type(predictions))
    print(predictions)
    print(df['Label'])
    report = classification_report(df['Label'], predictions, zero_division=0)
    print(report)

    cm = confusion_matrix(df['Label'], predictions)
    ConfusionMatrixDisplay(cm, display_labels=["non-offensive","offensive"]).plot()
    
