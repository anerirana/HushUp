
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np

class OffensiveTextClassifier():

    def __init__(self, sentences, model_dir='OLID_ALBERTBase', tokenizer_model_name='albert-base-v2'):
        self.model_dir = model_dir
        self.tokenizer_model_name = tokenizer_model_name
        self.data_loader = self.load_data(sentences)

    # to-do: clean data, one time activity call seperately
    # def clean_data():
        # Expand contractions
        # Remove everything except alphabets, question mark & excalamtion mark (check how bert processes special charcters and include charcaters in list)

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
        model_path = '../OffensiveTextClassifier/models/' + self.model_dir
        model = AlbertForSequenceClassification.from_pretrained(model_path)
        
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
            logits = logits.detach().cpu().numpy()

            # Store predictions and true labels
            predictions.append(logits)
        
        if predict:
            return (all_sentence_embeddings, np.argmax(predictions[0], axis=1)) 
        return all_sentence_embeddings
        

if __name__ == '__main__':
    # Create sentence and label lists
    sentences = np.array(["hey ya bitch !!", "how are you"])
    otc = OffensiveTextClassifier(sentences=sentences)
    (all_sentence_embeddings, predictions) = otc.get_sentence_embeddings(predict=True)
    
    print("Embedding matrix shape: ",all_sentence_embeddings.shape)
    print("Predictions: ",predictions)
