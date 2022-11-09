# use this file as a reference
# this is nlp hw for IMDB sentiment classification, tst_script is this file repurposed for time series prediction

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW
import os
from sklearn.metrics import precision_score, recall_score, f1_score
torch.manual_seed(42)
np.random.seed(42)

BATCH_SIZE = 16
EPOCHS = 6
TEST_PATH = "data/depression2labels_test.csv"
TRAIN_PATH = "data/suicide_detection_train.csv"
VAL_PATH = "data/suicide_detection_val.csv"
SAVE_PATH = "models/DistilBERT2"

def load_dataset(path):
  dataset = pd.read_csv(path)
  return dataset

train_data = load_dataset(TRAIN_PATH)
val_data = load_dataset(VAL_PATH)
test_data = load_dataset(TEST_PATH)

class DistillBERT():

  def __init__(self, model_name='distilbert-base-uncased', num_classes=3):
    # TODO(students): start
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    # TODO(students): end

  def get_tokenizer_and_model(self):
    return self.model, self.tokenizer

class DatasetLoader(Dataset):

  def __init__(self, data, tokenizer):
    self.data = data
    self.tokenizer = tokenizer

  def tokenize_data(self):
    print("Processing data..")
    tokens = []
    labels = []

    label_dict = {'suicide': 2, 'depression': 1, 'non-suicide':0}

    #200k dataset
    #label_dict = {'suicide': 1, 'non-suicide': 0}

    text_list = self.data['text'].to_list()
    label_list = self.data['class'].to_list()

    i = 0
    for (text, label) in tqdm(zip(text_list, label_list), total=len(text_list)):
      # TODO(students): start
      tokens += [self.tokenizer.encode_plus(text, max_length=512,truncation=True,return_tensors="pt").input_ids.permute(1,0)]
      labels += [label_dict[label]]
      # TODO(students): end

    tokens = pad_sequence(tokens, batch_first=True)
    labels = torch.tensor(labels)
    dataset = TensorDataset(tokens, labels)
    return dataset

  def get_data_loaders(self, batch_size=32, shuffle=True):
    processed_dataset = self.tokenize_data()

    data_loader = DataLoader(
        processed_dataset,
        shuffle=shuffle,
        batch_size=batch_size
    )

    return data_loader

class Trainer():

  def __init__(self, options):
    self.device = options['device']
    self.train_data = options['train_data']
    self.val_data = options['val_data']
    self.batch_size = options['batch_size']
    self.epochs = options['epochs']
    self.save_path = options['save_path']
    self.training_type = options['training_type']
    transformer = DistillBERT()
    self.model, self.tokenizer = transformer.get_tokenizer_and_model()
    self.model.to(self.device)

  def get_performance_metrics(self, preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    precision = precision_score(labels_flat, pred_flat, zero_division=0, average='weighted')
    recall = recall_score(labels_flat, pred_flat, zero_division=0, average='weighted')
    f1 = f1_score(labels_flat, pred_flat, zero_division=0, average='weighted')
    return precision, recall, f1

  def set_training_parameters(self):
    # TODO(students): start
    if self.training_type == "frozen_embeddings":
      for layer in self.model.named_parameters():
        layer[1].requires_grad=False
    elif self.training_type == 'top_2_training':
      for layer in self.model.named_parameters():
        if(layer[0] == 'distilbert.transformer.layer.4.attention.q_lin.weight'):
          break
        layer[1].requires_grad=False
    elif self.training_type == 'top_4_training':
      for layer in self.model.named_parameters():
        if(layer[0] == 'distilbert.transformer.layer.2.attention.q_lin.weight'):
          break
        layer[1].requires_grad=False
    # TODO(students): end

  def train(self, data_loader, optimizer):
    self.model.train()
    total_recall = 0
    total_precision = 0
    total_f1 = 0
    total_loss = 0

    for batch_idx, (texts, labels) in enumerate(tqdm(data_loader)):
      self.model.zero_grad()
      # TODO(students): start
      outputs = self.model(torch.squeeze(texts).cuda(), labels=labels.cuda())

      logits = outputs.logits
      loss = outputs.loss

      optimizer.zero_grad()
      if loss.requires_grad:
        loss.backward()
      optimizer.step()

      total_loss += float(loss)
      precision, recall, f1 = self.get_performance_metrics(logits.detach().cpu(), labels.cpu())
      total_recall += recall
      total_precision += precision
      total_f1 += f1
      #TODO(students): end

    precision = total_precision/len(data_loader)
    recall = total_recall/len(data_loader)
    f1 = total_f1/len(data_loader)
    loss = total_loss/len(data_loader)

    return precision, recall, f1, loss

  def eval(self, data_loader):
    self.model.eval()
    total_recall = 0
    total_precision = 0
    total_f1 = 0
    total_loss = 0

    with torch.no_grad():
      for (texts, labels) in tqdm(data_loader):
        # TODO(students): start
        outputs = self.model(torch.squeeze(texts).cuda(),labels=labels.cuda())
        total_loss += float(outputs.loss)

        precision, recall, f1 = self.get_performance_metrics(outputs.logits.cpu(), labels.cpu())
        total_precision += precision
        total_recall += recall
        total_f1 += f1

        # TODO(students): end
    
    precision = total_precision/len(data_loader)
    recall = total_recall/len(data_loader)
    f1 = total_f1/len(data_loader)
    loss = total_loss/len(data_loader)

    return precision, recall, f1, loss

  def save_transformer(self):
    self.model.save_pretrained(self.save_path)
    self.tokenizer.save_pretrained(self.save_path)

  def execute(self):
    last_best = 0
    train_dataset = DatasetLoader(self.train_data, self.tokenizer)
    train_data_loader = train_dataset.get_data_loaders(self.batch_size)
    val_dataset = DatasetLoader(self.val_data, self.tokenizer)
    val_data_loader = val_dataset.get_data_loaders(self.batch_size)
    optimizer = AdamW(self.model.parameters(), lr = 3e-5, eps = 1e-8)
    self.set_training_parameters()
    for epoch_i in range(0, self.epochs):
        train_precision, train_recall, train_f1, train_loss = self.train(train_data_loader, optimizer)
        print(f'Epoch {epoch_i + 1}: train_loss: {train_loss:.4f} train_precision: {train_precision:.4f} train_recall: {train_recall:.4f} train_f1: {train_f1:.4f}')
        val_precision, val_recall, val_f1, val_loss = self.eval(val_data_loader)
        print(f'Epoch {epoch_i + 1}: val_loss: {val_loss:.4f} val_precision: {val_precision:.4f} val_recall: {val_recall:.4f} val_f1: {val_f1:.4f}')

        if val_f1 > last_best:
            print("Saving model..")
            self.save_transformer()
            last_best = val_f1
            print("Model saved.")

class Tester():

  def __init__(self, options):
    self.save_path = options['save_path']
    self.device = options['device']
    self.test_data = options['test_data']
    self.batch_size = options['batch_size']
    transformer = DistillBERT(self.save_path)
    self.model, self.tokenizer = transformer.get_tokenizer_and_model()
    self.model.to(self.device)

  def get_performance_metrics(self, preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    precision = precision_score(labels_flat, pred_flat, zero_division=0, average='weighted')
    recall = recall_score(labels_flat, pred_flat, zero_division=0, average='weighted')
    f1 = f1_score(labels_flat, pred_flat, zero_division=0, average='weighted')
    return precision, recall, f1

  def test(self, data_loader):
    self.model.eval()
    total_recall = 0
    total_precision = 0
    total_f1 = 0
    total_loss = 0

    with torch.no_grad():
      for (texts, labels) in tqdm(data_loader):
        # TODO(students): start
        outputs = self.model(torch.squeeze(texts).cuda(),labels=labels.cuda())
        total_loss += float(outputs.loss)

        precision, recall, f1 = self.get_performance_metrics(outputs.logits.cpu(), labels.cpu())
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        # TODO(students): end
    
    precision = total_precision/len(data_loader)
    recall = total_recall/len(data_loader)
    f1 = total_f1/len(data_loader)
    loss = total_loss/len(data_loader)

    return precision, recall, f1, loss

  def execute(self):
    test_dataset = DatasetLoader(self.test_data, self.tokenizer)
    test_data_loader = test_dataset.get_data_loaders(self.batch_size)

    test_precision, test_recall, test_f1, test_loss = self.test(test_data_loader)

    print()
    print(f'test_loss: {test_loss:.4f} test_precision: {test_precision:.4f} test_recall: {test_recall:.4f} test_f1: {test_f1:.4f}')

#replace the lines below for different test cases
options = {}
options['batch_size'] = BATCH_SIZE
options['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
options['train_data'] = train_data
options['val_data'] = val_data
options['save_path'] = SAVE_PATH + '_top_2_training'
options['epochs'] = EPOCHS
options['training_type'] = 'top_2_training'
trainer = Trainer(options)
trainer.execute()

#testing

# options = {}
# options['batch_size'] = BATCH_SIZE
# options['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# options['test_data'] = test_data
# options['save_path'] = SAVE_PATH + '_top_2_training'
# tester = Tester(options)
# tester.execute()