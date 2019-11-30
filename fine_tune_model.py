import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertAdam, BertForSequenceClassification
from sklearn.metrics import matthews_corrcoef
import pandas as pd
import numpy as np

MAX_LEN = 128
RANDOM_STATE = 2018
TEST_SIZE = 0.1
BATCH_SIZE = 32
EPOCHS = 4

def get_cuda_device():
    if not torch.cuda.is_available():
        raise SystemError('cuda unavailable')
    device = torch.device('cuda')
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)
    return device

def parse_csv(path_to_corpus, column_names, delimiter='\t', header=None):
    return pd.read_csv(
            path_to_corpus,
            names=column_names,
            delimiter=delimiter,
            header=header)

def RENAME_ME(corpus_df, pretrained_corpus, do_lower_case, max_len):

    sentences = corpus_df.sentence.values
    sentences = ['[CLS] ' + sentence + ' [SEP]' for sentence in sentences]
    labels = corpus_df.label.values

    tokenizer = BertTokenizer.from_pretrained(pretrained_corpus, do_lower_case=do_lower_case)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(
            input_ids,
            maxlen=max_len,
            dtype='long',
            truncating='post',
            padding='post')

    attention_masks = []
    for seq in input_ids:
        seq_masks = [float(i>0) for i in seq]
        attention_masks.append(seq_masks)

    return input_ids, labels, attention_masks

def get_data_loader(inputs, masks, labels, batch_size):
    dataset = TensorDataset(*map(torch.tensor, [inputs, masks, labels]))
    sampler = RandomSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)

def get_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def fine_tune(corpus_name, train_corpus, dev_corpus, column_names):

    device = get_cuda_device()

    train_corpus_df = parse_csv(train_corpus, column_names)
    input_ids, labels, attention_masks = RENAME_ME(
            train_corpus_df,
            corpus_name,
            True,
            MAX_LEN)
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(
            input_ids,
            labels,
            random_state=RANDOM_STATE,
            test_size=TEST_SIZE)
    train_masks, test_masks, _, _ = train_test_split(
            attention_masks,
            input_ids,
            random_state=RANDOM_STATE,
            test_size=TEST_SIZE)
    train_data_loader = get_data_loader(train_inputs, train_labels, train_masks, BATCH_SIZE)
    test_data_loader = get_data_loader(test_inputs, test_labels, test_masks, BATCH_SIZE)

    model = BertForSequenceClassification.from_pretrained(corpus_name, num_labels=2)
    model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {
            'params': [ p for n, p in param_optimizer if not any(nd in n for nd in no_decay) ],
            'weight_decay_rate': 0.01
        },
        {
            'params': [ p for n, p in param_optimizer if any(nd in n for nd in no_decay) ],
            'weight_decay_rate': 0.0
        }
    ]

    optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=2e-5,
            warmup=.1)

    for _ in range(EPOCHS):

        model.train()

        for step, batch in enumerate(train_data_loader):
            input_ids, mask, labels = tuple(t.to(device) for t in batch)
            optimizer.zero_grad()
            loss = model(input_ids, token_type_ids=None, attention_mask=mask, labels=labels)
            loss.backward()
            optimizer.step()

        model.eval()

        for batch in test_data_loader:
            input_ids, mask, labels = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits = model(input_ids, token_type_ids=None, attention_mask=mask)
            #logits = logits.detach().cpu().numpy()
            #label_ids = labels.to('cpu').numpy()

    dev_corpus_df = parse_csv(dev_corpus, column_names)
    dev_input_ids, dev_labels, dev_attentions_masks = RENAME_ME(
            dev_corpus_df,
            corpus_name,
            True,
            MAX_LEN)
    dev_data_loader = get_data_loader(dev_input_ids, dev_labels, dev_masks, BATCH_SIZE)

    model.eval()

    predictions = []
    true_labels = []
    for batch in prediction_data_loader:
        input_ids, mask, labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(input_ids, token_type_ids=None, attention_mask=mask)
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        predictions.append(logits)
        true_labels.append(label_ids)

    matthews_set = []

    for true_label, prediction in zip(true_labels, predictions):
        matthews = matthews_corrcoef(true_label, np.argmax(prediction, axis=1).flatten())
        matthews_set.append(matthews)

    flat_predictions = [ item for sublist in predictions for item in sublist ]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [ item for sublist in true_labels for item in sublist ]

    matthews_corrcoef(flat_true_labels, flat_predictions)

if __name__ == '__main__':

    fine_tune('bert-base-uncased',
         '../corpora/cola_public/raw/in_domain_train.tsv',
         '../corpora/cola_public/raw/out_of_domain_dev.tsv',
         ['sentence_source', 'label', 'label_notes', 'sentence'])

