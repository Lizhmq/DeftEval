import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from dataset import LSTMDataset
import time, os
from sklearn.metrics import classification_report
from transformers import AdamW
from lstm_model import LSTMEncoder, LSTMClassifier


def get_model():

    vocab_size = 16000
    embedding_size = 256
    hidden_size = 256
    n_layers = 2
    num_classes = 2
    max_len = 384
    
    device = torch.device("cuda", 0)
    enc = LSTMEncoder(embedding_size, hidden_size, n_layers)
    classifier = LSTMClassifier(vocab_size, embedding_size, enc,
                                hidden_size, num_classes, max_len,
                                attn=True).to(device)
    return classifier


def adjust_learning_rate(optimizer, decay_rate=0.8):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def train(classifier, optimizer, criterion, epochs, train_loader, valid_loader, saving_path, plot_each=100, lr_decay=False):
    
    plot_losses = []
    epoch = 0
    i = 0
    print_loss_total = 0
    plot_loss_total = 0

    for epoch in range(epochs):
        classifier.train()
        print('start training epoch ' + str(epoch + 1) + '....')
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(classifier.device)
            labels = labels.to(classifier.device)
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            print_loss_total += loss.item()
            plot_loss_total += loss.item()

            if (i + 1) % plot_each == 0:
                plot_loss_avg = plot_loss_total / plot_each
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
            i += 1
        if lr_decay:
            adjust_learning_rate(optimizer, lr_decay)
        torch.save(classifier.state_dict(), os.path.join(saving_path, str(epoch + 1) + '.pt'))
        evaluate(classifier, valid_loader)

def evaluate(classifier, eval_loader):
    classifier.eval()
    correct_y, predict_y = [], []
    for inputs, labels in tqdm(eval_loader):
        inputs = inputs.to(classifier.device)
        labels = labels.to(classifier.device)
        with torch.no_grad():
            outputs = classifier(inputs) 
            preds = torch.argmax(outputs, dim=1)
            correct_y.extend(labels.cpu().numpy())
            predict_y.extend(preds.cpu().numpy())
    target_names = ["class_0", "class_1"]
    print(classification_report(correct_y, predict_y, target_names=target_names))


def main():
    data_path = "../../deft_corpus/data/deft_files/"
    save_path = "./save/lstm"
    train_dataset = LSTMDataset(file_path=data_path, file_type="train", output_dir=save_path)
    valid_dataset = LSTMDataset(file_path=data_path, file_type="dev", output_dir=save_path)
    test_dataset = LSTMDataset(file_path=data_path, file_type="test", output_dir=save_path)
    
    sampler1 = RandomSampler(train_dataset)
    sampler2 = RandomSampler(valid_dataset)
    sampler3 = RandomSampler(test_dataset)
    
    device = torch.device("cuda", 0)
    model = get_model().to(device)
    model.device = device
    train_loader = DataLoader(train_dataset, sampler=sampler1, batch_size=32, drop_last=False)
    valid_loader = DataLoader(valid_dataset, sampler=sampler2, batch_size=48, drop_last=False)
    test_loader = DataLoader(test_dataset, sampler=sampler3, batch_size=48, drop_last=False)

    lr, adam_eps = 1e-3, 1e-8
    optimizer = AdamW(model.parameters(), lr=lr, eps=adam_eps)
    criterion = nn.CrossEntropyLoss()


    # train(model, optimizer, criterion, 5, train_loader, valid_loader, save_path)
    # evaluate(model, test_loader)

    model.load_state_dict(torch.load("./save/lstm/4.pt"))
    evaluate(model, test_loader)


if __name__ == "__main__":
    main()