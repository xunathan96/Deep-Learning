import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText

"""
PROVIDED STARTING CODE FOR HOMEWORK 5
"""

################################
# DataLoader
################################

# set up fields
TEXT = data.Field()
LABEL = data.Field(sequential=False,dtype=torch.long)

# make splits for data
# DO NOT MODIFY: fine_grained=True, train_subtrees=False
train, val, test = datasets.SST.splits(
    TEXT, LABEL, root='./data/.data', fine_grained=True, train_subtrees=False)

# print information about the data
print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))
# {'text': ['The', 'Rock', 'is', 'destined', 'to', 'be', 'the', '21st', 'Century', "'s", 'new', '``', 'Conan', "''", 'and', 'that', 'he', "'s", 'going', 'to', 'make', 'a', 'splash', 'even', 'greater', 'than', 'Arnold', 'Schwarzenegger', ',', 'Jean-Claud', 'Van', 'Damme', 'or', 'Steven', 'Segal', '.'], 
# 'label': 'positive'}

# build the vocabulary
# you can use other pretrained vectors, refer to https://github.com/pytorch/text/blob/master/torchtext/vocab.py
TEXT.build_vocab(train, vectors=Vectors(name='vector.txt', cache='./data'))
LABEL.build_vocab(train)

# We can also see the vocabulary directly using either the stoi (string to int) or itos (int to string) method.
# itos
# LIST: integer ids (index) --> token strings (values)
# stoi
# DICTIONARY: token strings (keys) --> integer ids (value)
print(TEXT.vocab.itos[:10])
# ['<unk>', '<pad>', '.', ',', 'the', 'and', 'a', 'of', 'to', "'s"]
print(LABEL.vocab.stoi)
# {'<unk>': 0, 'positive': 1, 'negative': 2, 'neutral': 3, 'very positive': 4, 'very negative': 5})
print(TEXT.vocab.freqs.most_common(20))
#[('.', 8024), (',', 7131), ('the', 6037), ('and', 4431), ('a', 4403), ('of', 4386), ('to', 2995), ("'s", 2544), ('is', 2536), ('that', 1915), ('in', 1789), ('it', 1775), ('The', 1265), ('as', 1200), ('film', 1152), ('but', 1076), ('with', 1071), ('for', 963), ('movie', 959), ('its', 912)]

# print vocab information
print('len(TEXT.vocab)', len(TEXT.vocab))
# 18280
print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())
# torch.Size([18280, 300])  ~each vector/embedding has dim 300

# make iterator for splits
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=64)

# print batch information
batch = next(iter(train_iter)) # for batch in train_iter
print(batch.text) # input sequence
print(batch.label) # groud truth

# Attention: batch.label in the range [1,5] not [0,4] !!!





################################
# After build your network 
################################

# Copy the pre-trained word embeddings we loaded earlier into the embedding layer of our model.
pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)

# you should maintain a nn.embedding layer in your network
model.embedding.weight.data.copy_(pretrained_embeddings)

