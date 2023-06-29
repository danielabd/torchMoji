import json
import numpy as np
import emoji
import torch
import torchmoji
import os
from torchmoji.sentence_tokenizer import SentenceTokenizer
import numpy as np
from torchmoji.model_def import TorchMoji
# Load the pre-trained TorchMoji model
base_dir = os.path.join(os.path.expanduser('~'),'projects/torchMoji')

with open(os.path.join(base_dir,'data','emoji_codes.json'), 'r') as f:
    emojis = json.load(f)
    print(emojis.values())

with open(os.path.join(base_dir,'model','vocabulary.json'), 'r') as f:
    vocabulary = json.load(f)
    print(vocabulary.values())

nb_classes = len(emojis.values())
nb_tokens = len(vocabulary)
model = TorchMoji(nb_classes, nb_tokens)

model_path = os.path.join(base_dir,'model/pytorch_model.bin')

device = f"cuda" if torch.cuda.is_available() else "cpu"#todo: change
if 'cpu' in device:
    checkpoints = torch.load(model_path, map_location=torch.device('cpu'))
else:
    checkpoints = torch.load(model_path, map_location='cuda:0')

model.load_state_dict(checkpoints)
sentence_tokenizer = SentenceTokenizer(emojis,30) #todo: check if need bigger than 30

# Tokenize your input text
text = "I am so happy!"
tokens, _, _ = sentence_tokenizer.tokenize_sentences([text])


# Use the model to predict the emoji probabilities
torch.tensor(int(tokens[0]))
probabilities = model(tokens[0])
probabilities = model.predict(tokens)[0]

# Normalize the probabilities to sum to 1
probabilities = np.squeeze(probabilities)
probabilities /= np.sum(probabilities)

# Print the top 5 emojis with their probabilities
top_emojis = probabilities.argsort()[-5:][::-1]
for emoji_idx in top_emojis:
    print(f"{emojis[str(emoji_idx)]} : {probabilities[emoji_idx]}")




model = torch.load(model_path, map_location=torch.device('cpu'))

model = torchmoji.load_model(os.path.join(base_dir,'model/pytorch_model.bin'))

from torchmoji.sentence_tokenizer import SentenceTokenizer

# Instantiate a sentence tokenizer
sentence_tokenizer = SentenceTokenizer(emojis)

# Tokenize your input text
text = "I am so happy!"
tokens, _ = sentence_tokenizer.tokenize_sentences([text])



'/home/nlp/tzufar/projects/torchMoji/model/pytorch_model.bin'
with open(os.path.join(base_dir,'model/vocabulary.json'), 'r') as f:
    vocabulary = json.load(f)
model = torchmoji.models.PyTorchModel(
    torchmoji.__path__[0] + '/model/pytorch_model.bin',
    torchmoji.model_def.TorchMoji(nb_classes=len(vocabulary)))

