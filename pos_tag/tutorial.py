"""This file simply follows the POS-tagging tutorial at https://allennlp.org/tutorials"""

from typing import Dict, List, Iterator

import torch
import torch.optim as optim
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params
from allennlp.training.trainer import Trainer

class PosDatasetReader(DatasetReader):
    """ Reads a file of POS-annotated sentences, each line with a sentence on the format:
        WORD###POS WORD###POS WORD###POS
    """

    def __init__(self, 
                token_indexers: Dict[str, TokenIndexer] = {"tokens" : SingleIdTokenIndexer()}
        ) -> None:

        super().__init__()
        self.token_indexers = token_indexers

    def text_to_instance(self, tokens: List[Token], tags: List[SequenceLabelField] = None) -> Instance:
        """ Converts a list of tokens (and optionally tags) to an instance """

        sentence_field = TextField(tokens, self.token_indexers)
        fields = {'sentence' : sentence_field}

        if tags:
            tag_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields['labels'] = tag_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        """ Creates and iterator over instances from a file path """
        with open(file_path) as f:
            for line in f:
                pairs = line.split()
                words, tags = zip(*(pair.split("###") for pair in pairs))
                yield self.text_to_instance([Token(word) for word in words], tags)

class POSTagger(Model):
    """ A POS-tagger model """

    def __init__(self, 
                word_embeddings: TextFieldEmbedder,
                sequence_encoder: Seq2SeqEncoder, 
                vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.sequence_encoder = sequence_encoder

        # Fully connected layer from sequence encoding to tags
        self.fc = torch.nn.Linear(in_features = sequence_encoder.get_output_dim(),
                                    out_features = vocab.get_vocab_size('labels'))
        
        self.accuracy = CategoricalAccuracy()

    def forward(self, 
                sentence: Dict[str, torch.tensor],
                labels: torch.tensor = None
        ) -> Dict[str, torch.tensor]:

        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_output = self.sequence_encoder(embeddings, mask)

        tag_logits = self.fc(encoder_output)

        output = {'tag_logits' : tag_logits}

        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output['loss'] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)
        
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """ Returns a dictionary of metrics associated with the model """
        return {'accuracy' : self.accuracy.get_metric(reset)}


if __name__ == "__main__":
    reader = PosDatasetReader()
    train_dataset = reader.read(cached_path(
        'https://raw.githubusercontent.com/allenai/allennlp'
        '/master/tutorials/tagger/training.txt'))
    validation_dataset = reader.read(cached_path(
        'https://raw.githubusercontent.com/allenai/allennlp'
        '/master/tutorials/tagger/validation.txt'))

    vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6

    model_params = Params({
        'type' : 'lstm',
        'input_size' : EMBEDDING_DIM,
        'hidden_size' : HIDDEN_DIM
    })

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=EMBEDDING_DIM)
    word_embedding = BasicTextFieldEmbedder({'tokens' : token_embedding})
    lstm = Seq2SeqEncoder.from_params(model_params)

    model = POSTagger(word_embedding, lstm, vocab)

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    iterator = BucketIterator(batch_size=2, sorting_keys=[('sentence', 'num_tokens')])
    iterator.index_with(vocab)

    trainer = Trainer(
        model = model,
        optimizer = optimizer,
        iterator = iterator,
        train_dataset = train_dataset,
        validation_dataset = validation_dataset,
        patience = 10,
        num_epochs = 1000
    )

    trainer.train()



