from typing import Dict
from typing import List
import json
import logging
import os

from overrides import overrides
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import ListField, IndexField, MetadataField, Field



head_start_token = '[unused1]'  # fixme check if this indeed the token they used in the paper
head_end_token = '[unused2]'  # fixme check if this indeed the token they used in the paper
tail_start_token = '[unused3]'
tail_end_token = '[unused4]'

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def create_file(file_path):
    import subprocess
    preffix = "data/"
    data = "fewrel_val_markers.json" if "dev" in file_path or "val" in file_path else "fewrel_train_markers.json"
    size = file_path[:file_path.index(".")].split("_")[-1]
    if not size[-1].isdigit():
        unit = size[-1]
        unit = 1000 if unit == "K" else 1000**2
        size = int(size[:-1]) * unit
    size = str(size)
    seed = str(13)
    N = "5"
    K = "1"
    NOTA_RATE = "50"
    # os.system("python data/division/NOTA_random_division.py "+ preffix + data+ " " + size +" "+
    # N + " " + K + " " +NOTA_RATE + " " + seed + " "+  file_path)
    x = subprocess.Popen(["python","data/division/NOTA_random_division.py",preffix,data,size,N,K,NOTA_RATE,seed,file_path])
    x.wait()

@DatasetReader.register("locations_NOTA_reader")
class MTBDatasetReader(DatasetReader):
    def __init__(self,
                 bert_model: str,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.spacy_splitter = SpacyWordSplitter(keep_spacy_tokens=True)
        self.TRAIN_DATA = "meta_train"
        self.TEST_DATA = "meta_test"
        self.tokens_with_markers = "tokens_with_markers"
        self.head_bert = "head_after_bert"
        self.end_head = "head_end"
        self.end_tail = "tail_end"
        self.tail_bert = "tail_after_bert"


    @overrides
    def _read(self, file_path):
        if not os.path.exists(file_path):
            create_file(file_path)
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from json files at: %s", data_file)
            data = json.load(data_file)
            labels = data[1]
            data = data[0]
            for x, l in zip(data, labels):
                yield self.text_to_instance(x, l)

    @overrides
    def text_to_instance(self, data: dict, relation_type: int = None) -> Instance:  # type: ignore
        N_relations = []
        location_list = []
        all_tokens_sentences = []
        for i, K_examples in enumerate(data[self.TRAIN_DATA]):
            toknized_sentences = []
            head_markers_locations = []
            head_entity_list = []
            tail_entity_list = []
            tail_markers_locations = []
            clean_text_for_debug = []
            for single_relation in K_examples:
                list_of_string = " ".join(single_relation[self.tokens_with_markers])
                tokenized_tokens = self._tokenizer.tokenize(list_of_string)
                field_of_tokens = TextField(tokenized_tokens, self._token_indexers)

                clean_text_for_debug.append(MetadataField(list_of_string))

                head_after_bert_location, tail_after_bert_location = single_relation[self.head_bert], single_relation[self.tail_bert]
                head_end_after_bert,tail_end_after_bert = single_relation[self.end_head], single_relation[self.end_tail]
                head_markers_location = MetadataField([head_after_bert_location, head_end_after_bert])
                tail_markers_location = MetadataField([tail_after_bert_location, tail_end_after_bert])
                head_entity = MetadataField([iii for iii in range(head_after_bert_location+1,head_end_after_bert)])
                tail_entity = MetadataField([iii for iii in range(tail_after_bert_location+1,tail_end_after_bert)])

                head_markers_locations.append(head_markers_location)
                tail_markers_locations.append(tail_markers_location)
                head_entity_list.append(tail_markers_location)
                toknized_sentences.append(field_of_tokens)
            assert len(head_markers_locations) == len(toknized_sentences) == len(clean_text_for_debug)

            head_markers_locations, clean_text_for_debug, toknized_sentences = ListField(head_markers_locations), ListField(clean_text_for_debug), ListField(toknized_sentences)
            all_tokens_sentences.append(clean_text_for_debug),location_list.append(head_markers_locations), N_relations.append(toknized_sentences)

        assert len(N_relations) == len(location_list) == len(all_tokens_sentences)
        N_relations, location_list, all_tokens_sentences = ListField(N_relations), ListField(location_list),ListField(all_tokens_sentences)
        fields = {'sentences': N_relations, "locations": location_list, "clean_tokens": all_tokens_sentences}

        '''
        query_part
        '''
        query = data[self.TEST_DATA]
        list_of_string = " ".join(query[self.tokens_with_markers])
        tokenized_tokens = self._tokenizer.tokenize(list_of_string)
        field_of_tokens = TextField(tokenized_tokens, self._token_indexers)

        test_clean_text_for_debug = MetadataField(list_of_string)
        head_after_bert_location, tail_after_bert_location = query[self.head_bert], query[self.tail_bert]
        head_markers_location = MetadataField([head_after_bert_location, tail_after_bert_location])

        fields['test'] = field_of_tokens
        fields['test_location'] = head_markers_location
        fields['test_clean_text'] = test_clean_text_for_debug

        if relation_type is not None:
            fields['label'] = IndexField(relation_type, N_relations)
        return Instance(fields)
