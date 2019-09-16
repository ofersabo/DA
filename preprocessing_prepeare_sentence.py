from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from my_library.dataset_readers.mtb_reader import head_start_token,head_end_token,tail_start_token,tail_end_token
from my_library.models.my_bert_tokenizer import MyBertWordSplitter
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter, SimpleWordSplitter
def find_closest_distance_between_entities(head_start_location, head_end_location, tail_start_location,
                                           tail_end_location):
    min_distance = 99999
    for i, x in enumerate(head_start_location):
        for j, y in enumerate(tail_start_location):
            if abs(x - y) < min_distance:
                min_distance = abs(x - y)
                h_start, h_end, t_start, t_end = x, head_end_location[i], y, tail_end_location[j]

    return h_start, h_end, t_start, t_end


def encode_by_bert(sent ,tokenizer):
    my_sen = tokenizer.convert_tokens_to_string(sent)
    input_ids = tokenizer.encode(text=my_sen, add_special_tokens=True)
    return input_ids


class preprocessing(object):
    def __init__(self,bert_model):
        lower_case = True if "uncased" in bert_model else False
        self.bert_indexer,self.tokenizer = self.get_bert_indexer(bert_model,lower_case=lower_case)
        self.tokenizer_bert = MyBertWordSplitter(do_lower_case=lower_case)
        self.spacy_splitter = SpacyWordSplitter(keep_spacy_tokens=True)
        self.just_space_tokenization = JustSpacesWordSplitter()
        self.simple_tokenization = SimpleWordSplitter()

    def preprocessing_flow(self,relation):
        sentence, head, tail = relation["tokens"], relation["h"], relation["t"]
        if len(set([head_start_token,head_end_token,tail_start_token,tail_end_token]).intersection(set(sentence))) != 4:
            self.addStartEntityTokens(sentence, head, tail)
        bert_indexing = self.bert_indexer(sentence)
        head_location_after_bert,tail_location_after_bert,head_end,tail_end = self.get_h_t_index_after_bert(bert_indexing)
        return sentence,head_location_after_bert,tail_location_after_bert,head_end,tail_end

    def addStartEntityTokens(self, tokens_list, head_full_data, tail_full_data):
        if len(head_full_data[0]) > len(tail_full_data[0]):  # this is for handling nested tail and head entities
            # for example: head = NEC and tail = NEC corp
            # solution, make sure no overlapping entities mention
            head_start_location, head_end_location = self.find_locations(head_full_data, tokens_list)
            tail_start_location, tail_end_location = self.find_locations(tail_full_data, tokens_list)
            if tail_start_location[0] >= head_start_location[0] and tail_start_location[0] <= head_end_location[0]:
                tail_end_location, tail_start_location = self.deny_overlapping(tokens_list, head_end_location,
                                                                               tail_full_data)

        else:
            tail_start_location, tail_end_location = self.find_locations(tail_full_data, tokens_list)
            head_start_location, head_end_location = self.find_locations(head_full_data, tokens_list)
            if head_start_location[0] >= tail_start_location[0] and head_start_location[0] <= tail_end_location[0]:
                head_end_location, head_start_location = self.deny_overlapping(tokens_list, tail_end_location,
                                                                               head_full_data)

        # todo try different approchs on which entity location to choose
        h_start_location, head_end_location, tail_start_location, tail_end_location = find_closest_distance_between_entities \
            (head_start_location, head_end_location, tail_start_location, tail_end_location)

        # x = self._tokenizer.tokenize(head_start_token)
        # y = self._tokenizer.tokenize(head_end_token)
        # z = self._tokenizer.tokenize(tail_start_token)
        # w = self._tokenizer.tokenize(tail_end_token)

        offset_tail = 2 * (tail_start_location > h_start_location)
        tokens_list.insert(h_start_location, head_start_token)  # arbetrary pick a token for that
        tokens_list.insert(head_end_location + 1 + 1, head_end_token)  # arbetrary pick a token for that
        tokens_list.insert(tail_start_location + offset_tail, tail_start_token)  # arbetrary pick a token for that
        tokens_list.insert(tail_end_location + 2 + offset_tail, tail_end_token)  # arbetrary pick a token for that

        return h_start_location + 2 - offset_tail, tail_start_location + offset_tail


    def deny_overlapping(self, tokens_list, longest_entity_end_location, shortest_entity_full_data):
        start_location, end_location = self.find_locations(shortest_entity_full_data,
                                                           tokens_list[longest_entity_end_location[0] + 1:])
        start_location[0] = start_location[0] + longest_entity_end_location[0]
        end_location[0] = end_location[0] + longest_entity_end_location[0]
        return end_location, start_location


    def return_lower_text_from_tokens(self, tokens):
        try:
            return list(map(lambda x: x.text.lower(), tokens))
        except:
            return list(map(lambda x: x.lower(), tokens))


    def compare_two_token_lists(self, x, y):
        return self.return_lower_text_from_tokens(x) == self.return_lower_text_from_tokens(y)


    def spacy_work_toknizer(self, text):
        return list(map(lambda x: x.text, self.spacy_splitter.split_words(text)))


    def find_locations(self, head_full_data, token_list):
        if len(head_full_data[2]) == 1:
            end_location, start_location = [head_full_data[2][0][-1]], [head_full_data[2][0][0]]
            return start_location, end_location
        end_location, start_location = self._find_entity_name(token_list, head_full_data)

        assert len(start_location) == len(end_location)
        if not len(start_location) == len(head_full_data[2]):
            print(head_full_data)
            print(token_list)
            exit()

        return start_location, end_location

    def my_toeknization(self,my_str):
        if ".-" in my_str:
            my_str = my_str.split()
            middle = my_str[1].split(".-")
            return my_str[:1] + middle[:1] + [".-"] + middle[1:] + my_str[2:]
        first_word = my_str[my_str.find('"')+1:my_str.rfind('"')]
        second_part = my_str[my_str.find('-')+1:].split()
        return ['"' , first_word ,'"-'] + second_part



    def _find_entity_name(self, token_list, head_full_data):
        for function_tokenizer in [self.tokenizer_bert.split_words,self.spacy_work_toknizer,
                                   self.just_space_tokenization.split_words,self.simple_tokenization.split_words,
                                   self.my_toeknization]:
            head = function_tokenizer(head_full_data[0])

            start_head_entity_name = head[0]
            start_location = []
            end_location = []
            for i, token in enumerate(token_list):
                if self.compare_two_token_lists([token], [start_head_entity_name]):
                    if self.compare_two_token_lists(token_list[i:i + len(head)], head):
                        start_location.append(i)
                        end_location.append(i + len(head) - 1)
                        if len(start_location) == len(head_full_data[2]):
                            break

            if len(end_location) > 0 and len(start_location) > 0:
                break

        return end_location, start_location

    def get_bert_indexer(self, bert_model_this_model,lower_case):
        from pytorch_transformers import BertModel, BertTokenizer
        my_special = {'additional_special_tokens': [head_start_token, head_end_token, tail_start_token, tail_end_token]}
        model_class, tokenizer_class, pretrained_weights = BertModel, BertTokenizer, bert_model_this_model
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights,do_lower_case=lower_case,
                                                    never_split=[head_start_token, head_end_token, tail_start_token, tail_end_token])
        tokenizer.add_special_tokens(my_special)
        return lambda x: encode_by_bert(x,tokenizer),tokenizer

    def get_h_t_index_after_bert(self, bert_indexing):
        # fixme convert to arguments
        return bert_indexing.index(1), bert_indexing.index(3) , bert_indexing.index(2), bert_indexing.index(4)

    def convert_tokens_to_string_list(self, tokenized_tokens):
        return list(map(lambda x: x[0], tokenized_tokens))

    def add_special_tokens_to_token_list(self, relation, tokenized_tokens):
        head_location, tail_location = self.addStartEntityTokens(tokenized_tokens, relation['h'], relation['t'])
        assert tokenized_tokens[head_location].text == head_start_token
        assert tokenized_tokens[tail_location].text == tail_start_token



