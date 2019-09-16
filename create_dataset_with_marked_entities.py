from preprocessing_prepeare_sentence import preprocessing
import json
import copy
import sys
bert_model = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-cased'
pre = preprocessing(bert_model)
for f in ["data/fewrel_val.json","data/fewrel_train.json"]:
    total_relatoin = {}
    data = json.load(open(f))
    for relation_type in data:
        this_realtion_list = []
        for x in data[relation_type]:
            tokens_with_markers, head_start_location, tail_start_location,head_end_location,tail_end_location = pre.preprocessing_flow(copy.deepcopy(x))
            assert head_start_location < head_end_location
            assert tail_start_location < tail_end_location
            assert type(head_end_location) is int
            assert type(tail_end_location) is int
            this_instance_dict = x
            this_instance_dict["head_after_bert"] = head_start_location
            this_instance_dict["tail_after_bert"] = tail_start_location
            this_instance_dict["tokens_with_markers"] = tokens_with_markers
            this_instance_dict["head_end"] = head_end_location
            this_instance_dict["tail_end"] = tail_end_location
            this_realtion_list.append(this_instance_dict)
        total_relatoin[relation_type] = this_realtion_list
    json.dump(total_relatoin,open(f+bert_model+"with_markers","w"))