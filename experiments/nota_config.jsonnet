//local cuda = [0,1,2,3];
local cuda = [0,1,2,3,4,5,6,7];
//local cuda = [3];
//local bert_type = 'bert-base-cased';
local bert_type = 'bert-large-cased';
local batch_size = 10;
local full_training = true;
local max_query_training = false;
local small_dataset = false;
local lr_with_find = 0.00001;
//local instances_per_epoch = null;
local instances_per_epoch = 100000;

{
  "dataset_reader": {
    "type": "NOTA_reader",
    "bert_model": bert_type,
    "lazy": false,
    "tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "my-bert-basic-tokenizer",
        "do_lower_case": false
      }
    },
    "token_indexers": {
      "bert": {
          "type": "bert-pretrained",
          "pretrained_model": bert_type,
          "do_lowercase": false,
          "use_starting_offsets": false
      }
    }
  },
  "train_data_path":  if small_dataset then "data/train_NOTA_100.json" else if max_query_training then "data/max_query_picking/train_NOTA_2_max_query_100K.json"
   else if full_training then "data/train_NOTA_1M.json" else "data/train_NOTA_50K.json",
  "validation_data_path": if small_dataset then "data/dev_NOTA_100.json" else if full_training then "data/dev_NOTA_10K.json" else "data/dev_NOTA_5K.json",
  "model": {
    "type": "nota_with_cls",
    "bert_model": bert_type,
    "number_of_linear_layers": 2,
    "drop_out_rate": 0.2,
    "skip_connection": true,
    "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {

            "bert": ["bert"]
        },
        "token_embedders": {
            "bert": {
              "type": "bert-pretrained",
              "pretrained_model":  bert_type,
              "top_layer_only": true,
              "requires_grad": true
            }
        }
    },
    "regularizer": [[".*no_relation.*", {"type": "l2", "alpha": 1e-03}],["liner_layer", {"type": "l2", "alpha": 1e-03}], [".*", {"type": "l2", "alpha": 1e-07}]]

  },
  "iterator": {
    "type": "basic",
    "batch_size": batch_size,
    "instances_per_epoch": if full_training && small_dataset == false then instances_per_epoch else null
  },
    "validation_iterator": {
    "type": "basic",
    "batch_size": batch_size,
    "instances_per_epoch": null
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": lr_with_find
    },
    "num_serialized_models_to_keep": 1,
    "validation_metric": "+accuracy",
    "num_epochs": 25,
//    "grad_norm": 1.0,
    "patience": 12,
    "cuda_device": cuda
  }
}