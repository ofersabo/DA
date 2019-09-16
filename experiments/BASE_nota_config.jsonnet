local cuda = [0,1,2,3];
//local cuda = [3];
local bert_type = 'bert-base-cased';
//local bert_type = 'bert-large-cased';
local batch_size = 10;
local full_training = true;
local max_query_training = true;
local lr_with_find = 0.00001;
local instances_per_epoch = null;

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
  "train_data_path": if max_query_training then "data/max_query_picking/BERT_BASE_train_NOTA_max_query_80K.json" else if full_training then "data/BASE_TRAIN_NOTA_100K.json" else "data/BASE_TRAIN_NOTA_100.json",
  "validation_data_path": if full_training then "data/BASE_VAL_NOTA_5K.json" else "data/BASE_VAL_NOTA_100.json",
  "model": {
    "type": "nota_bert",
    "bert_model": bert_type,
    "number_of_linear_layers": 2,
    "drop_out_rate": 0.4,
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
//    "regularizer": [["liner_layer", {"type": "l2", "alpha": 1e-03}], [".*", {"type": "l2", "alpha": 1e-07}]]
    "regularizer": [[".*no_relation.*", {"type": "l2", "alpha": 1e-03}],["liner_layer", {"type": "l2", "alpha": 1e-03}], [".*", {"type": "l2", "alpha": 1e-07}]]

  },
  "iterator": {
    "type": "basic",
    "batch_size": batch_size,
    "instances_per_epoch": instances_per_epoch
  },
    "validation_iterator": {
    "type": "basic",
    "batch_size": batch_size,
    "instances_per_epoch": if full_training then instances_per_epoch else null
  },
  "trainer": {
//  "regularizers": "l2",
    "optimizer": {
        "type": "adam",
        "lr": lr_with_find
    },
    "num_serialized_models_to_keep": 1,
    "validation_metric": "+accuracy",
    "num_epochs": 25,
    "grad_norm": 2.0,
    "patience": 12,
    "cuda_device": cuda
  }
}