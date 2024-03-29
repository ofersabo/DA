from overrides import overrides
import numpy as np
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('mtb-submit-predictor')
class MTBClassifierPredictor(Predictor):
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        # mapping_set_index_to_realtion_type,gold_pred = self.extract_mapping_and_correct_answer(inputs)
        output_dict = {}

        instance = self._json_to_instance(inputs)
        scores = self.predict_instance(instance)['scores']
        prediction = np.argmax(scores)
        output_dict[""] = prediction

        return output_dict

    def extract_mapping_and_correct_answer(self, inputs):
        this_set_mapping = []
        for i,examples in enumerate(inputs[self._dataset_reader.TRAIN_DATA]):
            this_set_mapping.append(examples[0])
            inputs[self._dataset_reader.TRAIN_DATA][i] = examples[1]
        x = inputs[self._dataset_reader.TEST_DATA]
        correct = x[0]
        inputs[self._dataset_reader.TEST_DATA] = x[1]
        return this_set_mapping,correct


    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        #def text_to_instance(self, data: dict, relation_type: int = None) -> Instance:
        this_instance = self._dataset_reader.text_to_instance(data=json_dict)
        return this_instance
