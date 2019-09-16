import json
import shutil
import sys

from allennlp.commands import main

model_location = "result_linear_parameter/"
output_file = "predictor_result.jsonl"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!

'''
sage: allennlp predict [-h] [--output-file OUTPUT_FILE]
                            [--weights-file WEIGHTS_FILE]
                            [--batch-size BATCH_SIZE] [--silent]
                            [--cuda-device CUDA_DEVICE]
                            [--use-dataset-reader]
                            [--dataset-reader-choice {train,validation}]
                            [-o OVERRIDES] [--predictor PREDICTOR]
                            [--include-package INCLUDE_PACKAGE]
                            archive_file input_file
'''

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "predict",
    model_location,
    "my_library/predictors/NOTA2predictor.jsonl",
    "--output-file", model_location + output_file,
    "--include-package", "my_library",
    "--predictor", "nota-predictor"
    # "-o", overrides,
]

main()