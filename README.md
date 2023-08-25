# PL Similarity
This is the replication package for the paper accepted at 2023 IEEE 23rd International Working Conference on Source Code Analysis and Manipulation (SCAM):

[**On the Impact of Language Selection for Trainingand Evaluating Programming Language Models.**](URL)  
By: Jonathan Katzy, Maliheh Izadi, Arie van Deursen 

## Setup
To calculate the similarities we need to first run the setup. In the setup we need to train the models, calculate the common tokens, and run inference for the selected tokens.

### Training
We train the models according to the CodeBERTScore methodology.
The code is given in [Setup/train\_pretrain.py](/Setup/train_pretrain.py) for pretrained models and in [Setup/train\_no\_pretrain.py](/Setup/train_no_pretrain.py) to train the models from scratch. This uses the [default script from huggingface](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py), with the trainsteps hardcoded to 100,000, and reseting the parameters when training from scratch.

The code can be run from the command line as follows:
```bash
python train_pretrain.py \
    --model_name_or_path microsoft/codebert-base-mlm \
    --train_file java_train \
    --per_device_train_batch_size 8 \
    --do_train \
    --output_dir /outputs/java_pretrain
```

### Common Tokens
After training the models, also run the [Setup/tokens.py](/Setup/tokens.py) file, in order to calculate the common tokens, the results will be saved to a file.

### Inference
Finally the representations of the tokens can be calculated using the [Setup/eval.py](/Setup/eval.py) code.

## Similarity
To calculate the similarities, first run all the scripts form the Setup section in order to generate the required files (all intermediate outputs are saved to file), then run the functions from [Similarity/similarity.py](/Setup/similarity.py) to calculate the requiered similarities.

## Visualization
The visualizations for the paper are created using the scripts form the Visualization folder.

# Citation
If you use this paper please cite as:
```
@inproceedings{katzy2023impact,
  title={On the Impact of Language Selection for Trainingand Evaluating Programming Language Models},
  author={Katzy, Jonathan and Izadi, Maliheh and van Deursen, Arie},
  booktitle={2023 IEEE 23rd International Working Conference on Source Code Analysis and Manipulation (SCAM)},
  pages={},
  year={2023},
  organization={IEEE}
}
```