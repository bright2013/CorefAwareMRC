Data and code for the paper "[Tracing Origins: Coreference-aware Machine Reading Comprehension](https://www.semanticscholar.org/paper/EQG-RACE%3A-Examination-Type-Question-Generation-Jia-Zhou/f84b531135acc19191310537065a804c00814cdd)" at ACL2022.

## Dataset

There are three folders for our three models mentioned in the paper: Coref_additive_spacy, Coref_dgl_spacy and Coref_multiplication_spacy, and each contains the train data set and the dev data set under the **quoref** folder.

each sample contains
+ context: the paragraph text
+ context_id: the unique identifier of the context
+ qas: a group of questions
+ question: question text
+ id: the unique identifier of the question
+ answers: a group of the answers to one question
+ text: answer text
+ answer_start: the start_position of one answer

## Models
If you want to use our trained model, please download it from [Google drive](https://drive.google.com/drive/folders/1WDxyCRxDiOh5gcebYpInIdvHY9lCPCYx?usp=sharing)

## Training 
`python run_quoref.py --train_file "quoref/train.json" --predict_file "quoref/dev.json" --model_type "roberta_multi" --model_name_or_path "roberta-large" --output_dir "out" --do_train --do_eval --eval_all_checkpoints --learning_rate 1e-5 --num_train_epochs 6 --overwrite_output_dir --per_gpu_train_batch_size 4 --save_steps 6000 --coref_weight 0.4`

## Kindly Hint
There is an open issue regarding the compatibility between NeuralCoref and spaCy 3.0. If you intend to use the latest spaCy models, please watch the [issue](https://github.com/huggingface/neuralcoref/issues/295).

## Cite

If you extend or use this work, please cite the [paper](https://www.semanticscholar.org/paper/EQG-RACE%3A-Examination-Type-Question-Generation-Jia-Zhou/f84b531135acc19191310537065a804c00814cdd) where it was introduced:

```
@article{Huang2021TracingOC,
  title={Tracing Origins: Coref-aware Machine Reading Comprehension},
  author={Baorong Huang and Zhuosheng Zhang and Hai Zhao},
  journal={ArXiv},
  year={2021},
  volume={abs/2110.07961}
}
