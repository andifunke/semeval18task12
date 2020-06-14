# SemEval 2018 — Task 12 Contribution

#### [The Argument Reasoning Comprehension Task](https://competitions.codalab.org/competitions/17327)

The Research Group for Databases and Information Systems of the
Heinrich-Heine-University Duesseldorf contributed to the 
[Argument Reasoning Comprehension Task](https://competitions.codalab.org/competitions/17327)
of the [SemEval 2018](http://alt.qcri.org/semeval2018/#) conference.

This repository contains code and data of this contribution as described in 

* **Matthias Liebeck, Andreas Funke, Stefan Conrad**:
  *HHU at SemEval-2018 Task 12: Analyzing an Ensemble-based Deep Learning 
  Approach for the Argument Mining Task of Choosing the Correct Warrant*  
  [Proceedings of The 12th International Workshop on Semantic Evaluation](https://www.aclweb.org/anthology/volumes/S18-1/)
  (SemEval 18)

  **Abstract:**  
  > This paper describes our participation in the SemEval-2018 Task 12 Argument Reasoning 
  > Comprehension Task which calls to develop systems that, given a reason and a claim, 
  > predict the correct warrant from two opposing options. We decided to use a deep learning 
  > architecture and combined 623 models with different hyperparameters into an ensemble. 
  > Our extensive analysis of our architecture and ensemble reveals that the decision to use 
  > an ensemble was suboptimal. Additionally, we benchmark a support vector machine as a baseline. 
  > Furthermore, we experimented with an alternative data split and achieved more stable results.
  > 
  > → https://www.aclweb.org/anthology/S18-1188/


Besides custom trained vectors we also used pre-trained vectors from the following sources:

- https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
- https://github.com/tca19/dict2vec
- https://nlp.stanford.edu/projects/glove/

### Hyperparameter Search

Due to the small size of the training set and the models itself, training a model was done fairly
quickly. This gave us the opportunity to perform a substantial hyperparameter search.
We performed a broad search first followed by a second, fine-grained search using multiple seeds.

The accuracies are documented here:

- [Broad Search](detailed_results_broad_search.md)
- [Fine-Grained Search](detailed_results_finegrained_search.md)
