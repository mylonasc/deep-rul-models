# DenseNet with Dilated Convolutions
Implementation and applications of DenseNet with dilated convolutional blocks + parameterization with adversarial training.

## Datasets
### FEMTO Bearing fatigue dataset
This was a dataset used in a conference challenge (PHM conference). Bearings were loaded to failure from pristine conditions in an accelerated fatigue testing 
machine with loading conditions of varying applied force and speed. An unballanced dataset of 17 samples are available for 3 different loading conditions:

|condition |set|experiment indices|total time to failure|exp_strings|
|-- | -- | -- | -- |--|
|1| training | (4, 6, 8, 15)| (8700, 23010, 11380, 28072) |... |
|2|training|(9, 10, 0,7)|(20010,5710,1710,12010)|...|
|3|training|(2,11)|(16360,3510) | ...|
|1|validation| (1, 5, 16) | (15010, 18010, 23010) | ...|
|2|validation|(12,13,14)|(7960,9100,6110)|...|
|3|validation|(3) | (5140)|...|


## Adversarial techniques for better generalization
Adversarial techniques are interesting due to the (almost) likelihood-free nature of the training procedure.
The main ideas available for improving generalization, employing adversarial losses for deep learning techniques are the following:

* Conditional GAN generation in target domain (when having limited target labels but abundant source labels)
* Domain adversarial training (*GradReverse technique*) \[1\]
* Using a reconstruction loss as in GANs but with the purpose of learning in an unsupervised manner domain-shifts 

In the context of the bearing diagnosis dataset (or the anchors dataset) the following challenges arise which are not covered by the available techniques in the literature:
* Reconstruction is hard and difficult to judge its quality, therefore the domain separation idea cannot be straight-forwardly applied.
* The signal itself is stochastic.
* The progression of damage is important for prediction 
  * The progression of damage is different for each experiment
  * The underlying causes for the progression of damage are different for each experiment
  * IID assumption implicit in training with batches not satisfied!

This is about a technique to deal with the aformentioned difficulties. The proposed technique is termed *Adversarial discriminative Domain Separation*.
This is a combination of the domain separation idea (implicit analysis on shared/private features) of \[2\] and the shift of focus from *generation* to *discrimination* 
as the objective driving learning as suggested in \[1\] and \[3\]. 

## References
* [\[1\]Domain adversarial training of neural networks](https://arxiv.org/abs/1505.07818)
* [\[2\] Domain separation networks](http://papers.nips.cc/paper/6254-domain-separation-networks.pdf) 
* [\[3\]Adversarial discriminative domain adaptation](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.pdf)
