## Tensorflow Implementation of the (Dual)-Associative Memory GRUs

If you make use of these implementations please cite the following paper:

**Neural Associative Memory for Dual-Sequence Modeling**. Dirk Weissenborn. [*arXiv:1606.03864*](http://arxiv.org/abs/1606.03864). (*to appear in RepL4NLP@ACL2016*).

* requires TensorFlow 0.9
* contains RNNCell implementations for:
  * AssociativeGRU: Associative-Memory GRU
  * DualAssociativeGRU: Dual-Associative Memory
  * SelfControllerWrapper: Self-controller wrapper that is useful in combination with (Dual)-Associative Memory GRUs 
  * ControllerWrapper: Controller wrapper that is useful in combination with (Dual)-Associative Memory GRUs or AttentionCell
  * AttentionCell: A RNNCell that can be used in combination with ControllerWrapper to realize attention
   
