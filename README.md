# generating sequences

Exploring some different ways of generating sequences with recurrent neural
networks.

## Plan:
- *next step prediction* approaches
  - Starting [with the classic](https://arxiv.org/pdf/1308.0850.pdf).
  - this [variational approach](https://arxiv.org/abs/1506.02216), or something similar (depending whether I can figure out what is actually
  going on with the prior).
  - some more [classic variational approaches](http://papers.nips.cc/paper/4329-practical-variational-inference-for-neural-networks.pdf) to help avoid overfitting (which can be in general applied to anything, might be handy training the embeddings below).
  - probably some crazy hybrids of the below approaches, if there's a sensible seeming way to apply them to the running states.
- *seq2seq* approaches -- probably with data consisting of many short sequences, because otherwise it's a pain. The simplest way to do this is to first train a model to embed the sequences as per [Dai and Le](https://arxiv.org/abs/1511.01432), then try and model the distribution of the states that get passed between the encoder and the decoder.
  - use a [VAE on the states in the middle](https://arxiv.org/abs/1412.6581).
  - use a [GAN](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) on the states in the middle (tried this already, GANs are a hassle to train).
  - use an [Adversarial Autoencoder](https://arxiv.org/abs/1511.05644) on the states in the middle (find a better term for these states as well -- I'm sure Hinton had one).
  - anything related (such as [Generative Moment Matching Networks](https://arxiv.org/abs/1502.02761)) which seems promising, but only if this angle shows promise.
- anything else
  - there are a lot of ways to extend the sequence to sequence approaches to work more naturally on sequences, eg. by using a recurrent discriminator.
  - it's also possible to use policy gradients to backpropagate around the sampling and train a next-step prediction in a generative-adversarial fashion, but in my experience it doesn't work very well as policy gradients tends to take a lot of updates. Apparently keeping it on track with regular next step training can help, but that seems tacky.
- evaluation:
  - how do you know if something is generating good sequences?
  - [how do you know if something is generating good anything?](https://arxiv.org/abs/1511.01844)
  - for next step prediction we can check our models by the log-likelihood they assign to test/validation data
  - for the less orthodox models it's more challenging to evaluate
  - plus, we really care about making interesting samples, and there's no clear way to evaluate that.
