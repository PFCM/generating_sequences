# scripts

This directory contains things to run.

Of note:
- _train.py_ -- trains a model. Hopefully this will become pretty modular and flexible, we will have to see. It accepts the following arguments:
  - `logdir` (string) where to store checkpoints/summaries. If absent,
    the default is something sensible based on the rest of the options
    and the current date/time.
  - `cell` (string) the RNN cell to use. This is likely to become pretty
    flexible, at the moment the following options are supported:
    - `vanilla`: classic RNN with `tanh` nonlinearity.
    - `lstm`: long short term memory, standard (no peepholes) with forget
      gate bias set to 1.0.
    - `gru` (default): gated recurrent unit, a sensible default they tend
      do pretty well.
  - `model` (string) how we are structuring the RNN. Hopefully there'll be
    a lot of these, this is the main event. Options are:
      - `nextstep/standard`: the classic approach, mostly due to Graves.
        In the learning phase we train it to predict the next character
        given the current sequence so far. To sample we start go one
        step at a time generating an output based on the outputs
        generated thus far.
  - `width` (int) the number of hidden units per RNN cell.
  - `layers` (int) the number of RNN cells stacked up.
  - `keep_prob` (float) the amount of dropout applied to inputs.
