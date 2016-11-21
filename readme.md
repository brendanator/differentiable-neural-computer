# Differentiable Neural Computer

An implementation of DeepMind's [Differentiable Neural Computer](http://www.nature.com/nature/journal/v538/n7626/pdf/nature20101.pdf) as described in Nature

## Try it out

1. Install tensorflow
2. Run `python tasks/copy_sequence.py`

This will train a Differentiable Neural Computer with a feedforward controller to learn to copy sequences. It uses curriculum learning with increasing length sequences as the lessons
