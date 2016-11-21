import random

def learn_curriculum(lessons, train, evaluate, random_lesson_ratio, level_up_streak, max_steps, eval_period):
  """Learn a curriculum of lessons ordered in order of increasing difficulty

  Implements the 'combined strategy' from https://arxiv.org/pdf/1410.4615v3.pdf

  A naive curriculum strategy is to increase the difficulty each time the
  algorithm has learned the previous lesson

  A mixed curriculum strategy is to randomly pick a lesson at each training
  step

  A combined strategy combines the naive strategy with the mixed strategy


  Args:
    lessons: A list of lessons. Each lesson is function taking zero arguments
      and returns a example for training: (inputs, targets)
    train: A function called to perform training. It takes two arguments.
      The first is the inputs and the second the targets. Returns a bool
      indicating whether the targets were predicted correctly from the inputs
    evaluate: A function called for evaluation. It takes four arguments. The
      first two are the same as the train function. The third is the number of
      steps that has been trained. The fourth is the lesson level of the example
    random_lesson_ratio: How often to train on an example from a random lesson
    level_up_streak: the number of correct predictions in a row to start the
      next lesson
    max_steps: Maximum number of training steps to perform
    eval_period: Steps between calls to evaluate
  """

  def lesson(level): return lessons[level]()

  current_level = 0
  step = 0
  streak = 0

  while step < max_steps and current_level < len(lessons):
    if current_level == 0 or random.random() > random_lesson_ratio:
      inputs, targets = lesson(current_level)
      correct = train(inputs, targets)
      if correct:
        streak +=1
        if streak >= level_up_streak:
          current_level += 1
          streak = 0
          print('Levelling up to %d at step %d' % (current_level, step))
      else:
        streak = 0
    else:
      # Train on a random lesson
      inputs, targets = lesson(random.randint(0, len(lessons)-1))
      train(inputs, targets)

    if step % eval_period == 0:
      inputs, targets = lesson(current_level)
      evaluate(inputs, targets, step, current_level)

    step += 1

  print('Training over')
  final_level = min(current_level, len(lessons)-1)
  inputs, targets = lesson(final_level)
  evaluate(inputs, targets, step, final_level)
