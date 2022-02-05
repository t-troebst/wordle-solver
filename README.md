# Wordle Solver

This is a simple wordle solver that works by computing the word which maximizes the expected information gain (i.e. the expected reduction in entropy).
It uses an efficient representation of the word information, C++ standard parallelism, and some memoization to compute even the starting word very quickly.

## Compilation

The code requires C++20 including support for standard parallelism and ranges, it should compile on a recent version of g++ via:

    ./g++ -Ofast -ltbb -std=c++20 wordle_solver.cpp -o wordle_solver

## Usage

The general usage of the tool is as follows:

    ./wordle_solver wordle_guesses.txt wordle_words.txt [hard mode] [adversarial] [word_freqs.txt]

* `wordle_guesses.txt` supplies the list of allowed guesses for the tool.
* `wordle_words.txt` supplies the list of potential secret words.
* `[hard mode]` can be set to 0/1 and refers to the hard mode setting on wordle. If hard mode is activated, then the tool will only generate guesses that conform to previously received information.
* `[adversarial]` can be set to 0/1 and makes the tool assume that the correct word is chosen *adversarially* rather than *randomly*. In particular, this works for absurdle.
* `[word_freqs.txt]` supplies a list of words with associated frequencies that may be used as a tiebreaker. This is useful if the hidden words are not known.

The tool will then suggest the best possible choice and accept a response in the form of 5 letters: b for gray/black squares, g for green squares, and y for yellow squares.
