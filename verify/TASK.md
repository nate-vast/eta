# Task: textstats

Build a small Python module `textstats.py` in this folder with these functions:

1. `word_count(text) -> int`: number of words (split on whitespace).
2. `top_words(text, n=3) -> list[tuple[str, int]]`: the n most common lowercase words, punctuation stripped, ties broken alphabetically.
3. `reading_time(text, wpm=200) -> int`: minutes to read, rounded up, minimum 1 for non-empty text, 0 for empty.
4. `summary_line(text) -> str`: "<words> words, ~<minutes> min read, top: <w1>, <w2>, <w3>".

Then:
5. Write `test_textstats.py` (unittest) covering every function, including empty text and ties.
6. Run the tests with `python3 -m unittest -v` and fix anything that fails.
7. Write a short README.md with usage examples.
