footer: Julien Couillard, Gwendal Rio, Jean Thévenet 
slidenumbers: true

# Machine Learning
## Determining income 

---

- Table of Contents
    1. Dataset Description
    1. Data Quality Report and Analysis
    1. Data Handling
	1. Modelling
	1. Jazz
	1. Reflections


---

# Dataset Description

- Gathered during a survey between the years *1994-1995*. 
- Goal: find a correlation between *income and people's status*.
- Contains *199523 survey* responses from U.S. population.
- *41* demographic and employment related questions.

---

# Data Quality Report and Analysis

- *40* features 
- A lot of *missing informations*. 
- Some *duplicated features* (numerical indexes vs. words).
- Priority : *select the best features.*

---

# Data Handling

- 18 features ignored based on misses and duplicate rates.
- Used a default of 50%, 20%, 30% split after shuffling the rows.

---

# Modelling

---

# Jazz

- Everything designed using an easy to use interface & object-based code.
- Test also a model which always reports "true" (binary target).
- Display of each of the trees types accuracies using graphs.

---

# Reflections

- Misunderstanding between members of the group.
- plop

---

| Tree type             | Accuracy  | F1 Score  |
| --------------------- |:---------:| ---------:|
| Entropy Decision Tree | 93.04%    | 96.28%    |
| Gini Decision Tree    | 93.04%    | 96.29%    |
| Random Forest         | 95.07%    | 97.42%    |
| Naive Gaussian Tree   | 71.72%    | 83.34%    |

---

# Footnotes

Manage your footnotes[^1] directly where you need them. Alongside numbers, you can also use text references[^Sample Footnote].

Include footnotes by inserting`[^Your Footnote]` within the text. The accompanying reference can appear anywhere in the document:

`[^Your Footnote]: Full reference here`

[^1]: This is the first footnote reference

[^Sample Footnote]: This is the second footnote reference

---

# Footnotes

Footnote references need to be *unique in the markdown file*. This means, that you can also reference footnotes from any slide, no matter where they are defined.

When there are multiple references are listed, they must all be separated by blanks lines.

---


# Nested Lists

- You can create nested lists
    1. by indenting
    1. each item with 
    1. 4 spaces
- It’s that simple

---

# Links

Create links to any external resource—like [a website](http://www.decksetapp.com)—by wrapping link text in square brackets, followed immediately by a set of regular parentheses containing the URL where you want the link to point:

`‘[a website](http://www.decksetapp.com)’`

Your links will be clickable in exported PDFs as well! 

---

# Display formulas

Easily include mathematical formulas by enclosing TeX commands in `$$` delimiters. Deckset uses [MathJax](http://www.mathjax.org/) to translate TeX commands into beautiful vector graphics.

Formula support is available as in-app purchase. See the next slide for an example.

<a name="formulas"></a>

---

## Schrödinger equation

The simplest way to write the time-independent Schrödinger equation is $$H\psi = E\psi$$, however, with the Hamiltonian operator expanded it becomes:

$$
-\frac{\hbar^2}{2m} \frac{d^2 \psi}{dx^2} + V\psi = E\psi
$$

---

# Captioned Images and Videos

![inline](room.jpg)

Easily create captions using [inline] images/videos with text underneath.

---

# Plus: 

- PDF export for printed handouts
- Speaker notes and rehearsal mode
- Switch theme and ratio on the fly
- Animated GIFs for cheap wins and LOLs :-)
