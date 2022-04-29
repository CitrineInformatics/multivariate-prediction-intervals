# Multivariate Prediction Intervals for Random Forests

This repository contains the data and code used to generate the results in Multivariate Prediction Intervals for Random Forests (TODO: add link to ArXiv once available).
We propose a ''recalibrated bootstrap'' method to generate multivariate prediction intervals for predictions made by bagged models such as random forest.
We show that the resulting prediction intervals are well-calibrated on a variety of synthetic and real-world test problems.
We then apply the recalibrated bootstrap and other competing techniques to simulated sequential learning problems in which there are multiple competing objectives.
Due to its ability to capture correlation information between the outputs, the recalibrated bootstrap results in drastically more efficient sequential learning.
When compared to the naive method, the recalibrated bootstrap is 90% more efficient on a problem using synthetic data and 60% more efficient on a problem using real-world thermoelectrics data.

## Requirements

Model training and evaluation is done using Scala.
Evaluation and plotting of the resulting data is done using Python.

Compile the project and download dependencies using sbt (Scala build tool).
You must have Java 8 JDK already installed.
If stuck, see the [Scala reference manual](https://docs.scala-lang.org/getting-started/sbt-track/getting-started-with-scala-and-sbt-on-the-command-line.html).

```
brew install sbt
sbt clean compile
```

Install the packages in `requirements.txt` in order to run the analysis notebooks.

```
pip install -r requirements.txt
```

## Training and Evaluation

This repo includes the raw data resulting from all numerical experiments described in the manuscript.
If you would like to re-run any experiments, they are in the directory `io/citrine/loloExtension/benchmarks/` and can be run either from an IDE or from the command line.
For example, the following command would re-run simulated sequential learning on synthetic data:

```
sbt "runMain io.citrine.loloExtension.benchmarks.SequentialLearningFriedmanGrosse"
```

## Analysis

All results in this manuscript are derived in Jupyter notebooks found in the direcotry `manuscript-figures/`.
There are several ways to open a Jupyter notebook; one is to run the command `jupyter notebook` and then navigate to the desired notebook file.

## Results

Our main result is the increased efficiency of sequential learning.
The table below summarizes the results of Table 1 and Figure 4 in the manuscript.
Given two test problems and several methods of generating a multivariate prediction interval to select a trial candidate,
we see that using the recalibrated bootstrap leads to the fewest number of iterations required to find a candidate that satisfies all objectives.

| Correlation Method | Median Iterations (Synthetic Data)  | Median Iterations (Thermoelectrics Data) |
| ------------------ |------------------------- | ----------------------------- |
| Trivial            |     43                   |      309      |
| Random             |     33                   |      328      |
| Training Data      |     51                   |      201      |
| Jackknife          |     7                    |      203      |
| **Bootstrap**      |     **4.5**              |      **125**  |