package io.citrine.loloExtension.benchmarks

import io.citrine.lolo.bags.CorrelationMethods
import io.citrine.loloExtension.benchmarks.library.{GreaterThan, Objective}

object SequentialLearningFriedmanGrosse {

  def main(args: Array[String]): Unit = {
    val csvPath = "friedman_grosse_data.csv"
    val outputHeaders = Vector("y", "z")
    val driver = SequentialLearningDriver(csvPath, outputHeaders)

    val seed = 713259630L
    val filepathBase = "./sequential-learning-study/friedman-grosse-sl"
    val numTrials = 64
    val numInitialTraining = 16
    // The goal is to achieve y > 22 and z > 22. There are 2 points that meet these objectives.
    val objectives: Set[Objective] = Set(GreaterThan(outputHeaders(0), 0, 22.0), GreaterThan(outputHeaders(1), 1, 22.0))

    driver.runTrials(
      filepath = s"${filepathBase}-trivial.csv",
      numTrials = numTrials,
      objectives = objectives,
      numInitialTraining = numInitialTraining,
      method = CorrelationMethods.Trivial,
      seed = seed
    )

    driver.runTrials(
      filepath = s"${filepathBase}-training-data.csv",
      numTrials = numTrials,
      objectives = objectives,
      numInitialTraining = numInitialTraining,
      method = CorrelationMethods.FromTraining,
      seed = seed
    )

    driver.runTrials(
      filepath = s"${filepathBase}-bootstrap.csv",
      numTrials = numTrials,
      objectives = objectives,
      numInitialTraining = numInitialTraining,
      method = CorrelationMethods.Bootstrap,
      seed = seed
    )

    driver.runTrials(
      filepath = s"${filepathBase}-jackknife.csv",
      numTrials = numTrials,
      objectives = objectives,
      numInitialTraining = numInitialTraining,
      method = CorrelationMethods.Jackknife,
      seed = seed
    )
  }

}
