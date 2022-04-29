package io.citrine.loloExtension.benchmarks

import io.citrine.lolo.bags.CorrelationMethods
import io.citrine.loloExtension.benchmarks.library.{GreaterThan, Objective}

object SequentialLearningThermoelectrics {

  def main(args: Array[String]): Unit = {
    val outputHeaders = Vector(
      "ZT",
      "Seebeck coefficient (uV/K)",
      "log Resistivity",
      "Power factor (W*m/K^2)",
      "Thermal conductivity (W/(m*K))"
    )
    val seed = 19121119L
    val numTrials = 64
    val numInitialTraining = 32

    // ZT > 1.25, Seebeck > 175, power factor > 5e-3, thermal conductivity > 1.5
    // Because the multivariate normal distribution has problems with singular matrices when the variances
    // vary over orders of magnitude, we rescale the outputs to all be of order ~10.
    // ZT is multiplied by 1e1, Seebeck coefficient by 1e-2, and power factor by 1e4.
    // Only one training row satisfies these constraints. It has index 339
    val csvPathRescaled = "thermoelectrics_clean_rescaled.csv"
    val driverRescaled = SequentialLearningDriver(csvPathRescaled, outputHeaders)
    val filepathBase4 = "./sequential-learning-study/thermoelectrics/4-objectives"
    val objectives4: Set[Objective] = Set(
      GreaterThan(outputHeaders(0), 0, 12.5),
      GreaterThan(outputHeaders(1), 1, 1.75),
      GreaterThan(outputHeaders(3), 3, 50),
      GreaterThan(outputHeaders(4), 4, 1.5)
    )

    driverRescaled.runTrials(
      filepath = s"${filepathBase4}-trivial.csv",
      numTrials = numTrials,
      objectives = objectives4,
      numInitialTraining = numInitialTraining,
      method = CorrelationMethods.Trivial,
      seed = seed
    )

    driverRescaled.runTrials(
      filepath = s"${filepathBase4}-training-data.csv",
      numTrials = numTrials,
      objectives = objectives4,
      numInitialTraining = numInitialTraining,
      method = CorrelationMethods.FromTraining,
      seed = seed
    )

    driverRescaled.runTrials(
      filepath = s"${filepathBase4}-bootstrap.csv",
      numTrials = numTrials,
      objectives = objectives4,
      numInitialTraining = numInitialTraining,
      method = CorrelationMethods.Bootstrap,
      seed = seed
    )

    driverRescaled.runTrials(
      filepath = s"${filepathBase4}-jackknife.csv",
      numTrials = numTrials,
      objectives = objectives4,
      numInitialTraining = numInitialTraining,
      method = CorrelationMethods.Jackknife,
      seed = seed
    )

  }
}
