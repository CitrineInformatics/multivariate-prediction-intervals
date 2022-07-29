package io.citrine.loloExtension.benchmarks

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import io.citrine.lolo.PredictionResult
import io.citrine.lolo.bags.MultiTaskBagger
import io.citrine.lolo.transformers.MultiTaskStandardizer
import io.citrine.lolo.trees.multitask.MultiTaskTreeLearner
import io.citrine.lolo.validation.Merit
import io.citrine.loloExtension.benchmarks.BenchmarkUtils.extractInputsOutputsFromCSV
import io.citrine.loloExtension.benchmarks.library.{Metric, NLPD, NumTraining, StdConfidence, VariedParameter}
import org.apache.commons.math3.random.MersenneTwister

import scala.util.Random

object CorrelationAccuracyRealData {

  def main(args: Array[String]): Unit = {
    runThermoelectrics()
    runMechanicalProperties()
  }

  private def runThermoelectrics(): Unit = {
    val seed = 810561772L

    val csvPath = "thermoelectrics_clean.csv"
    val numTest = 64
    val numTrainSeq: Seq[Int] = Seq(64, 128, 256, 512)
    val numBags = Some(64)
    val numTrials = 32

    val outputHeaders = Vector(
      "ZT",
      "Seebeck coefficient (uV/K)",
      "log Resistivity",
      "Power factor (W*m/K^2)",
      "Thermal conductivity (W/(m*K))"
    )

    CorrelationAccuracyRealData(
      csvPath = csvPath,
      outputHeaders = outputHeaders,
      metric = NLPD(Seq(0, 1, 3, 4), observational = true),
      numTest = numTest,
      numTrainSeq = numTrainSeq,
      numBags = numBags,
      numTrials = numTrials
    ).runTrialsAndSave(s"./correlation-study-n-dims/thermoelectrics-nlpd", new Random(seed))

    CorrelationAccuracyRealData(
      csvPath = csvPath,
      outputHeaders = outputHeaders,
      metric = StdConfidence(Seq(0, 1, 3, 4), observational = true),
      numTest = numTest,
      numTrainSeq = numTrainSeq,
      numBags = numBags,
      numTrials = numTrials,
    ).runTrialsAndSave(s"./correlation-study-n-dims/thermoelectrics-stdcon", new Random(seed))
  }

  private def runMechanicalProperties(): Unit = {
    val seed = 810561772L

    val csvPath = "mechanical_properties_clean.csv"
    val numTest = 48
    val numTrainSeq: Seq[Int] = Seq(32, 64, 96, 128)
    val numBags = Some(64)
    val numTrials = 32

    val outputHeaders = Vector(
      "PROPERTY: YS (MPa)",
      "PROPERTY: Elongation (%)"
    )

    CorrelationAccuracyRealData(
      csvPath = csvPath,
      outputHeaders = outputHeaders,
      metric = NLPD(Seq(0, 1), observational = true),
      numTest = numTest,
      numTrainSeq = numTrainSeq,
      numBags = numBags,
      numTrials = numTrials
    ).runTrialsAndSave(s"./correlation-study-n-dims/mechanical-nlpd", new Random(seed))

    CorrelationAccuracyRealData(
      csvPath = csvPath,
      outputHeaders = outputHeaders,
      metric = StdConfidence(Seq(0, 1), observational = true),
      numTest = numTest,
      numTrainSeq = numTrainSeq,
      numBags = numBags,
      numTrials = numTrials,
    ).runTrialsAndSave(s"./correlation-study-n-dims/mechanical-stdcon", new Random(seed))
  }

}

/**
  * Compare the accuracy of several methods of estimating correlation on real, tabular data.
  *
  * @param csvPath path from which to read file
  * @param outputHeaders names of the headers corresponding to outputs
  * @param metric figure of merit to calculate over the predictions
  * @param numTest number of test data points
  * @param numTrainSeq sequence of number of training points
  * @param numBags number of trees in the ensemble
  * @param numTrials number of independent trials to run
  */
case class CorrelationAccuracyRealData(
                                        csvPath: String,
                                        outputHeaders: Vector[String],
                                        metric: Metric,
                                        numTest: Int,
                                        numTrainSeq: Seq[Int],
                                        numBags: Option[Int],
                                        numTrials: Int
                                       ) extends AccuracyScan[Seq[Any]] {
  override def variedParameter: VariedParameter = NumTraining(numBags = numBags)

  override def meritsMap: Map[String, Merit[Seq[Any]]] = metric.makeMeritsMap

  override def parameterValues: Seq[Double] = numTrainSeq.map(_.toDouble)

  override def makeParameterSet(parameterValue: Double): CorrelationAccuracyRealDataParameters =
    CorrelationAccuracyRealDataParameters(
      outputHeaders = outputHeaders,
      metric = metric,
      numTest = numTest,
      numTrain = parameterValue.toInt,
      numBags = numBags.getOrElse(parameterValue.toInt),
      numTrials = numTrials
    )

  private val (inputs, outputs) = extractInputsOutputsFromCSV(csvPath, outputHeaders).unzip
  private val numData = inputs.length

  override def runTrial(parameterValue: Double, rng: Random): (PredictionResult[Seq[Any]], Seq[Seq[Any]]) = {
    val parameterSet = makeParameterSet(parameterValue)
    val numTrain = parameterSet.numTrain
    val numBags = parameterSet.numBags

    val dataSplitRng = new Random(rng.nextLong())
    val trainRng = new Random(rng.nextLong())
    val bagSeed = rng.nextLong()

    val shuffledIndices = dataSplitRng.shuffle((0 until numData).toVector)
    val testIndices = shuffledIndices.take(numTest)
    val trainIndices = shuffledIndices.slice(numTest, numTest + numTrain.toInt)

    val trainInputs = trainIndices.map(inputs)
    val trainOutputs = trainIndices.map(outputs)
    val testInputs = testIndices.map(inputs)
    val testOutputs = testIndices.map(outputs)

    val learner = new MultiTaskStandardizer(MultiTaskTreeLearner(rng = trainRng))
    val baggedLearner = MultiTaskBagger(
      learner,
      numBags = numBags,
      randBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(bagSeed))),
      uncertaintyCalibrationLevel = Some(0.683)
    )

    val RF = baggedLearner.train(
      trainInputs,
      trainOutputs.transpose
    ).getModel()
    val predictionResult = RF.transform(testInputs)
    (predictionResult, testOutputs)
  }
}

case class CorrelationAccuracyRealDataParameters(
                                                outputHeaders: Vector[String],
                                                metric: Metric,
                                                numTest: Int,
                                                numTrain: Int,
                                                numBags: Int,
                                                numTrials: Int,
                                                ) extends ParameterSet {

  override def headers: Seq[String] = Seq(
    "correlation method", "metric", "mean value", "std error of value",
    "trials", "train", "test", "bags", "observational"
  ) ++ metric.indices.map(i => s"output index $i")

  override def makeRow(label: String, y: Double, yErr: Double): Seq[Any] = Seq(
    label, metric.name, y, yErr,
    numTrials, numTrain, numTest, numBags, metric.observational,
  ) ++ metric.indices.map(outputHeaders)

}
