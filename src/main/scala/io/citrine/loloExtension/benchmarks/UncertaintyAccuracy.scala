package io.citrine.loloExtension.benchmarks

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import io.citrine.lolo.PredictionResult
import io.citrine.lolo.bags.{BaggedResult, Bagger, UncertaintyMethods}
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.lolo.trees.splits.RegressionSplitter
import io.citrine.lolo.validation.Merit
import io.citrine.loloExtension.benchmarks.library.function.{DataGenerator, FriedmanGrosseFunction, FriedmanSilvermanFunction, MechanicalPropertiesData, ThermoelectricsData}
import io.citrine.loloExtension.benchmarks.library.{NumTraining, VariedParameter}
import io.citrine.loloExtension.validation.{NegativeLogProbabilityDensity, StandardConfidenceExtension, StandardErrorExtension}
import org.apache.commons.math3.random.MersenneTwister

import scala.util.Random

object UncertaintyAccuracy {

  def main(args: Array[String]): Unit = {
    runFriedmanSilverman()
    runFriedmanGrosse()
    runMechanicalProperties()
    runThermoelectrics()
  }

  def runFriedmanSilverman(): Unit = {
    val seed = 9153201002L
    val directory = "./uncertainty-study/Friedman-Silverman/"
    val numTrials = 64
    val numTest = 128
    val numBags = Some(64)
    val numTrainSeq = Seq(32, 64, 128, 256, 512, 1024)
    val function = FriedmanSilvermanFunction(columns = 12)

    UncertaintyAccuracy(
      dataGenerator = function,
      numTrainSeq = numTrainSeq,
      numTrials = numTrials,
      numTest = numTest,
      numBags = numBags,
      samplingNoise = 2.0,
      observational = true
    ).runTrialsAndSave(s"${directory}noise-2-prediction", new Random (seed))
  }

  def runFriedmanGrosse(): Unit = {
    val seed = 9153201002L
    val directory = "./uncertainty-study/Friedman-Grosse/"
    val numTrials = 64
    val numTest = 128
    val numBags = Some(64)
    val numTrainSeq = Seq(32, 64, 128, 256, 512, 1024)
    val function = FriedmanGrosseFunction(columns = 8)

    UncertaintyAccuracy(
      dataGenerator = function,
      numTrainSeq = numTrainSeq,
      numTrials = numTrials,
      numTest = numTest,
      numBags = numBags,
      samplingNoise = 2.0,
      observational = true
    ).runTrialsAndSave(s"${directory}noise-2-prediction", new Random (seed))

    UncertaintyAccuracy(
      dataGenerator = function,
      numTrainSeq = numTrainSeq,
      numTrials = numTrials,
      numTest = numTest,
      numBags = numBags,
      samplingNoise = 0.0,
      observational = true
    ).runTrialsAndSave(s"${directory}noise-0-prediction", new Random (seed))
  }

  def runMechanicalProperties(): Unit = {
    val seed = 9153201002L
    val directory = "./uncertainty-study/mechanical-properties/"
    val numTrials = 64
    val numTest = 64
    val numBags = Some(64)
    val numTrainSeq = Seq(32, 64, 128, 196)

    UncertaintyAccuracy(
      dataGenerator = MechanicalPropertiesData(1),
      numTrainSeq = numTrainSeq,
      numTrials = numTrials,
      numTest = numTest,
      numBags = numBags,
      samplingNoise = 0.0,
      observational = true
    ).runTrialsAndSave(s"${directory}Elongation", new Random (seed))

    UncertaintyAccuracy(
      dataGenerator = MechanicalPropertiesData(0),
      numTrainSeq = numTrainSeq,
      numTrials = numTrials,
      numTest = numTest,
      numBags = numBags,
      samplingNoise = 0.0,
      observational = true
    ).runTrialsAndSave(s"${directory}Youngs-Modulus", new Random (seed))
  }

  def runThermoelectrics(): Unit = {
    val seed = 9153201002L
    val directory = "./uncertainty-study/thermoelectrics/"
    val numTrials = 64
    val numTest = 64
    val numBags = Some(64)
    val numTrainSeq = Seq(32, 64, 128, 256, 512)

    UncertaintyAccuracy(
      dataGenerator = ThermoelectricsData(0),
      numTrainSeq = numTrainSeq,
      numTrials = numTrials,
      numTest = numTest,
      numBags = numBags,
      samplingNoise = 0.0,
      observational = true
    ).runTrialsAndSave(s"${directory}ZT", new Random (seed))
  }
}

/**
  * Calculate several metrics that measure the accuracy of uncertainty estimates.
  *
  * @param dataGenerator object that generates training and test inputs and labels
  * @param numTrainSeq sequence of number of training data to generate
  * @param numTrials number of independent trials to run and combine the results of
  * @param numTest number of test data points
  * @param numBags number of bags in the random forest (if None, set numBags = numTrain)
  * @param samplingNoise level of normally-distributed random noise to apply to the training and test data
  * @param observational whether to compare uncertainty estimates to the observed value or the ground-truth value
  */
case class UncertaintyAccuracy(
                                dataGenerator: DataGenerator,
                                numTrainSeq: Seq[Int],
                                numTrials: Int,
                                numTest: Int,
                                numBags: Option[Int],
                                samplingNoise: Double,
                                observational: Boolean
                         ) extends AccuracyScan[Double] {
  override def variedParameter: VariedParameter = NumTraining(numBags = numBags)

  override def meritsMap: Map[String, Merit[Double]] = Map(
    "Standard Confidence (bootstrap)" -> StandardConfidenceExtension(method = UncertaintyMethods.Bootstrap),
    "Standard Confidence (jackknife)" -> StandardConfidenceExtension(method = UncertaintyMethods.Jackknife),
    "Standard Confidence (oob constant)" -> StandardConfidenceExtension(method = UncertaintyMethods.OutOfBagConstant),
    "Standard Error (bootstrap)" -> StandardErrorExtension(method = UncertaintyMethods.Bootstrap),
    "Standard Error (jackknife)" -> StandardErrorExtension(method = UncertaintyMethods.Jackknife),
    "Standard Error (oob constant)" -> StandardErrorExtension(method = UncertaintyMethods.OutOfBagConstant),
    "NLPD (bootstrap)" -> NegativeLogProbabilityDensity(method = UncertaintyMethods.Bootstrap),
    "NLPD (jackknife)" -> NegativeLogProbabilityDensity(method = UncertaintyMethods.Jackknife),
    "NLPD (oob constant)" -> NegativeLogProbabilityDensity(method = UncertaintyMethods.OutOfBagConstant)
  )

  override def parameterValues: Seq[Double] = numTrainSeq.map(_.toDouble)

  override def makeParameterSet(parameterValue: Double): UncertaintyAccuracyParameters =
    UncertaintyAccuracyParameters(
      dataGenerator = dataGenerator,
      numTrials = numTrials,
      numTrain = parameterValue.toInt,
      numTest = numTest,
      numBags = numBags.getOrElse(parameterValue.toInt),
      samplingNoise = samplingNoise,
      observational = observational
    )

  override def runTrial(parameterValue: Double, rng: Random): (PredictionResult[Double], Seq[Double]) = {
    val parameterSet = makeParameterSet(parameterValue)
    val numTrain = parameterSet.numTrain
    val numBags = parameterSet.numBags

    val dataGenSeed = rng.nextLong()
    val dataNoiseSeed = new Random(rng.nextLong())
    val trainRng = new Random(rng.nextLong())
    val bagSeed = rng.nextLong()

    val (inputs, label) = dataGenerator.generateData(numTrain + numTest, dataGenSeed).unzip
    val labelNoise = label.map(_ + samplingNoise * dataNoiseSeed.nextGaussian())

    val learner = RegressionTreeLearner(
      splitter = RegressionSplitter(rng = trainRng),
      rng = trainRng
    )
    val baggedLearner = Bagger(
      method = learner,
      numBags = numBags,
      uncertaintyCalibration = true,
      randBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(bagSeed)))
    )

    val RF = baggedLearner.train(
      inputs.take(numTrain).zip(labelNoise.take(numTrain))
    ).getModel()

    val predictionResult = RF.transform(inputs.drop(numTrain)).asInstanceOf[BaggedResult[Double]]
    val actual = if (observational) labelNoise.drop(numTrain) else label.drop(numTrain)
    (predictionResult, actual)
  }
}

case class UncertaintyAccuracyParameters(
                                          dataGenerator: DataGenerator,
                                          numTrials: Int,
                                          numTrain: Int,
                                          numTest: Int,
                                          numBags: Int,
                                          samplingNoise: Double,
                                          observational: Boolean
                                        ) extends ParameterSet {

  override def headers: Seq[String] = Seq(
    "metric", "mean value", "std error of value",
    "function", "trials", "train", "test", "bags", "sample noise", "observational"
  )

  override def makeRow(label: String, y: Double, yErr: Double): Seq[Any] = Seq(
    label, y, yErr,
    dataGenerator.name, numTrials, numTrain, numTest, numBags, samplingNoise, observational
  )

}