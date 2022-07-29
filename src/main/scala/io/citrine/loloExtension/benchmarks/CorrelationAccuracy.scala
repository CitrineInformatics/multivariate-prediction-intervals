package io.citrine.loloExtension.benchmarks

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import io.citrine.lolo.PredictionResult
import io.citrine.lolo.bags.MultiTaskBagger
import io.citrine.lolo.stats.StatsUtils.makeLinearCorrelatedData
import io.citrine.lolo.trees.multitask.MultiTaskTreeLearner
import io.citrine.lolo.validation.Merit
import io.citrine.loloExtension.benchmarks.library._
import io.citrine.loloExtension.benchmarks.library.function._
import io.citrine.loloExtension.stats.StatsUtils.{generateTrainingData, makeQuadraticCorrelatedData}
import org.apache.commons.math3.random.MersenneTwister

import scala.util.Random

object CorrelationAccuracy {

  def main(args: Array[String]): Unit = {
    val seed = 52109317L
    val numTrials = 16
    val numTest = 128
    val numTrain = 128
    val friedmanSilverman = FriedmanSilvermanFunction(columns = 12)
    val friedmanGrosse = FriedmanGrosseFunction(columns = 8)

    val standardNoise = 1.0
    val trainRhoDefault = 0.9
    val quadraticFuzzDefault = 0.5
    val numBagsDefault = 64
    val basePath = "./correlation-study-n-dims/"

    val numTrainSeq: Seq[Double] = Seq(32, 64, 128, 256, 512, 1024)
    val numBagsSeq: Seq[Double] = Seq(64, 128, 256, 512, 1024)
    val noiseSeq = Seq(0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)

    val nlpdPrediction = NLPD(Seq(0, 1, 2), observational = true)
    val stdconPrediction = StdConfidence(Seq(0, 1, 2), observational = true)

    CorrelationAccuracy(
      metric = nlpdPrediction,
      variedParameter = NumTraining(numBags = Some(numBagsDefault)),
      parameterValues = numTrainSeq,
      function = friedmanSilverman,
      numTrials = numTrials,
      numTrain = numTrain,
      numTest = numTest,
      samplingNoise = standardNoise,
      rhoTrain = trainRhoDefault,
      quadraticCorrelationFuzz = quadraticFuzzDefault
    ).runTrialsAndSave(s"${basePath}fixed-bags-64-nlpd", new Random(seed))
    CorrelationAccuracy(
      metric = stdconPrediction,
      variedParameter = NumTraining(numBags = Some(numBagsDefault)),
      parameterValues = numTrainSeq,
      function = friedmanSilverman,
      numTrials = numTrials,
      numTrain = numTrain,
      numTest = numTest,
      samplingNoise = standardNoise,
      rhoTrain = trainRhoDefault,
      quadraticCorrelationFuzz = quadraticFuzzDefault
    ).runTrialsAndSave(s"${basePath}fixed-bags-64-stdcon", new Random(seed))

    CorrelationAccuracy(
      metric = nlpdPrediction,
      variedParameter = Bags,
      parameterValues = numBagsSeq,
      function = friedmanGrosse,
      numTrials = numTrials,
      numTrain = numTrain,
      numTest = numTest,
      samplingNoise = standardNoise,
      rhoTrain = trainRhoDefault,
      quadraticCorrelationFuzz = quadraticFuzzDefault
    ).runTrialsAndSave(s"${basePath}Friedman-Grosse-vary-bags-nlpd", new Random(seed))

    CorrelationAccuracy(
      metric = nlpdPrediction,
      variedParameter = NumTraining(numBags = Some(numBagsDefault)),
      parameterValues = numTrainSeq,
      function = friedmanGrosse,
      numTrials = numTrials,
      numTrain = numTrain,
      numTest = numTest,
      samplingNoise = standardNoise,
      rhoTrain = trainRhoDefault,
      quadraticCorrelationFuzz = quadraticFuzzDefault
    ).runTrialsAndSave(s"${basePath}Friedman-Grosse-fixed-bags-nlpd", new Random(seed))
    CorrelationAccuracy(
      metric = stdconPrediction,
      variedParameter = NumTraining(numBags = Some(numBagsDefault)),
      parameterValues = numTrainSeq,
      function = friedmanGrosse,
      numTrials = numTrials,
      numTrain = numTrain,
      numTest = numTest,
      samplingNoise = standardNoise,
      rhoTrain = trainRhoDefault,
      quadraticCorrelationFuzz = quadraticFuzzDefault
    ).runTrialsAndSave(s"${basePath}Friedman-Grosse-fixed-bags-stdcon", new Random(seed))

    CorrelationAccuracy(
      metric = nlpdPrediction,
      variedParameter = Noise,
      parameterValues = noiseSeq,
      function = friedmanGrosse,
      numTrials = numTrials,
      numTrain = numTrain,
      numTest = numTest,
      samplingNoise = 0.0,
      rhoTrain = trainRhoDefault,
      quadraticCorrelationFuzz = quadraticFuzzDefault,
    ).runTrialsAndSave(s"${basePath}Friedman-Grosse-vary-noise-nlpd", new Random(seed))

  }
}

/**
  * Compare the accuracy of several methods of estimating correlation.
  *
  * @param metric figure of merit to calculate over the predictions
  * @param variedParameter the parameter to vary as part of the scan
  * @param parameterValues values that the varied parameter takes on
  * @param function function to generate training and test data
  * @param numTrials number of independent trials to run and combine the results of
  * @param numTrain number of training data points
  * @param numTest number of test data points
  * @param samplingNoise level of normally-distributed random noise to apply to the training and test data
  * @param rhoTrain linear correlation coefficient bewteen label 0 and label 1
  * @param quadraticCorrelationFuzz level of normally-distributed random values to apply to the second label
  *                                 in the quadratic pair. Note that this value modifies the true underlying data,
  *                                 which is separate from applying sampling noise.
  */
case class CorrelationAccuracy(
                                metric: Metric,
                                variedParameter: VariedParameter,
                                parameterValues: Seq[Double],
                                function: AnalyticFunction,
                                numTrials: Int,
                                numTrain: Int,
                                numTest: Int,
                                samplingNoise: Double,
                                rhoTrain: Double,
                                quadraticCorrelationFuzz: Double,
                              ) extends AccuracyScan[Seq[Any]] {
  override def meritsMap: Map[String, Merit[Seq[Any]]] = metric.makeMeritsMap

  override def makeParameterSet(parameterValue: Double): CorrelationAccuracyParameters = {
    variedParameter match {
      case TrainRho => CorrelationAccuracyParameters(
        metric = metric,
        function = function,
        numTrials = numTrials,
        numTrain = numTrain,
        numTest = numTest,
        numBags = numTrain,
        rhoTrain = parameterValue,
        quadraticCorrelationFuzz = quadraticCorrelationFuzz,
        samplingNoise = samplingNoise
      )
      case TrainQuadraticFuzz => CorrelationAccuracyParameters(
        metric = metric,
        function = function,
        numTrials = numTrials,
        numTrain = numTrain,
        numTest = numTest,
        numBags = numTrain,
        rhoTrain = rhoTrain,
        quadraticCorrelationFuzz = parameterValue,
        samplingNoise = samplingNoise
      )
      case Noise => CorrelationAccuracyParameters(
        metric = metric,
        function = function,
        numTrials = numTrials,
        numTrain = numTrain,
        numTest = numTest,
        numBags = numTrain,
        rhoTrain = rhoTrain,
        quadraticCorrelationFuzz = quadraticCorrelationFuzz,
        samplingNoise = parameterValue
      )
      case Bags => CorrelationAccuracyParameters(
        metric = metric,
        function = function,
        numTrials = numTrials,
        numTrain = numTrain,
        numTest = numTest,
        numBags = parameterValue.toInt,
        rhoTrain = rhoTrain,
        quadraticCorrelationFuzz = quadraticCorrelationFuzz,
        samplingNoise = samplingNoise
      )
      case NumTraining(optionNumBags) => CorrelationAccuracyParameters(
        metric = metric,
        function = function,
        numTrials = numTrials,
        numTrain = parameterValue.toInt,
        numTest = numTest,
        numBags = optionNumBags.getOrElse(parameterValue.toInt),
        rhoTrain = rhoTrain,
        quadraticCorrelationFuzz = quadraticCorrelationFuzz,
        samplingNoise = samplingNoise
      )
    }
  }

  override def runTrial(parameterValue: Double, rng: Random): (PredictionResult[Seq[Any]], Seq[Seq[Any]]) = {
    val parameterSet = makeParameterSet(parameterValue)
    val numTrain = parameterSet.numTrain
    val numTest = parameterSet.numTest
    val samplingNoise = parameterSet.samplingNoise
    val numBags = parameterSet.numBags
    val rhoTrain = parameterSet.rhoTrain
    val quadraticCorrelationFuzz = parameterSet.quadraticCorrelationFuzz

    val dataGenSeed = rng.nextLong()
    val dataNoiseSeed = new Random(rng.nextLong())
    val trainRng = new Random(rng.nextLong())
    val bagSeed = rng.nextLong()
    val linearCorrelationRng = new Random(rng.nextLong())
    val quadraticCorrelationRng = new Random(rng.nextLong())

    val fullData: Seq[(Vector[Double], Double)] = generateTrainingData(
      numTrain + numTest,
      function.numCols,
      noise = 0.0, // Add noise later, after computing covariate labels
      function = function.function,
      seed = dataGenSeed
    )

    val inputs: Seq[Vector[Double]] = fullData.map(_._1)
    val realLabel: Seq[Double] = fullData.map(_._2)
    val linearLabel: Seq[Double] = makeLinearCorrelatedData(realLabel, rhoTrain, linearCorrelationRng)
    val quadraticLabel: Seq[Double] = makeQuadraticCorrelatedData(realLabel, quadraticCorrelationFuzz, quadraticCorrelationRng)

    val realLabelNoise = realLabel.map(_ + samplingNoise * dataNoiseSeed.nextGaussian())
    val linearLabelNoise = linearLabel.map(_ + samplingNoise * dataNoiseSeed.nextGaussian())
    val quadraticLabelNoise = quadraticLabel.map(_ + samplingNoise * dataNoiseSeed.nextGaussian())

    val learner = MultiTaskTreeLearner(rng = trainRng)
    val baggedLearner = MultiTaskBagger(
      learner,
      numBags = numBags,
      randBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(bagSeed))),
      uncertaintyCalibrationLevel = Some(0.683)
    )

    val RF = baggedLearner.train(
      inputs.take(numTrain),
      Seq(realLabelNoise.take(numTrain), linearLabelNoise.take(numTrain), quadraticLabelNoise.take(numTrain))
    ).getModel()

    val predictionResult = RF.transform(inputs.drop(numTrain))
    val trueLabels = Seq(
      realLabel.drop(numTrain),
      linearLabel.drop(numTrain),
      quadraticLabel.drop(numTrain)
    ).transpose
    val observedLabels = Seq(
      realLabelNoise.drop(numTrain),
      linearLabelNoise.drop(numTrain),
      quadraticLabelNoise.drop(numTrain)
    ).transpose
    val actualLabels = if (metric.observational) observedLabels else trueLabels
    (predictionResult, actualLabels)
  }
}

case class CorrelationAccuracyParameters(
                                          metric: Metric,
                                          function: AnalyticFunction,
                                          numTrials: Int,
                                          numTrain: Int,
                                          numTest: Int,
                                          numBags: Int,
                                          rhoTrain: Double,
                                          quadraticCorrelationFuzz: Double,
                                          samplingNoise: Double,
                                        ) extends ParameterSet {

  override def headers: Seq[String] = Seq(
    "correlation method", "metric", "mean value", "std error of value",
    "function", "trials", "train", "test", "bags", "observational",
    "sample noise", "linear rho", "quadratic fuzz"
  ) ++ metric.indices.map(i => s"output index $i")

  override def makeRow(label: String, y: Double, yErr: Double): Seq[Any] = Seq(
    label, metric.name, y, yErr,
    function.name, numTrials, numTrain, numTest, numBags, metric.observational,
    samplingNoise, rhoTrain, quadraticCorrelationFuzz
  ) ++ metric.indices

}