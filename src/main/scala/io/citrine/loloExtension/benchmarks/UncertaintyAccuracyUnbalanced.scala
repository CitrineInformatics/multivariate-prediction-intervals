package io.citrine.loloExtension.benchmarks

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import io.citrine.lolo.bags.{BaggedResult, Bagger, UncertaintyMethods}
import io.citrine.lolo.stats.StatsUtils
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.lolo.trees.splits.RegressionSplitter
import io.citrine.lolo.validation.Merit
import io.citrine.loloExtension.benchmarks.BenchmarkUtils.{extractInputsOutputsFromCSV, writeText}
import io.citrine.loloExtension.validation.{NegativeLogProbabilityDensity, StandardConfidenceExtension, StandardErrorExtension}
import org.apache.commons.math3.random.MersenneTwister

import scala.util.Random

/**
  * This is meant to be a quick test of how the "recalibrated bootstrap" and "out-of-bag constant" methods do at
  * estimating prediction intervals when the training data and test data are not drawn from identical distributions.
  * Specifically, this uses the Mechanical Properties dataset. In the training data, 60 out of 64 points
  * were measured with the sample under tension. In the test data, all 32 points were measured with the sample
  * under compression. We find that the "out-of-bag constant" method is highly over-confident in this situation,
  * whereas the "recalibrated bootstrap" is much more well calibrated, though still slightly over-confident.
  *
  * This method includes lots of duplicate code. It is similar to the calculation done in UncertaintyAccuracy.scala,
  * except that the method of generating training/test data is modified and we are calculating the merits for
  * a single configuration instead of making a plot over multiple configurations.
  * Given the one-off nature of this calculation, I elected to keep the repeated code here rather than trying
  * to fold this situation into the AccuracyScan framework.
  */
object UncertaintyAccuracyUnbalanced {

  def main(args: Array[String]): Unit = {
    val seed = 7321578032L
    val rng = new Random(seed)
    val numTrials = 50
    val numTensionTrain = 60
    val numCompressionTrain = 4
    val numTensionTest = 0
    val numCompressionTest = 32
    val numBags = 64
    val csvPath = "mechanical_properties_clean.csv"
    val outputHeaders = Vector(
      "PROPERTY: YS (MPa)",
      "PROPERTY: Elongation (%)"
    )
    val outputIndex = 0

    val outputPath = "./uncertainty-study/mechanical-properties/Youngs-Modulus-disbalanced.csv"
    val metadata: String = s"output: ${outputHeaders(outputIndex)}\ntrials: $numTrials\nbags: $numBags\n" ++
      s"training points (tension): $numTensionTrain\ntraining points (compression): $numCompressionTrain\n" ++
      s"test point (tension): $numTensionTest\ntest points (compression): $numCompressionTest\n"
    writeText(outputPath, metadata, append = false)

    val fullData = extractInputsOutputsFromCSV(csvPath, outputHeaders)
    val (compressionData, tensionData) = fullData.partition(data => data._1(2).asInstanceOf[String] == "C")

    val meritsMap: Map[String, Merit[Double]] = Map(
      "Standard Confidence (bootstrap)" -> StandardConfidenceExtension(method = UncertaintyMethods.Bootstrap),
      "Standard Confidence (oob constant)" -> StandardConfidenceExtension(method = UncertaintyMethods.OutOfBagConstant),
      "Standard Error (bootstrap)" -> StandardErrorExtension(method = UncertaintyMethods.Bootstrap),
      "Standard Error (oob constant)" -> StandardErrorExtension(method = UncertaintyMethods.OutOfBagConstant),
      "NLPD (bootstrap)" -> NegativeLogProbabilityDensity(method = UncertaintyMethods.Bootstrap),
      "NLPD (oob constant)" -> NegativeLogProbabilityDensity(method = UncertaintyMethods.OutOfBagConstant)
    )



    val pvas = (0 until numTrials).map { _ =>
      val thisRng = new Random(rng.nextLong())
      val dataGenRng = new Random(thisRng.nextLong())
      val trainRng = new Random(thisRng.nextLong())
      val bagSeed = thisRng.nextLong()

      val thisCompressionData = dataGenRng
        .shuffle(compressionData).take(numCompressionTest + numCompressionTrain)
        .map { case (inputs, outputs) => (inputs, outputs(outputIndex).asInstanceOf[Double]) }
      val thisTensionData = dataGenRng
        .shuffle(tensionData).take(numTensionTest + numTensionTrain)
        .map { case (inputs, outputs) => (inputs, outputs(outputIndex).asInstanceOf[Double]) }

      val trainData = thisCompressionData.take(numCompressionTrain) ++ thisTensionData.take(numTensionTrain)
      val testData = thisCompressionData.drop(numCompressionTrain) ++ thisTensionData.drop(numTensionTrain)

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
      val RF = baggedLearner.train(trainData).getModel()

      val predictionResult = RF.transform(testData.map(_._1)).asInstanceOf[BaggedResult[Double]]
      val actual = testData.map(_._2)

      (predictionResult, actual)
    }

    meritsMap.foreach { case (key, merit) =>
      val results = pvas.map { case (predictionResult, actual) => merit.evaluate(predictionResult, actual, new Random())}
      val mean = StatsUtils.mean(results)
      val stderr = math.sqrt(StatsUtils.variance(results, dof = 1)) / math.sqrt(results.length)
      val text = s"$key: $mean +/- $stderr\n"
      writeText(outputPath, text, append = true)
    }
  }

}
