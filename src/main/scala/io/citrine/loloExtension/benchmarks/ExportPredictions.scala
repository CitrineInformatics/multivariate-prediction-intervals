package io.citrine.loloExtension.benchmarks

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import io.citrine.lolo.{Model, PredictionResult, RegressionResult}
import io.citrine.lolo.bags.{BaggedModel, BaggedTrainingResult, Bagger, MultiPredictionBaggedResult}
import io.citrine.lolo.trees.regression.RegressionTreeLearner
import io.citrine.lolo.trees.splits.RegressionSplitter
import io.citrine.loloExtension.benchmarks.BenchmarkUtils.{writeCSVRow, writeText}
import io.citrine.loloExtension.benchmarks.library.function.{DataGenerator, FriedmanGrosseFunction, FriedmanSilvermanFunction, MechanicalPropertiesData, ThermoelectricsData}
import org.apache.commons.math3.random.MersenneTwister

import scala.collection.parallel.immutable.ParSeq
import scala.util.Random

/**
  * Export the raw data for a series of test/train samples generated by some function.
  * This includes the input, the true output, and the observed output (with noise).
  * For the train data this includes the out-of-bag mean and standard deviation.
  * For the test data this includes the bootstrap mean and standard deviation.
  * If uninterested in a test set, let numTest = None.
  */
object ExportPredictions {

  def main(args: Array[String]): Unit = {
    val seed = 12536035898L
    val numTrials = 100
    val numTrain = 128
    val numBags = 64

    // Generate both test and train data to study the predicted uncertainty
    Seq(0.0, 1.0, 2.0, 5.0, 10.0, 20.0).foreach { noise =>
      runTrials(
        directory = s"./data-export/Friedman-Grosse-noise-${noise.toInt}/",
        numTrials = numTrials,
        numTrain = numTrain,
        numBags = numBags,
        numTest = Some(numTrain),
        dataGenerator = FriedmanGrosseFunction(columns = 8),
        noise = noise,
        rng = new Random(seed)
      )
    }

    // Generate only train data to study the recalibration factor
    runTrials(
      directory = "./data-export/Friedman-Silverman-noise-2/",
      numTrials = numTrials,
      numTrain = numTrain,
      numBags = numBags,
      numTest = None,
      dataGenerator = FriedmanSilvermanFunction(columns = 12),
      noise = 2.0,
      rng = new Random(seed)
    )

    runTrials(
      directory = "./data-export/thermoelectrics-zt/",
      numTrials = numTrials,
      numTrain = numTrain,
      numBags = numBags,
      numTest = None,
      dataGenerator = ThermoelectricsData(outputIndex = 0),
      noise = 0.0,
      rng = new Random(seed)
    )

    runTrials(
      directory = "./data-export/mechanical-properties-elongation/",
      numTrials = numTrials,
      numTrain = numTrain,
      numBags = numBags,
      numTest = None,
      dataGenerator = MechanicalPropertiesData(outputIndex = 1),
      noise = 0.0,
      rng = new Random(seed)
    )
  }

  def runTrials(
                  directory: String,
                  numTrials: Int,
                  numTrain: Int,
                  numBags: Int,
                  numTest: Option[Int],
                  dataGenerator: DataGenerator,
                  noise: Double,
                  rng: Random
                ): Unit = {
    val metadata: String = s"f: ${dataGenerator.name}\nnoise level: $noise\ntraining points: $numTrain\n" ++
      s"bags: $numBags\ntest points: ${numTest.getOrElse(0)}"
    val metadataPath = s"${directory}metadata.csv"
    writeText(metadataPath, metadata, append = false)

    (0 until numTrials).foreach { i =>
      val thisRng = new Random(rng.nextLong())
      generateAndExportData(
        directory = directory,
        index = i,
        numTrain = numTrain,
        numBags = numBags,
        numTest = numTest,
        dataGenerator = dataGenerator,
        noise = noise,
        rng = thisRng
      )
    }
  }

  def generateAndExportData(
                             directory: String,
                             index: Int,
                             numTrain: Int,
                             numBags: Int,
                             numTest: Option[Int],
                             dataGenerator: DataGenerator,
                             noise: Double,
                             rng: Random
                           ): Unit = {
    val dataGenSeed = rng.nextLong()
    val dataNoiseRng = new Random(rng.nextLong())
    val trainRng = new Random(rng.nextLong())
    val bagSeed = rng.nextLong()

    // input, true output, observed output
    val allData: Vector[(Vector[Any], Double, Double)] = dataGenerator
      .generateData(numTrain + numTest.getOrElse(0), dataGenSeed)
      .map { case (inputs, output) =>
        (inputs, output, output + noise * dataNoiseRng.nextGaussian())
      }
    val trainData = allData.take(numTrain)
    val testData = allData.drop(numTrain)

    val learner = RegressionTreeLearner(
      splitter = RegressionSplitter(rng = trainRng),
      rng = trainRng
    )
    val baggedLearner = Bagger(
      method = learner,
      numBags = numBags,
      uncertaintyCalibration = false, // I'm interested in the bootstrap standard deviation, and will investigate the rescaling externally
      randBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(bagSeed)))
    )

    val RFmeta = baggedLearner.train(
      trainData.map { x => (x._1, x._3) }
    ).asInstanceOf[BaggedTrainingResult[Double]]
    val RF = RFmeta.getModel().asInstanceOf[BaggedModel[Double]]

    val Nib: Vector[Vector[Int]] = RF.getNib
    val models: ParSeq[Model[PredictionResult[Double]]] = RF.getModels()
    // out of bag (mean, bootstrap standard deviation) for each test point
    val oobPredictions: Seq[(Double, Double)] = trainData.indices.map { idx =>
      val inputs = trainData(idx)._1
      val oobModels = models.zip(Nib.map(_ (idx))).filter(_._2 == 0).map(_._1)
      val model = new BaggedModel(oobModels, Nib.filter {
        _ (idx) == 0
      }, useJackknife = true)
      val predicted = model.transform(Seq(inputs))
      val mean = predicted.getExpected().head
      val uncertainty = predicted.asInstanceOf[RegressionResult].getStdDevObs().get.head
      (mean, uncertainty)
    }

    val trainPath = s"${directory}train-$index.csv"
    // Not including x coordinates for now since I do not intend to plot results as a function of x
    val trainHeaders = Seq("y", "y (observed)", "out of bag mean", "out of bag standard deviation", "normalized residual")
    writeCSVRow(trainPath, trainHeaders, append = false)
    trainData.indices.foreach { i =>
      val trainPoint = trainData(i)
      val oobPrediction = oobPredictions(i)
      val mean = oobPrediction._1
      val uncertainty = oobPrediction._2
      val normResid = (mean - trainPoint._3) / uncertainty
      val row = Seq(trainPoint._2, trainPoint._3, mean, uncertainty, normResid)
      writeCSVRow(trainPath, row, append = true)
    }

    if (testData.nonEmpty) {
      val testPredictions = RF.transform(testData.map(_._1)).asInstanceOf[MultiPredictionBaggedResult]
      val testMean = testPredictions.getExpected()
      val testStd = testPredictions.getUncertainty(observational = true).get.asInstanceOf[Seq[Double]]

      val testHeaders = Seq("y", "y (observed)", "bootstrap mean", "bootstrap standard deviation", "normalized residual")
      val testPath = s"${directory}test-$index.csv"
      writeCSVRow(testPath, testHeaders, append = false)
      testData.indices.foreach { i =>
        val testPoint = testData(i)
        val mean = testMean(i)
        val uncertainty = testStd(i)
        val normResid = (mean - testPoint._3) / uncertainty
        val row = Seq(testPoint._2, testPoint._3, mean, uncertainty, normResid)
        writeCSVRow(testPath, row, append = true)
      }
    }
  }

}