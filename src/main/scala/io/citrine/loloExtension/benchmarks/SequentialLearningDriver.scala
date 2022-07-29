package io.citrine.loloExtension.benchmarks

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import io.citrine.lolo.bags.{CorrelationMethods, MultiTaskBagger}
import io.citrine.lolo.transformers.MultiTaskStandardizer
import io.citrine.lolo.trees.multitask.MultiTaskTreeLearner
import io.citrine.loloExtension.benchmarks.BenchmarkUtils.{extractInputsOutputsFromCSV, writeCSVRow}
import io.citrine.loloExtension.benchmarks.library.Objective
import io.citrine.loloExtension.stats.StatsUtils.estimateSatisfactionProbability
import io.citrine.loloExtension.stats.predictedVsActual.PvaRealNDimensions
import org.apache.commons.math3.random.MersenneTwister

import scala.util.Random

/**
  * A driver for running sequential learning simulations on finite data sets.
  *
  * @param csvPath path to a CSV containing data points.
  * @param outputHeaders strings corresponding to headers in the csv that are to be treated as outputs.
  */
case class SequentialLearningDriver(
                                   csvPath: String,
                                   outputHeaders: Vector[String]
                                   ) {

  lazy val baseData = extractInputsOutputsFromCSV(csvPath, outputHeaders)

  /**
    * Run a sequence of sequential learning trials, selecting new data and re-training until a suitable data point is found.
    *
    * @param filepath where to save the results of each trial
    * @param numTrials number of independent trials to run
    * @param objectives to meet in order to halt sequential learning. The indices of the objectives correspond to
    *                   the indices of the desired outputs as specified in `outputHeaders`.
    * @param numInitialTraining number of randomly selected data points to begin SL with
    * @param method for calculating the correlation coefficient
    * @param seed random seed, for reproducibility
    * @param findAll whether the goal is to find all points that meet the targets or just one.
    */
  def runTrials(
                 filepath: String,
                 numTrials: Int,
                 objectives: Set[Objective],
                 numInitialTraining: Int,
                 method: CorrelationMethods.CorrelationMethod,
                 seed: Long,
                 uncertaintyCalibrationLevel: Double = 0.683,
                 findAll: Boolean = false
               ): Unit = {
    val allObjectives = objectives.toVector

    val rng = new Random(seed)
    val headers = Seq("method", "initial points") ++ allObjectives.indices.map(i => s"goal ${i + 1}") ++ Seq("points to find", "rounds")
    writeCSVRow(filepath, headers, append = false)

    // Determine which indices correspond to points that satisfy the objectives, and hence should not be part of initial trianing.
    val hiddenIndices = indicesOfSatisfyingData(baseData, objectives).toSet
    val targetsToFind = if (findAll) hiddenIndices.size else 1

    (0 until numTrials).foreach { _ =>
      val thisRng = new Random(rng.nextLong())
      val numRounds = runTrial(
        filepath = filepath,
        hiddenIndices = hiddenIndices,
        objectives = allObjectives,
        numInitialTraining = numInitialTraining,
        method = method,
        targetsToFind = targetsToFind,
        uncertaintyCalibrationLevel = uncertaintyCalibrationLevel,
        rng = thisRng
      )
      val row = Seq(method.toString, numInitialTraining) ++ allObjectives.map(_.toString) ++ Seq(targetsToFind, numRounds)
      writeCSVRow(filepath, row, append = true)
    }
  }

  private def runTrial(
                        filepath: String,
                        hiddenIndices: Set[Int],
                        objectives: Vector[Objective],
                        numInitialTraining: Int,
                        method: CorrelationMethods.CorrelationMethod,
                        targetsToFind: Int = 1,
                        uncertaintyCalibrationLevel: Double = 0.683,
                        rng: Random = new Random()
                      ): Int = {
    // Shuffle data so that the hidden indices are at the end, then set the first points as the initial training set.
    val shuffleSeed = rng.nextLong()
    val data = shuffleAndHideData(baseData, hiddenIndices, new Random(shuffleSeed))
    var trainIndices = (0 until numInitialTraining).toSet
    val allIndices = data.indices.toSet

    val treeRng = new Random(rng.nextLong())
    val baggerRng = new Random(rng.nextLong())
    val samplingRng = new Random(rng.nextLong())

    var numTargetsFound = 0
    var numRounds = 0
    while (numTargetsFound < targetsToFind && trainIndices.size < allIndices.size) {
      // make a random forest model and train it
      val baggedLearner = MultiTaskBagger(
        new MultiTaskStandardizer(MultiTaskTreeLearner(rng = new Random(treeRng.nextLong()))),
        numBags = 64,
        randBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(baggerRng.nextLong()))),
        uncertaintyCalibrationLevel = Some(uncertaintyCalibrationLevel)
      )
      val (trainData, testData) = partitionVectorByIndices(data, trainIndices)
      val RF = baggedLearner.train(
        trainData.map(_._1),
        trainData.map(_._2).transpose
      ).getModel()

      // predict on all of the test data
      val predictionResults = RF.transform(testData.map(_._1))
      val pvas = PvaRealNDimensions.makePva(
        predictionResults,
        actual = testData.map(_._2),
        indices = objectives.map(_.index),
        method = method,
        observational = true
      )

      // Calculate the probability of each test point satisfying the objectives, and select the most promising
      // point to be added to the training data set.
      val satisfactionProbs = pvas.map { pva =>
        estimateSatisfactionProbability(pva, objectives, samplingRng.nextLong())
      }
      // `falseIndex` is relative to the `testData` vector, but we want the index relative to the `data` vector.
      val falseIndex = satisfactionProbs.indices.maxBy(satisfactionProbs)
      val sortedTestIndices = (allIndices -- trainIndices).toVector.sorted
      val trueIndex = sortedTestIndices(falseIndex)

      // Add the new training point, iterate the round counter, and check to see if the goal has been achieved.
      trainIndices ++= Set(trueIndex)
      numRounds += 1
      val trueLabels = data(trueIndex)._2
      if (allObjectivesSatisfied(objectives.toSet, trueLabels)) numTargetsFound += 1
    }
    numRounds
  }

  /** Determine the indices of data points that satisfy a set of objectives. */
  private def indicesOfSatisfyingData(data: Vector[(Vector[Any], Vector[Any])], objectives: Set[Objective]): Vector[Int] = {
    data.zipWithIndex.filter { case (datum, _) => allObjectivesSatisfied(objectives, datum._2)}.map(_._2)
  }

  /** Determine if a given vector satisfies a set of objectives. */
  private def allObjectivesSatisfied(objectives: Set[Objective], v: Vector[Any]): Boolean = {
    objectives.forall(obj => obj.satisfiesByIndex(v))
  }

  /** Shuffles data so that initial training set is at the front, respecting hidden indices
    * (data points that are explicitly banned from the initial training set).
    */
  private def shuffleAndHideData(
                                  data: Vector[(Vector[Any], Vector[Any])],
                                  hiddenIndices: Set[Int],
                                  rng: Random
                                ): Vector[(Vector[Any], Vector[Any])] = {
    val (hiddenData, knownData) = partitionVectorByIndices(data, hiddenIndices)
    val shuffledData = rng.shuffle(knownData)
    shuffledData ++ hiddenData
  }

  /** Partition a vector based on the elements that correspond to a set of indices. */
  private def partitionVectorByIndices[T](
                                           v: Vector[T],
                                           indices: Set[Int]
                                         ): (Vector[T], Vector[T]) = {
    var idx = -1
    v.partition { _ =>
      idx += 1
      indices.contains(idx)
    }
  }

}
