package io.citrine.loloExtension.stats

import io.citrine.loloExtension.benchmarks.library.Objective
import io.citrine.loloExtension.stats.predictedVsActual.PvaRealNDimensions

import scala.util.Random

object StatsUtils {

  /** The median of a sequence of real values */
  def median(X: Seq[Double]): Double = {
    val (lower, upper) = X.sorted.splitAt(X.size / 2)
    if (X.size % 2 == 0) (lower.last + upper.head) / 2.0 else upper.head
  }

  /**
    * Given a sequence of real values, produce corresponding values that are quadratically related.
    *
    * @param X input real values
    * @param fuzz magnitude of a normally distributed random variable to add to Y, to decorrelate it from X
    * @param rng random seed, for reproducibility
    * @return Y = (X - mean(X))**2 + N(0, fuzz)
    */
  def makeQuadraticCorrelatedData(X: Seq[Double], fuzz: Double = 0.0, rng: Random = new Random()): Seq[Double] = {
    require(fuzz >= 0.0)
    val mu = X.sum / X.size
    X.map(x => math.pow(x - mu, 2.0) + fuzz * rng.nextGaussian())
  }

  /**
    * Generate real-valued training data given a ground-truth function.
    * The domain of the function is assumed to be [0, 1]**d, for dimensionality d
    *
    * @param rows number of training rows
    * @param cols dimensionality of training data
    * @param function takes a real-valued input vector and produces a real-valued output
    * @param noise normally distributed random noise to add to the output
    * @param seed random seed, for reproducibility
    * @return sequence of tuples (input, output)
    */
  def generateTrainingData(
                            rows: Int,
                            cols: Int,
                            function: (Seq[Double] => Double),
                            noise: Double = 0.0,
                            seed: Long = 0L
                          ): Vector[(Vector[Double], Double)] = {
    val rnd = new Random(seed)
    Vector.fill(rows) {
      val input = Vector.fill(cols)(rnd.nextDouble())
      (input, function(input) + noise * rnd.nextGaussian())
    }
  }

  /**
    * Use sampling to estimate the likelihood of a predicted distribution satisfying two objectives.
    *
    * @param pva predicted-vs-actual point in N dimensions
    * @param objectives sequence of objectives corresponding to the dimensions of the pva object
    * @param seed random seed, for reproducibility
    * @param numSamples number of samples to draw
    * @return
    */
  def estimateSatisfactionProbability(
                                       pva: PvaRealNDimensions,
                                       objectives: Seq[Objective],
                                       seed: Long,
                                       numSamples: Int = 10000
                                     ): Double = {
    require(pva.numOutputs == objectives.length)
    val samples = pva.sample(numSamples)
    val numSatisfaction = samples.count { sample =>
      objectives.zipWithIndex.forall { case (objective, i) =>
        objective.satisfies(sample(i))
      }
    }
    numSatisfaction.toDouble / numSamples
  }

}
