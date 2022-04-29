package io.citrine.loloExtension.stats

import io.citrine.loloExtension.benchmarks.library.{GreaterThan, LessThan}
import io.citrine.loloExtension.stats.predictedVsActual.PvaRealNDimensions
import org.junit.Test

class StatsUtilsTest {

  /** Test the implementation of median */
  @Test
  def testMedian(): Unit = {
    assert(StatsUtils.median(Seq(14.0, 17.0, 12.0, 13.0, 13.0)) == 13.0)
    assert(StatsUtils.median(Seq(1.0, 0.0, 3.0, 4.0)) == 2.0)
  }

  /** Test the satisfaction probability estimate via sampling */
  @Test
  def testSatisfactionProbabilityEstimate(): Unit = {
    val mu1 = 0.0
    val mu2 = 0.0
    val sigma1 = 1.0
    val sigma2 = 1.0
    // Objectives are to be 1-sigma above on first output and 1-sigma below on second output.
    // Indices arguments are irrelevant, because the real data has already been packaged into a Pva
    // object so that the order of the values matches the order of the objectives.
    val objectives = Seq(
      GreaterThan("", 5, mu1 + sigma1),
      LessThan("", 3, mu2 - sigma2)
    )
    val seed = 34L

    /** Calculate the satisfaction probability for a given correlation coefficient. */
    def calculateProbability(rho: Double): Double = {
      val covariance = Array(
        Array(sigma1 * sigma1, rho * sigma1 * sigma2),
        Array(rho * sigma1 * sigma2, sigma2 * sigma2)
      )
      val pva = PvaRealNDimensions(
        prediction = Seq(mu1, mu2),
        actual = Seq(mu1, mu2), // actual values are irrelevant to this calculation
        covariance = covariance
      )
      StatsUtils.estimateSatisfactionProbability(
        pva = pva,
        objectives = objectives,
        seed = seed
      )
    }

    val rhoSeq = Seq(-0.8, -0.4, 0.0, 0.4, 0.8)
    val probs = rhoSeq.map(calculateProbability)
    val noCorrelationProbability = probs(2)
    // Probability of exceeding 1-sigma is 15.9%, so the uncorrelated case is (0.159)^2 ~ 0.025.
    assert(noCorrelationProbability > 0.02 && noCorrelationProbability < 0.03)
    // Increasing the correlation coefficient decreases the satisfaction probability
    probs.indices.tail.foreach { i =>
      assert(probs(i) < probs(i - 1))
    }
  }

}
