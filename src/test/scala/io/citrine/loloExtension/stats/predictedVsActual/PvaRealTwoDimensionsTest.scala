package io.citrine.loloExtension.stats.predictedVsActual

import org.junit.Test

class PvaRealTwoDimensionsTest {

  /** Test correctness of Mahalanobis distance calculation. */
  @Test
  def testMahalanobis(): Unit = {
    val pva = PvaRealTwoDimensions(
      predictions = (5.0, 0.0),
      actuals = (3.0, 1.0),
      uncertainties = (4.0, 0.5),
      correlationCoefficient = -0.6
    )
    // Using dx/sx = 2.0 / 4.0 = 0.5, dy = -1.0 / 0.5 = -2.0, and rho = -0.6, we get r^2 = 4.765625
    assert(pva.mahalanobisSquared == 4.765625)
  }

}
