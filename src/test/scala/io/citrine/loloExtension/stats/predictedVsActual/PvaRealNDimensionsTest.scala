package io.citrine.loloExtension.stats.predictedVsActual

import org.junit.Test

class PvaRealNDimensionsTest {

  @Test
  def testMahalanobis(): Unit = {
    // For an uncorrelated matrix, Mahalanobis is the normalized Euclidean distance
    val pva3d = PvaRealNDimensions(
      prediction = Seq(0.0, 0.0, 0.0),
      actual = Seq(1.0, 2.0, 10.0),
      covariance = Array(
        Array(1.0, 0.0, 0.0),
        Array(0.0, 4.0, 0.0),
        Array(0.0, 0.0, 100.0)
      )
    )
    assert(math.abs(pva3d.mahalanobisDistance - 1.732) < 0.001)

    // For a more complicated case check that the answer matches that calculated by PvaRealTwoDimensions
    val prediction = Seq(0.15, -0.52)
    val actual = Seq(1.18, -2.7)
    val uncertainties = Seq(0.9, 1.62)
    val rho = -0.73
    val pvaNd = PvaRealNDimensions(
      prediction = prediction,
      actual = actual,
      covariance = Array(
        Array(uncertainties(0) * uncertainties(0), rho * uncertainties(0) * uncertainties(1)),
        Array(rho * uncertainties(0) * uncertainties(1), uncertainties(1) * uncertainties(1))
      )
    )

    val pva2d = PvaRealTwoDimensions(
      predictions = (prediction(0), prediction(1)),
      actuals = (actual(0), actual(1)),
      uncertainties = (uncertainties(0), uncertainties(1)),
      correlationCoefficient = rho
    )

    assert(pvaNd.mahalanobisDistance == math.sqrt(pva2d.mahalanobisSquared))
  }

  @Test
  def testInvalid(): Unit = {
    val outOfBoundsRho = PvaRealNDimensions(
      prediction = Seq(0.0, 0.0),
      actual = Seq(0.0, 0.0),
      covariance = Array(
        Array(1.0, 1.1),
        Array(1.1, 1.0)
      )
    )
    val outOfBoundsRhoDraw = outOfBoundsRho.distribution.draw()
    assert(outOfBoundsRhoDraw.length == 2)
    assert(!outOfBoundsRhoDraw.data.sameElements(Array(0.0, 0.0)))
    assert(outOfBoundsRho.mahalanobisDistance == 0.0)
  }

}
