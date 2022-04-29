package io.citrine.loloExtension.validation

import io.citrine.lolo.PredictionResult
import org.apache.commons.math3.distribution.MultivariateNormalDistribution
import org.apache.commons.math3.random.MersenneTwister
import org.junit.Test

import scala.util.Random

class MeritTest {

  /**
    * Generate test data by adding Gaussian noise to a uniformly distributed response
    *
    * Uncertainty estimates are also produced.  The degree of correlation between the uncertainty estimate and the
    * error is configurable.
    *
    * @param noiseScale             the scale of the errors added to the responses
    * @param uncertaintyCorrelation the degree of correlation between the predicted uncertainty and the local error scale
    * @param batchSize              the number of predictions per result
    * @param numBatch               the number of prediction results produced
    * @return predicted-vs-actual data in the format expected by Metric.estimate
    */
  private def getNormalPVA(
                            noiseScale: Double = 1.0,
                            uncertaintyCorrelation: Double = 0.0,
                            batchSize: Int = 32,
                            numBatch: Int = 1,
                            rng: Random = Random
                          ): Iterable[(PredictionResult[Double], Seq[Double])] = {
    val maximumCorrelation = 0.999

    val noiseVariance = noiseScale * noiseScale
    val noiseUncertaintyCovariance =
      noiseVariance * Math.min(uncertaintyCorrelation, maximumCorrelation) // avoid singular matrices
    val errorDistribution = new MultivariateNormalDistribution(
      new MersenneTwister(rng.nextLong()),
      Array(0.0, 0.0),
      Array(Array(noiseVariance, noiseUncertaintyCovariance), Array(noiseUncertaintyCovariance, noiseVariance))
    )

    Seq.fill(numBatch) {
      val pua = Seq.fill(batchSize) {
        val y = rng.nextDouble()
        val draw = errorDistribution.sample().toSeq
        val error: Double = draw(0) * rng.nextGaussian()
        val uncertainty = if (uncertaintyCorrelation >= maximumCorrelation) {
          Math.abs(draw(0))
        } else {
          Math.abs(draw(1))
        }
        (y + error, uncertainty, y)
      }
      val predictionResult = new PredictionResult[Double] {
        override def getExpected(): Seq[Double] = pua.map(_._1)

        override def getUncertainty(includeNoise: Boolean = true): Option[Seq[Any]] = Some(pua.map(_._2))
      }
      (predictionResult, pua.map(_._3))
    }
  }

  /**
    * Test that the standard confidence comes out right
    */
  @Test
  def testStandardConfidence(): Unit = {
    val rng = new Random(34578L)
    val pva = getNormalPVA(uncertaintyCorrelation = 1.0, batchSize = 256, numBatch = 32, rng = rng)
    val expected = 0.68
    val (confidence, uncertainty) = StandardConfidenceExtension().estimate(pva, rng = rng)
    assert(Math.abs(confidence - expected) < 3 * uncertainty, "Confidence estimate was not accurate enough")
    assert(uncertainty < 0.05, s"Confidence estimate was not precise enough")
  }

  /**
    * Test that the standard error comes out right
    */
  @Test
  def testStandardError(): Unit = {
    val rng = new Random(34578L)
    val pva = getNormalPVA(noiseScale = 0.01, uncertaintyCorrelation = 1.0, batchSize = 256, numBatch = 32, rng = rng)
    val expected = 1.0
    val (error, uncertainty) = StandardErrorExtension().estimate(pva, rng = rng)
    assert(Math.abs(error - expected) < 3 * uncertainty, "Standard error estimate was not accurate enough")
    assert(uncertainty < 0.05, s"Standard error estimate was not precise enough")
  }

  @Test
  def testNLPD(): Unit = {
    val rng = new Random(34578L)
    val pva = getNormalPVA(noiseScale = 1.0, uncertaintyCorrelation = 1.0, batchSize = 256, numBatch = 32, rng = rng)
    val (nlpd, uncertainty) = NegativeLogProbabilityDensity().estimate(pva, rng = rng)
    // The optimal nlpd occurs when sigma ~ error. With a noise scale of 1 and a high correlation we therefore expect
    // ln(sqrt(2 * pi)) + ln(sigma) + 0.5 * error**2/sigam**2 to be about ln(sqrt(2*pi)) + 1/2 which is less than 1.5
    assert(nlpd < 1.5, "error estimate was not accurate enough")
    assert(uncertainty < 0.05, "nlpd estimate was not precise enough")
  }

}
