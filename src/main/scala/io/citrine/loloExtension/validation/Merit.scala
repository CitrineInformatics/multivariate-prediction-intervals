package io.citrine.loloExtension.validation

import io.citrine.lolo.PredictionResult
import io.citrine.lolo.bags.CorrelationMethods.CorrelationMethod
import io.citrine.lolo.bags.UncertaintyMethods.UncertaintyMethod
import io.citrine.lolo.bags.{BaggedRealResult, MultiTaskBaggedResult, UncertaintyMethods}
import io.citrine.lolo.validation.Merit
import io.citrine.loloExtension.stats.StatsUtils
import io.citrine.loloExtension.stats.predictedVsActual.PvaRealNDimensions
import breeze.stats.distributions.ChiSquared

import scala.util.Random


/**
  * Negative log probability density.
  * For error err and uncertainty sigma, this is ln(sqrt(2*pi)) + ln(sigma) + 1/2 * err**2
  * @param method to compute the standard deviation
  */
case class NegativeLogProbabilityDensity(method: UncertaintyMethod = UncertaintyMethods.Bootstrap) extends Merit[Double] {
  override def evaluate(predictionResult: PredictionResult[Double], actual: Seq[Double], rng: Random): Double = {
    val uncertaintyOption = Merit.uncertaintiyOption(predictionResult, method)
    if (uncertaintyOption.isEmpty) return Double.PositiveInfinity
    val nlpd = (predictionResult.getExpected(), uncertaintyOption.get, actual).zipped.map {
      case (pred, sigma: Double, actual) =>
        val normError = (pred - actual) / sigma
        math.log(math.sqrt(2 * math.Pi) * sigma) + 0.5 * math.pow(normError, 2)
    }
    StatsUtils.median(nlpd)
  }
}

/**
  * The fraction of predictions that fall within one standard deviation of the prediction.
  * @param method to compute the standard deviation
  */
case class StandardConfidenceExtension(method: UncertaintyMethod = UncertaintyMethods.Bootstrap) extends Merit[Double] {
  override def evaluate(predictionResult: PredictionResult[Double], actual: Seq[Double], rng: Random = Random): Double = {
    val uncertaintyOption = Merit.uncertaintiyOption(predictionResult, method)
    if (uncertaintyOption.isEmpty) return 0.0
    (predictionResult.getExpected(), uncertaintyOption.get, actual).zipped.count {
      case (x, sigma: Double, y) => Math.abs(x - y) < sigma
    } / predictionResult.getExpected().size.toDouble
  }
}

/**
  * Root mean square of (error divided by predicted uncertainty).
  * @param rescale Rescaling factor to apply to the final result (useful if the uncertainties are to be uniformly rescaled)
  * @param method to compute the uncertainty
  */
case class StandardErrorExtension(rescale: Double = 1.0, method: UncertaintyMethod = UncertaintyMethods.Bootstrap) extends Merit[Double] {
  override def evaluate(predictionResult: PredictionResult[Double], actual: Seq[Double], rng: Random = Random): Double = {
    val uncertaintyOption = Merit.uncertaintiyOption(predictionResult, method)
    if (uncertaintyOption.isEmpty) return Double.PositiveInfinity
    val standardized = (predictionResult.getExpected(), uncertaintyOption.get, actual).zipped.map {
      case (x, sigma: Double, y) => (x - y) / sigma
    }
    rescale * Math.sqrt(standardized.map(Math.pow(_, 2.0)).sum / standardized.size)
  }
}

/**
  * Negative Log Probability Density (NLPD) in N dimensions
  * NLPD is calculated for each point and the median value is returned.
  *
  * @param indices of the labels to be compared
  * @param method method to calculate correlation coefficient
  * @param observational whether or not to calculate the observational uncertainty
  */
case class NegativeLogProbabilityDensityNd(indices: Seq[Int], method: CorrelationMethod, observational: Boolean) extends Merit[Seq[Any]] {
  override def evaluate(predictionResult: PredictionResult[Seq[Any]], actual: Seq[Seq[Any]], rng: Random): Double = {
    val pvas = PvaRealNDimensions.makePva(predictionResult.asInstanceOf[MultiTaskBaggedResult], actual, indices, method, observational)
    val nlpds = pvas.map(pva => -1 * pva.logPdf)
    StatsUtils.median(nlpds)
  }
}

/**
  * Standard confidence in N dimensions. This is the fraction of predictions that fall within a given uncertainty band.
  * Uses the Mahalanobis distance to calculate the distance from the prediction to the actual point.
  *
  * @param indices of the labels to be compared
  * @param method method to calculate correlation coefficient
  * @param observational whether or not to calculate the observational uncertainty
  * @param confidenceLevel the fraction of predictions that are expected to be within this distance.
  *                        By default it is 68.3%, corresponding to one standard deviation in one dimension.
  */
case class StandardConfidenceNd(
                                 indices: Seq[Int],
                                 method: CorrelationMethod,
                                 observational: Boolean,
                                 confidenceLevel: Double = 0.683
                               ) extends Merit[Seq[Any]] {
  override def evaluate(predictionResult: PredictionResult[Seq[Any]], actual: Seq[Seq[Any]], rng: Random): Double = {
    val pvas = PvaRealNDimensions.makePva(predictionResult.asInstanceOf[MultiTaskBaggedResult], actual, indices, method, observational)
    // r^2, where r is the Mahalanobis distance, follows a chi-squared distribution with d degrees of freedom, where d is the dimensionality.
    val cutoffDistance = math.sqrt(ChiSquared(indices.length).inverseCdf(confidenceLevel))
    pvas.count(_.mahalanobisDistance < cutoffDistance).toDouble / pvas.length
  }
}

object Merit {
  def uncertaintiyOption(predictionResult: PredictionResult[Double], method: UncertaintyMethod): Option[Seq[Double]] = {
    predictionResult match {
      case p: BaggedRealResult => p.getUncertaintyBuffet(method)
      case p: PredictionResult[Double] => p.getUncertainty().asInstanceOf[Option[Seq[Double]]]
    }
  }
}