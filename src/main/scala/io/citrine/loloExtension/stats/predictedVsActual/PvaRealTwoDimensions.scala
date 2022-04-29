package io.citrine.loloExtension.stats.predictedVsActual

import io.citrine.lolo.bags.CorrelationMethods.CorrelationMethod
import io.citrine.lolo.bags.MultiTaskBaggedResult
import io.citrine.loloExtension.stats.predictedVsActual.RealPva.extractComponentByIndex


/**
  * A single predicted-vs-actual point in two real-valued dimensions.
  * The covariance matrix of the prediction has diagonals sigmaX**2 and sigmaY**2, and off-diagonals rho*sigmaX*sigmaY
  *
  * @param predictions the predictions along the two dimensions
  * @param actuals the actual values along the two dimensions
  * @param uncertainties the uncertainty estimates along the two dimensions
  * @param correlationCoefficient the correlation coefficient between the two uncertainties
  */
case class PvaRealTwoDimensions(
                                 predictions: (Double, Double),
                                 actuals: (Double, Double),
                                 uncertainties: (Double, Double),
                                 correlationCoefficient: Double
                               ) extends RealPva {

  override def prediction: Seq[Double] = Seq(predictions._1, predictions._2)

  override def actual: Seq[Double] = Seq(actuals._1, actuals._2)

  override def error: Seq[Double] = Seq(predictions._1 - actuals._1, predictions._2 - actuals._2)

  private lazy val normX = error.head / uncertainties._1
  private lazy val normY = error(1) / uncertainties._2
  private lazy val rhoSquared = correlationCoefficient * correlationCoefficient

  /**
    * Mahalanobis distance is a generalization of standardized distance for a multivariate normal distribution.
    * It corresponds to the Euclidean distance if space is rescaled such that the principal axes of the ellipsoid
    * formed by the covariance matrix each have length 1. In one dimension, this is (x - mu) / sigma.
    * In two dimensions, it is ((dx / sigmaX)**2 + (dy / sigmaY)**2 - 2 * rho * dx * dy / (sigmaX * sigmaY)) / (1 - rho**2)
    */
  def mahalanobisSquared: Double = {
    (math.pow(normX, 2.0) + math.pow(normY, 2.0) - 2 * correlationCoefficient * normX * normY) / (1 - rhoSquared)
  }
}

object PvaRealTwoDimensions {

  /**
    * Convert a prediction result into a sequence of PvaRealTwoDimensions objects, which are easier to compute metrics on.
    *
    * @param predictionResult multivariate prediction result
    * @param actual multivariate ground-truth values
    * @param i index of the first label to be compared
    * @param j index of the second label to be compared
    * @param method method to calculate correlation coefficient
    * @param observational whether or not to calculate the observational uncertainty
    * @return a case class containing the error and uncertainty for each predicted-actual pair
    */
  def makePva2d(
                 predictionResult: MultiTaskBaggedResult,
                 actual: Seq[Seq[Any]],
                 i: Int,
                 j: Int,
                 method: CorrelationMethod,
                 observational: Boolean
               ): Seq[PvaRealTwoDimensions] = {
    val allPredictions = predictionResult.getExpected()
    // get predictions
    val predictionsI = extractComponentByIndex(allPredictions, i)
    val predictionsJ = extractComponentByIndex(allPredictions, j)
    val predictions = predictionsI.zip(predictionsJ)
    // get actual
    val actualI = extractComponentByIndex(actual, i)
    val actualJ = extractComponentByIndex(actual, j)
    val actuals = actualI.zip(actualJ)
    // get terms of covariance matrix
    val allSigma = predictionResult.getUncertainty(observational).get
    val sigmaI = extractComponentByIndex(allSigma, i, Some(0.0))
    val sigmaJ = extractComponentByIndex(allSigma, j, Some(0.0))
    val uncertainties = sigmaI.zip(sigmaJ)
    val correlation = predictionResult.getUncertaintyCorrelationBuffet(i, j, method).get
    allPredictions.indices.map { i =>
      PvaRealTwoDimensions(predictions(i), actuals(i), uncertainties(i), correlation(i))
    }
  }
}