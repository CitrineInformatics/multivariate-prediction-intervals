package io.citrine.loloExtension.stats.predictedVsActual

import breeze.linalg.{DenseMatrix, DenseVector, NotConvergedException}
import io.citrine.lolo.bags.CorrelationMethods.CorrelationMethod
import io.citrine.lolo.bags.MultiTaskBaggedResult
import io.citrine.loloExtension.stats.predictedVsActual.RealPva.extractByIndices
import breeze.stats.distributions.MultivariateGaussian

case class PvaRealNDimensions(
                             prediction: Seq[Double],
                             actual: Seq[Double],
                             covariance: Array[Array[Double]]
                             ) extends RealPva {

  val numOutputs = prediction.length

  override def error: Seq[Double] = prediction.zip(actual).map {
    case (pred, act) => pred - act
  }

  def covarianceAsString: String = {
    covariance.map { row =>
      row.mkString("Array(", ", ", ")")
    }.mkString("Array(", ", ", ")")
  }

  private val predictionVector: DenseVector[Double] = DenseVector(prediction: _*)
  private val covarianceMatrix = DenseMatrix(covariance: _*)

  lazy val distribution: MultivariateGaussian = try {
    MultivariateGaussian(
      mean = predictionVector,
      covariance = covarianceMatrix
    )
  } catch {
    case _: NotConvergedException =>
      val uncorrelatedMatrix = covarianceMatrix.copy
      prediction.indices.foreach { i =>
        prediction.indices.foreach { j =>
          if (i != j) uncorrelatedMatrix.update(i, j, 0.0)
        }
      }
      MultivariateGaussian(predictionVector, uncorrelatedMatrix)
    case e: Throwable => throw e
  }

  def sample(n: Int): IndexedSeq[DenseVector[Double]] = distribution.sample(n)

  private lazy val covInv = breeze.linalg.pinv(distribution.covariance)
  private lazy val errorVector: DenseVector[Double] = DenseVector(error: _*)
  lazy val mahalanobisDistance: Double = math.sqrt(errorVector dot (covInv * errorVector))

  lazy val logPdf: Double = distribution.logPdf(DenseVector(actual: _*))
}

object PvaRealNDimensions {

  def makePva(
             predictionResult: MultiTaskBaggedResult,
             actual: Seq[Seq[Any]],
             indices: Seq[Int],
             method: CorrelationMethod,
             observational: Boolean = true
             ): Seq[PvaRealNDimensions] = {
    val numOutputs = indices.length
    val numPredictions = predictionResult.numPredictions

    val predictions = extractByIndices(predictionResult.getExpected(), indices)
    val actuals = extractByIndices(actual, indices)
    val allSigma = predictionResult.getUncertainty(observational).get
    val uncertainties = extractByIndices(allSigma, indices, Some(0.0))

    val correlationMatrices = Array.ofDim[Seq[Double]](numOutputs, numOutputs)
    for (i <- 0 until numOutputs - 1) {
      for (j <- i + 1 until numOutputs) {
        val rhoSeq =
          predictionResult.getUncertaintyCorrelationBuffet(indices(i), indices(j), method).getOrElse(Seq.fill(numPredictions)(0.0))
        correlationMatrices(i)(j) = rhoSeq
      }
    }

    Seq.tabulate(numPredictions) { predictionIndex =>
      val uncertainty = uncertainties(predictionIndex)
      val covarianceMatrix = Array.ofDim[Double](numOutputs, numOutputs)
      for (i <- 0 until numOutputs) {
        val sigma = uncertainty(i)
        covarianceMatrix(i)(i) = sigma * sigma
      }
      for (i <- 0 until numOutputs - 1) {
        for (j <- i + 1 until numOutputs) {
          val rho = correlationMatrices(i)(j)(predictionIndex)
          val covariance = rho * uncertainty(i) * uncertainty(j)
          covarianceMatrix(i)(j) = covariance
          covarianceMatrix(j)(i) = covariance
        }
      }
      PvaRealNDimensions(
        prediction = predictions(predictionIndex),
        actual = actuals(predictionIndex),
        covariance = covarianceMatrix
      )
    }
  }
}