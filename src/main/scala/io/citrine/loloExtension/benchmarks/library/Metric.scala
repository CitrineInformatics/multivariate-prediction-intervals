package io.citrine.loloExtension.benchmarks.library

import io.citrine.lolo.bags.CorrelationMethods
import io.citrine.lolo.validation.Merit
import io.citrine.loloExtension.validation.{NegativeLogProbabilityDensityNd, StandardConfidenceNd}

/** Metric used to evaluate the accuracy of the estimated covariance matrix. */
sealed trait Metric {
  def name: String
  def observational: Boolean
  def indices: Seq[Int]
  def makeMeritsMap: Map[String, Merit[Seq[Any]]]
}

case class NLPD(indices: Seq[Int], observational: Boolean) extends Metric {
  def name = "negative log probability density"

  override def makeMeritsMap: Map[String, Merit[Seq[Any]]] = Map(
    "Trivial" -> NegativeLogProbabilityDensityNd(indices, CorrelationMethods.Trivial, observational),
    "Training Data" -> NegativeLogProbabilityDensityNd(indices, CorrelationMethods.FromTraining, observational),
    "Bootstrap" -> NegativeLogProbabilityDensityNd(indices, CorrelationMethods.Bootstrap, observational),
    "Jackknife" -> NegativeLogProbabilityDensityNd(indices, CorrelationMethods.Jackknife, observational)
  )
}

case class StdConfidence(indices: Seq[Int], observational: Boolean, coverageLevel: Double = 0.683) extends Metric {
  def name = s"${coverageLevel * 100}% standard confidence"

  override def makeMeritsMap: Map[String, Merit[Seq[Any]]] = Map(
    "Trivial" -> StandardConfidenceNd(indices, CorrelationMethods.Trivial, observational, coverageLevel),
    "Training Data" -> StandardConfidenceNd(indices, CorrelationMethods.FromTraining, observational, coverageLevel),
    "Bootstrap" -> StandardConfidenceNd(indices, CorrelationMethods.Bootstrap, observational, coverageLevel),
    "Jackknife" -> StandardConfidenceNd(indices, CorrelationMethods.Jackknife, observational, coverageLevel)
  )
}