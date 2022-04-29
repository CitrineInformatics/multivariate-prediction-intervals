package io.citrine.loloExtension.benchmarks.library

/** Parameter to vary as part of a scan. */
sealed trait VariedParameter {
  def name: String
}

case object TrainRho extends VariedParameter {
  def name: String = "linear training correlation"
}

case object TrainQuadraticFuzz extends VariedParameter {
  def name: String = "shift to decorrelate quadratic data"
}

case object Noise extends VariedParameter {
  def name: String = "observational noise level"
}

case object Bags extends VariedParameter {
  def name: String = "number of bags"
}

/** The number of bags can optionally be fixed. If it is None, then numBags = numTrain */
case class NumTraining(numBags: Option[Int] = None) extends VariedParameter {
  def name: String = "number of training rows"
}