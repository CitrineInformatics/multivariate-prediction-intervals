package io.citrine.loloExtension.benchmarks.library

/** A numeric objective in the context of sequential learning. */
sealed trait Objective {
  /** Whether or not a given value satisfies the objective. */
  def satisfies(x: Double): Boolean

  /** Human-readable name of the target variable. */
  val name: String

  /** Index of the target variable in a label vector. */
  val index: Int

  /** Convenience method for seeing if a label vector satisfies the objective. */
  def satisfiesByIndex(v: Vector[Any]): Boolean = satisfies(v(index).asInstanceOf[Double])
}

/** The objective is to exceed some threshold. */
case class GreaterThan(name: String, index: Int, threshold: Double) extends Objective {
  override def satisfies(x: Double): Boolean = x > threshold

  override def toString: String = s"$name > $threshold"
}

/** The objective is to be lower than some threshold. */
case class LessThan(name: String, index: Int, threshold: Double) extends Objective {
  override def satisfies(x: Double): Boolean = x < threshold

  override def toString: String = s"$name < $threshold"
}
