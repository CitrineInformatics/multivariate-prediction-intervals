package io.citrine.loloExtension.benchmarks.library.function

import io.citrine.lolo.stats.functions.Friedman
import io.citrine.loloExtension.stats.StatsUtils.generateTrainingData

/** Underlying function from which to draw training and test data. */
sealed trait AnalyticFunction extends DataGenerator {
  def function: Seq[Double] => Double
  def domain: Seq[(Double, Double)]
  def numCols: Int = domain.length

  override def generateData(n: Int, seed: Long): Vector[(Vector[Any], Double)] = {
    generateTrainingData(
      rows = n,
      cols = numCols,
      function = function,
      noise = 0.0, // noise can be added later
      seed = seed
    )
  }
}

case class FriedmanSilvermanFunction(columns: Int) extends AnalyticFunction {
  def name = s"Friedman-Silverman $columns columns"
  def function: Seq[Double] => Double = Friedman.friedmanSilverman
  def domain: Seq[(Double, Double)] = Seq.fill(columns)((0.0, 1.0))
}

case class FriedmanGrosseFunction(columns: Int) extends AnalyticFunction {
  def name = s"Friedman-Grosse $columns columns"
  def function: Seq[Double] => Double = Friedman.friedmanGrosseSilverman
  def domain: Seq[(Double, Double)] = Seq.fill(columns)((0.0, 1.0))
}

/** Output is quadratically correlated with Friedman-Grosse function */
case class FriedmanGrosseFunctionQuadratic(columns: Int) extends AnalyticFunction {
  def name = s"Friedman-Grosse quadratic $columns columns"
  private val shift = 15.0 // roughly the center of the Friedman-Grosse function's range
  private val scale = 0.2 // rescale so that the outputs are of the same order
  def function: Seq[Double] => Double = { x: Seq[Double] =>
    val y = Friedman.friedmanGrosseSilverman(x) - shift
    y * y * scale
  }
  def domain: Seq[(Double, Double)] = Seq.fill(columns)((0.0, 1.0))
}

case class DoubleTopHat() extends AnalyticFunction {
  def name = s"Double top-hat"
  def domain = Seq((-1.0, 1.0))
  def function: Seq[Double] => Double = { x: Seq[Double] =>
    val x0 = x.padTo(1, 0.0).head
    if (math.abs(x0) < 0.33) {
      1.0
    } else if (math.abs(x0) < 0.67) {
      0.5
    } else {
      0.0
    }
  }
}

case class SimpleCubic() extends AnalyticFunction {
  def name = "Simple cubic"
  def domain = Seq((-1.0, 1.0))
  def function: Seq[Double] => Double = { x: Seq[Double] =>
    val x0 = x.padTo(1, 0.0).head
    x0 * x0 * x0
  }
}