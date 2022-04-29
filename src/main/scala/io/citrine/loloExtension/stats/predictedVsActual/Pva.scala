package io.citrine.loloExtension.stats.predictedVsActual

trait Pva[T] {

  def prediction: Seq[T]

  def actual: Seq[T]

}

trait RealPva extends Pva[Double] {

  def error: Seq[Double]
}

object RealPva {

  /**
    * A convenience method for slicing a two dimensional data structure
    *
    * @param data a sequence of sequences
    * @param index to slice (must correspond to real-valued data)
    * @param default default to use in case the values are Option[Double]
    * @return
    */
  def extractComponentByIndex(data: Seq[Seq[Any]], index: Int, default: Option[Double] = None): Seq[Double] = {
    extractByIndices(data, Seq(index), default).flatten
  }

  def extractByIndices(data: Seq[Seq[Any]], indices: Seq[Int], default: Option[Double] = None): Seq[Seq[Double]] = {
    data.map { row =>
      indices.map { i =>
        row(i) match {
          case x: Double => x
          case x: Option[Double] => x.getOrElse(default.get)
        }
      }
    }
  }
}