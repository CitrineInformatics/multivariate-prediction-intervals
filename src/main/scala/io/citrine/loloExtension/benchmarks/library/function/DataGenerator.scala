package io.citrine.loloExtension.benchmarks.library.function

trait DataGenerator {
  def name: String
  def generateData(n: Int, seed: Long): Vector[(Vector[Any], Double)]
}
