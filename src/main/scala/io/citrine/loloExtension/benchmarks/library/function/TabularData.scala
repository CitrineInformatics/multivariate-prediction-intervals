package io.citrine.loloExtension.benchmarks.library.function

import io.citrine.loloExtension.benchmarks.BenchmarkUtils.extractInputsOutputsFromCSV

import scala.util.Random

sealed trait TabularData extends DataGenerator {
  def csvPath: String
  def outputHeaders: Vector[String]
  def outputIndex: Int

  lazy val fullData = extractInputsOutputsFromCSV(csvPath, outputHeaders)
  lazy val singleOutputData = fullData.map { case (inputs, outputs) =>
    (inputs, outputs(outputIndex).asInstanceOf[Double])
  }
  lazy val numRows = fullData.length

  override def generateData(n: Int, seed: Long): Vector[(Vector[Any], Double)] = {
    if (n > numRows) throw new IllegalArgumentException(s"Cannot request $n rows from table $name with only $numRows rows")
    new Random(seed).shuffle(singleOutputData).take(n)
  }

}

case class ThermoelectricsData(outputIndex: Int) extends TabularData {

  val csvPath = "thermoelectrics_clean.csv"

  val outputHeaders = Vector(
    "ZT",
    "Seebeck coefficient (uV/K)",
    "log Resistivity",
    "Power factor (W*m/K^2)",
    "Thermal conductivity (W/(m*K))"
  )

  val name = s"Thermoelectrics ${outputHeaders(outputIndex)}"
}

case class MechanicalPropertiesData(outputIndex: Int) extends TabularData {

  val csvPath = "mechanical_properties_clean.csv"

  val outputHeaders = Vector(
    "PROPERTY: YS (MPa)",
    "PROPERTY: Elongation (%)"
  )

  val name = s"Mechanical Properties ${outputHeaders(outputIndex)}"
}