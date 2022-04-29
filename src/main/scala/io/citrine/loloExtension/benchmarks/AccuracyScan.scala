package io.citrine.loloExtension.benchmarks

import io.citrine.lolo.PredictionResult
import io.citrine.lolo.validation.Merit
import io.citrine.loloExtension.benchmarks.BenchmarkUtils.{saveChart, writeCSVRow}
import io.citrine.loloExtension.benchmarks.library.VariedParameter
import org.knowm.xchart.XYChart

import scala.util.Random

/** A generic trait for producing merit scan figures. */
trait AccuracyScan[T] {

  /**
    * Run trials and measure figures of merit, producing a chart and a csv.
    * @param fname path to save the figure and csv
    * @param rng random number generator, to make it mostly reproducible. Because the trees are trained in parallel
    *            and we do not have splittable random numbers, it is impossible to make the results completely reproducible.
    *            But the data generation, application of noise, and Poisson draws for bagging are reproducible.
    */
  def runTrialsAndSave(fname: String, rng: Random): Unit = {
    val chart = makeChart(rng: Random)
    saveRawData(chart, fname)
    saveChart(chart, fname)
  }

  /** Run a single trial for a given parameter value. */
  def runTrial(parameterValue: Double, rng: Random): (PredictionResult[T], Seq[T])

  /** The parameter that is being varied */
  def variedParameter: VariedParameter

  /** Given a parameter value, construct an object that holds the full set of parameters describing this trial. */
  def makeParameterSet(parameterValue: Double): ParameterSet

  /** Sequence of parameter values to scan over */
  def parameterValues: Seq[Double]

  /** Number of independent trials to run and average the results over. */
  def numTrials: Int

  /** Merits that are measured for each trial (the key is a human-readable label) */
  def meritsMap: Map[String, Merit[T]]

  private def saveRawData(chart: XYChart, fname: String): Unit = {
    val path = fname + ".csv"
    writeCSVRow(path, parameterSets.head.headers, append = false)
    parameterSets.zipWithIndex.foreach { case (parameterSet, index) =>
      chart.getSeriesMap.forEach { case (key, series) =>
        val y = series.getYData.apply(index)
        val yErr = series.getExtraValues.apply(index)
        val rowData = parameterSet.makeRow(key, y, yErr)
        writeCSVRow(path, rowData, append = true)
      }
    }
  }

  private def makeChart(rng: Random): XYChart = {
    Merit.plotMeritScan(
      parameterName = variedParameter.name,
      parameterValues = parameterValues,
      merits = meritsMap,
      rng = new Random(0L)
    )(pvaBuilder(rng))
  }

  private def pvaBuilder(rng: Random): Double => Iterator[(PredictionResult[T], Seq[T])] =
    parameterValue => Iterator.tabulate(numTrials) { _ =>
      val thisRng = new Random(rng.nextLong())
      runTrial(parameterValue, thisRng)
    }

  private def parameterSets: Seq[ParameterSet] = parameterValues.map(makeParameterSet)
}

trait ParameterSet {

  def headers: Seq[String]

  def makeRow(label: String, y: Double, yErr: Double): Seq[Any]

}
