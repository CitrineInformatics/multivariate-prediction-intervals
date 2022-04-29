package io.citrine.loloExtension.benchmarks

import org.knowm.xchart.BitmapEncoder.BitmapFormat
import org.knowm.xchart.{BitmapEncoder, XYChart}

import java.io.{BufferedWriter, File, FileWriter}
import scala.util.Try

object BenchmarkUtils {

  /**
    * Write a sequence of values as a row of a CSV
    *
    * @param path path to CSV file
    * @param entries values to be written
    * @param append whether or not to append if file exists already
    */
  def writeCSVRow(path: String, entries: Seq[Any], append: Boolean): Unit = {
    val text = entries.map(_.toString).mkString(start = "", sep = ",", end = "\n")
    writeText(path, text, append)
  }

  /**
    * Write text to a file
    *
    * @param path to file
    * @param text to be written
    * @param append whether or not to append if the file exists already
    */
  def writeText(path: String, text: String, append: Boolean): Unit = {
    makeDirectories(path)
    val bw = new BufferedWriter(new FileWriter(new File(path), append))
    bw.write(text)
    bw.close()
  }

  /** Given a file path, create the necessary directories. */
  private def makeDirectories(filePath: String): Unit = {
    val directory = new File(filePath).getParentFile
    if (!directory.exists()) directory.mkdirs()
  }

  /** Save chart as a png */
  def saveChart(chart: XYChart, fname: String): Unit = BitmapEncoder.saveBitmap(chart, fname + ".png", BitmapFormat.PNG)

  /**
    * Read a CSV file
    *
    * @param name path to the CSV file
    * @return CSV as a sequence of vectors, where each outer element is a row and each inner element is a cell value
    */
  def readCsv(name: String): Seq[Vector[Any]] = {
    val stream = getClass.getClassLoader.getResourceAsStream(name)
    val bs = scala.io.Source.fromInputStream(stream)
    val res = bs.getLines().flatMap{line =>
      Try(line.split(",").map(_.trim).map { token =>
        try {
          token.toDouble
        } catch {
          case _: Throwable if token == "NaN" => Double.NaN
          case _: Throwable if token.nonEmpty => token
        }
      }.toVector).toOption
    }.toVector
    bs.close()
    res
  }

  /**
    * Read a CSV and parse its columns into inputs and outputs.
    * The first row of the CSV is the header row.
    * All columns that are not marked as outputs are assumed to be inputs.
    * All rows of the CSV must have the same length.
    *
    * @param path to the CSV file
    * @param outputHeaders vector of the headers corresponding to the output columns.
    *                      If a header is not found, an IllegalArgumentException is thrown.
    * @return tuple of (inputs, outputs), each a sequence of vectors where each vector corresponds to one row.
    */
  def extractInputsOutputsFromCSV(path: String, outputHeaders: Vector[String]): Vector[(Vector[Any], Vector[Any])] = {
    val csv = BenchmarkUtils.readCsv(path).toVector
    val outputIndices = outputHeaders.map { header =>
      val index = csv.head.indexOf(header)
      if (index == -1) throw new IllegalArgumentException(s"Output \'${header}\' not found in header row")
      index
    }
    val transposedData = csv.drop(1).transpose

    val outputs = outputIndices.map(transposedData).transpose
    val inputs = transposedData.zipWithIndex.filterNot { case (_, index) =>
      outputIndices.contains(index)
    }.map(_._1).transpose
    inputs.zip(outputs)
  }
}
