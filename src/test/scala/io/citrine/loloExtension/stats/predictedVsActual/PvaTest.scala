package io.citrine.loloExtension.stats.predictedVsActual

import org.junit.Test

class PvaTest {

  /** Test behavior of extraction convenience methods. */
  @Test
  def testComponentExtraction(): Unit = {
    val testData = Seq(
      Seq(0.0, "blue", Some(3.5)),
      Seq(1.0, "green", Some(5.5)),
      Seq(2.0, "yellow", None),
    )
    assert(
      RealPva.extractComponentByIndex(testData, index = 0) == Seq(0.0, 1.0, 2.0)
    )
    assert(
      RealPva.extractComponentByIndex(testData, index = 2, default = Some(7.0)) == Seq(3.5, 5.5, 7.0)
    )
    assert(
      RealPva.extractByIndices(testData, indices = Seq(2, 0), default = Some(7.0)) ==
        Seq(Seq(3.5, 0.0), Seq(3.5, 1.0), Seq(7.0, 2.0))
    )
  }

}
