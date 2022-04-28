import sbt._

object Dependencies {
  lazy val junitVersion = "4.13.1"
  lazy val apacheMath3Version = "3.6.1"

  val deps = Seq(
    "junit" % "junit" % junitVersion % "test",
    "org.apache.commons" % "commons-math3" % apacheMath3Version
  )
}
