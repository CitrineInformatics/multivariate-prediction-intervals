import Dependencies._

name := "multivariate-prediction-intervals"
version := "0.1"
scalaVersion := "2.13.4"

// This commit corresponds to the feature/uncertainty-correlation-archived branch as of 2022-07-28
val loloVersion = "911e078"

lazy val root = (project in file(".")).dependsOn(loloProject)
lazy val loloProject = RootProject(uri("https://github.com/CitrineInformatics/lolo.git#%s".format(loloVersion)))

libraryDependencies ++= deps