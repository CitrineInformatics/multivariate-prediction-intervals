import Dependencies._

name := "uncertainty-correlation-study"
version := "0.1"
scalaVersion := "2.13.4"

// This commit corresponds to the feature/uncertainty-correlation-archived branch as of 2021-02-17
val loloVersion = "ab296d0"

lazy val root = (project in file(".")).dependsOn(loloProject)
lazy val loloProject = RootProject(uri("git://github.com/CitrineInformatics/lolo.git#%s".format(loloVersion)))

libraryDependencies ++= deps