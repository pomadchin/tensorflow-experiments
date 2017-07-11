name := "tensorflow-experiments"

version := "0.0.1-SNAPSHOT"

scalaVersion := "2.11.11"

scalacOptions ++= Seq(
  "-deprecation",
  "-unchecked",
  "-Yinline-warnings",
  "-language:implicitConversions",
  "-language:reflectiveCalls",
  "-language:higherKinds",
  "-language:postfixOps",
  "-language:existentials",
  "-feature")

libraryDependencies ++= Seq(
  "org.tensorflow"   %  "tensorflow" % "1.2.1",
  "org.apache.spark" %% "spark-core" % "2.1.1" % "provided",
  "org.scalatest"    %%  "scalatest" % "3.0.3" % "test"
)

