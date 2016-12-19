package flightDelay
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.log4j.{ Level, Logger }

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.VectorAssembler

/**
 * @author ${user.name}
 */
object App {

  // Define class FlightDataRow with csv info
  case class FlightDataRow(
    year: Int, month: Int, dayOfMonth: Int, dayOfWeek: Int, //4, 0-3
    depTime: Int, crsDepTime: Int, arrTime: Int, crsArrTime: Int, //4, 4-7
    uniqueCarrier: String, flightNum: Int, tailNum: String, //3, 8-10
    actualElapsedTime: Int, crsElapsedTime: Int, airTime: Int, arrDelay: Int, depDelay: Int, //5, 11-15
    origin: String, dest: String, distance: Int, //3, 16-18
    taxiIn: Int, taxiOut: Int, //2, 19-20
    cancelled: Int, cancellationCode: String, diverted: Int, //3, 21-23
    carrierDelay: Int, weatherDelay: Int, nasDelay: Int, securityDelay: Int, lateAircraftDelay: Int //5, 24-28
    )

  def main(args: Array[String]) {

    Logger.getRootLogger().setLevel(Level.WARN)
    
    val conf = new SparkConf().setAppName("FlightDelayPredictor")
    val sc = new SparkContext(conf)

    //Se tiene que hacer asi si no peta
    val sqlContext= new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val csv = sc.textFile("hdfs:///user/hadoop/2008.csv")

    //To find the headers
    val header = csv.first

    //To remove the header
    val data = csv.filter(_(0) != header(0))

    var flightsDF = data
      .map((s: String) => {
        var fields = s.split(",")
        var i = 0
        //Para quitar los NA
        for (i <- 0 until fields.length) {
          if (fields(i) == "NA") {
            fields(i) = "0"
          }
        }
        FlightDataRow(
          fields(0).toInt, fields(1).toInt, fields(2).toInt, fields(3).toInt, //4 
          fields(4).toInt, fields(5).toInt, fields(6).toInt, fields(7).toInt, //4
          fields(8), fields(9).toInt, fields(10), //3
          fields(11).toInt, fields(12).toInt, fields(13).toInt, fields(14).toInt, fields(15).toInt, //5
          fields(16), fields(17), fields(18).toInt, //3
          fields(19).toInt, fields(20).toInt, //2
          fields(21).toInt, fields(22), fields(23).toInt, //3
          fields(24).toInt, fields(25).toInt, fields(26).toInt, fields(27).toInt, fields(28).toInt //5
          )
      }).toDF()

    //Si el vuelo esta cancelado no nos interesa, asi que cogemos los que estan sin cancelar
    flightsDF = flightsDF.filter(flightsDF("cancelled") === "0")

    // Shows 15 elements 
    flightsDF.show(15)

    // Realmente no lo estamos usando asique lo dejo comentado
    // SQL 
    //flightsDF.createOrReplaceTempView("flights")

    // Count number of rows
    //spark.sql("select count(1) from flights").collect()(0).getLong(0)

    // Remove unused columns
    flightsDF = flightsDF.drop("arrTime", "actualElapsedTime", "airTime",
      "taxiIn", "diverted", "carrierDelay", "weatherDelay", "nasDelay",
      "securityDelay", "lateAircraftDelay")

    //flightsDF.show(15)

    // Remove categorical string variables
    flightsDF = flightsDF.drop("uniqueCarrier", "tailNum",
      "origin", "dest", "cancellationCode")

    //flightsDF.show(15)

    //remove unwanted variables
    flightsDF = flightsDF.drop("cancelled", "year")
    flightsDF.show(15)

    val splits = flightsDF.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val assembler = new VectorAssembler()
      .setInputCols(Array("month", "dayOfMonth", "dayOfWeek", "depTime", "crsDepTime", "crsArrTime", "flightNum", "crsElapsedTime", "depDelay", "distance", "taxiOut"))
      .setOutputCol("features")

    val output = assembler.transform(training)

    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("arrDelay")
      .setMaxIter(10)

    val lrModel = lr.fit(output)

    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
  }

}