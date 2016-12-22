package flightDelay
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.log4j.{ Level, Logger }

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator, CrossValidatorModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.RegressionMetrics

/**
 * @author Borja Illescas and Oscar Fernández
 */
 object App {

  // Define class FlightDataRow with csv info
  // ArrDelay is a double in order to avoid problems with the model training
  case class FlightDataRow(
    year: Int, month: Int, dayOfMonth: Int, dayOfWeek: Int,//4, 0-3
    depTime: Int, crsDepTime: Int, arrTime:Int, crsArrTime: Int,//4, 4-7
    uniqueCarrier: String, flightNum: Int, tailNum: String,//3, 8-10
    actualElapsedTime: Int, crsElapsedTime: Int, airTime: Int, arrDelay: Double, depDelay: Int,//5, 11-15
    origin: String, dest: String, distance: Int,//3, 16-18
    taxiIn: Int, taxiOut: Int,//2, 19-20
    cancelled: Int, cancellationCode: String, diverted: Int,//3, 21-23
    carrierDelay: Int, weatherDelay: Int, nasDelay: Int, securityDelay: Int, lateAircraftDelay: Int//5, 24-28
    )

  def main(args: Array[String]) {

    Logger.getRootLogger().setLevel(Level.WARN)
    
    val conf = new SparkConf().setAppName("FlightDelayPredictor")
    val sc = new SparkContext(conf)

    //This has to be this way
    val sqlContext= new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val csv = sc.textFile("hdfs:///user/hadoop/data.csv")

    //To find the headers
    val header = csv.first

    //To remove the header
    val data = csv.filter(_(0) != header(0))

    var flightsDF = data
    .map((s: String) => {
      var fields = s.split(",")
      var i = 0

      // Removes NA
      for(i <- 0 until fields.length){
        if(fields(i)=="NA"){
          fields(i) = "0"
        }
      }

      // Converts hours to minutes
      var times = Array(fields(4).toInt, fields(5).toInt, fields(6).toInt, fields(7).toInt)//4

      for(i <- 0 until times.length){
        if(times(i)>99){
          times(i) = times(i)/100 * 60 +(times(i) - (times(i)/100)*100)
        }
      }

      FlightDataRow(
        fields(0).toInt, fields(1).toInt, fields(2).toInt, fields(3).toInt,//4 
        times(0), times(1), times(2), times(3),//4
        fields(8), fields(9).toInt, fields(10),//3
        fields(11).toInt, fields(12).toInt, fields(13).toInt, fields(14).toDouble, fields(15).toInt,//5
        fields(16), fields(17), fields(18).toInt,//3
        fields(19).toInt, fields(20).toInt,//2
        fields(21).toInt, fields(22), fields(23).toInt,//3
        fields(24).toInt, fields(25).toInt, fields(26).toInt, fields(27).toInt, fields(28).toInt//5
        )
      }).toDF()

    // Removes cancelled flights
    flightsDF = flightsDF.filter(flightsDF("cancelled")==="0")

    // Remove unused columns
    flightsDF = flightsDF.drop("arrTime","actualElapsedTime","airTime",
      "taxiIn","diverted","carrierDelay","weatherDelay","nasDelay",
      "securityDelay","lateAircraftDelay")

    // Remove categorical string variables
    flightsDF = flightsDF.drop("uniqueCarrier","tailNum",
      "origin", "dest", "cancellationCode")

    // Remove unwanted variables
    flightsDF = flightsDF.drop("cancelled", "year", "month", "dayOfMonth", "dayOfWeek")

    //Prepare the data for the model
    val assembler = new VectorAssembler()
    .setInputCols(Array("depTime", "crsDepTime", "crsArrTime", "crsElapsedTime", "depDelay", "distance", "taxiOut"))
    .setOutputCol("features")

    flightsDF = assembler.transform(flightsDF)

    //Just in case we only leave he important columns (features and label)
    flightsDF = flightsDF
    .withColumn("features", flightsDF("features"))
    .withColumn("label", flightsDF("arrDelay"))
    .select("features", "label")

    // Split dataset for training and testing
    val splits = flightsDF.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Generate linear regression
    val lr = new LinearRegression()
    .setFeaturesCol("features")
    .setLabelCol("label")
    .setMaxIter(10)

    // Prepare parameters grid to generate several models
    val paramGrid = new ParamGridBuilder()
    .addGrid(lr.regParam, Array(0.1, 0.01))
    .addGrid(lr.fitIntercept)
    .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
    .build()

    // Prepare Cross Validation to obtain best model
    val cv = new CrossValidator()
    .setEstimator(lr)
    .setEvaluator(new RegressionEvaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(3)

    //Train the model with cross validation
    val lrModel = cv.fit(training)

    // This is only if you want to save time in later executions
    // Saves the model to hdfs
    //lrModel.save("hdfs:///user/hadoop/bestLinearRegressionSpark")

    // Loads the model from hdfs
    //val lrModelLoaded = CrossValidatorModel.load("hdfs:///user/hadoop/bestLinearRegressionSpark")
    //val bestModel = lrModelLoaded.bestModel

    // Obtain the best model
    val bestModel = lrModel.bestModel

    // Test the model
    val resultDF = bestModel.transform(test)

    // DF to rdd[(double, double)] -> rdd[(Prediction, labels)]
    val predictionAndLabels = resultDF.rdd.map[(Double,Double)] (row=>{(row.getDouble(2), row.getDouble(1))})

    // Instantiate metrics object
    val metrics = new RegressionMetrics(predictionAndLabels)

    // Print the different metrics
    // Squared error
    println(s"MSE = ${metrics.meanSquaredError}")
    println(s"RMSE = ${metrics.rootMeanSquaredError}")

    // R-squared
    println(s"R-squared = ${metrics.r2}")

    // Mean absolute error
    println(s"MAE = ${metrics.meanAbsoluteError}")

    // Explained variance
    println(s"Explained variance = ${metrics.explainedVariance}")
  }

}