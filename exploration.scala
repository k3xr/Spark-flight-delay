import spark.implicits._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator, CrossValidatorModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.VectorAssembler

val csv = sc.textFile("hdfs:///user/hadoop/2008.csv")

// Define class FlightDataRow with csv info
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

// Find the headers
val header = csv.first

// Remove the header
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

        //Convertimos de hhmm a minutos directamente
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

// Remove cancelled flights
flightsDF = flightsDF.filter(flightsDF("cancelled")==="0")

// Shows 15 elements 
//flightsDF.show(15)

// SQL 
// flightsDF.createOrReplaceTempView("flights")
// Count number of rows
//spark.sql("select count(1) from flights").collect()(0).getLong(0)

// Remove unused columns
flightsDF = flightsDF.drop("arrTime","actualElapsedTime","airTime",
    "taxiIn","diverted","carrierDelay","weatherDelay","nasDelay",
    "securityDelay","lateAircraftDelay")

// Remove categorical string variables
flightsDF = flightsDF.drop("uniqueCarrier","tailNum",
    "origin", "dest", "cancellationCode")

// Remove unwanted variables
flightsDF = flightsDF.drop("cancelled", "year")
//flightsDF.show(15)

val assembler = new VectorAssembler()
 //.setInputCols(Array("month", "dayOfMonth", "dayOfWeek", "depTime", "crsDepTime", "crsArrTime", "flightNum", "crsElapsedTime", "depDelay", "distance", "taxiOut"))
 .setInputCols(Array("depTime", "crsDepTime", "crsArrTime", "crsElapsedTime", "depDelay", "distance", "taxiOut"))
 .setOutputCol("features")

flightsDF = assembler.transform(flightsDF)

var finalDF = flightsDF
  .withColumn("features", flightsDF("features"))
  .withColumn("label", flightsDF("arrDelay"))
  .select("features", "label")

// Split dataset for training and testing
val splits = finalDF.randomSplit(Array(0.6, 0.4), seed =11L)
val training = splits(0).cache()
val test = splits(1)

// Generamos la regresion linear
val lr = new LinearRegression()
  .setFeaturesCol("features")
  .setLabelCol("label")
  .setMaxIter(10)

// Preparamos el grid de distintos parametros para generar varios modelos
val paramGrid = new ParamGridBuilder()
  .addGrid(lr.regParam, Array(0.1, 0.01))
  .addGrid(lr.fitIntercept)
  .addGrid(lr.elasticNetParam, Array0(.0, 0.5, 1.0))
  .build()


// Preparamos la Cross Validation para obtener el mejor modelo
val cv = new CrossValidator()
  .setEstimator(lr)
  .setEvaluator(new RegressionEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)

val lrModel = cv.fit(training)
//Guardar el modelo
lrModel.save("hdfs:///user/hadoop/bestLinearRegressionSpark")

//Cargar el modelo
//val lrModelLoaded = CrossValidatorModel.load("hdfs:///user/hadoop/bestLinearRegressionSpark")
//val bestModel = lrModelLoaded.bestModel

val bestModel = lrModel.bestModel

// Test the model
val resultDF = bestModel.transform(test)

//resultDF.show(40)

//Esto peta: Si consigues pasar el resultDF a un rdd que sea (Double, Double) de prediction, label (o al reves, no se) ya esta
predictionAndLabels = resultDF.rdd.map[(Double,Double)] (row=>{(row.getAs(Double)("prediction"), row.getAs(Double)("label"))})

//No queda claro cual alternativa es me salen las dos en internet hay que probar
val metrics = new RegressionMetrics(predictionAndLabels)
// Instantiate metrics object
val metrics = new RegressionMetrics(valuesAndPreds)

// Squared error
println(s"MSE = ${metrics.meanSquaredError}")
println(s"RMSE = ${metrics.rootMeanSquaredError}")

// R-squared
println(s"R-squared = ${metrics.r2}")

// Mean absolute error
println(s"MAE = ${metrics.meanAbsoluteError}")

// Explained variance
println(s"Explained variance = ${metrics.explainedVariance}")


/*
println(s"Coefficients: ${lrModel.bestModel.coefficients} Intercept: ${lrModel.intercept}")
val trainingSummary = lrModel.summary
println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"r2: ${trainingSummary.r2}")*/

///Siempre falla: [41,1.0,[1.0,1.0,2.0,2.0,2355.0,755.0,166.0,300.0,7.0,2454.0,34.0]]