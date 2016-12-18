import spark.implicits._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.SQLContext

val csv = sc.textFile("hdfs:///user/hadoop/2008.csv")

// Define class FlightDataRow with csv info
case class FlightDataRow(
    year: Int, month: Int, dayOfMonth: Int, dayOfWeek: Int,//4
    depTime: Int, crsDepTime: Int, arrTime:Int, crsArrTime: Int,//4
    uniqueCarrier: String, flightNum: Int, tailNum: String,//3
    actualElapsedTime: Int, crsElapsedTime: Int, airTime: Int, arrDelay: Int, depDelay: Int,//5
    origin: String, dest: String, distance: Int,//3
    taxiIn: Int, taxiOut: Int,//2
    cancelled: Int, cancellationCode: String, diverted: Int,//3    
    carrierDelay: Int, weatherDelay: Int, nasDelay: Int, securityDelay: Int, lateAircraftDelay: Int//5
    )

//To find the headers
val header = csv.first

//To remove the header
val data = csv.filter(_(0) != header(0))

var flightsDF = data.map((s: String) => {
        var fields = s.split(",")
        var i = 0
        //Para quitar los NA
        for(i <- 0 until fields.length){
          if(fields(i)=="NA"){
            fields(i) = "0"
          }
        }

        FlightDataRow(
          fields(0).toInt, fields(1).toInt, fields(2).toInt, fields(3).toInt,//4 
          fields(4).toInt, fields(5).toInt, fields(6).toInt, fields(7).toInt,//4
          fields(8), fields(9).toInt, fields(10),//3
          fields(11).toInt, fields(12).toInt, fields(13).toInt, fields(14).toInt, fields(15).toInt,//5
          fields(16), fields(17), fields(18).toInt,//3
          fields(19).toInt, fields(20).toInt,//2
          fields(21).toInt, fields(22), fields(23).toInt,//3
          fields(24).toInt, fields(25).toInt, fields(26).toInt, fields(27).toInt, fields(28).toInt//5
        )
    }).toDF

// Shows 15 elements 
flightsDF.show(15)

// SQL
flightsDF.createOrReplaceTempView("flights")

// Count number of rows
spark.sql("select count(1) from flights").collect()(0).getLong(0)

// Remove unused columns
flightsDF = flightsDF.drop("arrTime","actualElapsedTime","airTime",
    "taxiIn","diverted","carrierDelay","weatherDelay","nasDelay",
    "securityDelay","lateAircraftDelay")

flightsDF.show(15)

//To create a RDD of (label, features) pairs
/*val parsedData = data.map { line =>
    val parts = line.split(',')
    LabeledPoint(parts(0).toInt, Vectors.dense(parts(1).split(' ').map(_.toInt)))
    }.cache()
*/