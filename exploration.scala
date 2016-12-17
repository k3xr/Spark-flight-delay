import spark.implicits._

val myRDD3 = sc.textFile("hdfs:///user/hadoop/2008.csv")