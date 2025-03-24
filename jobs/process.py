
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, to_date, month, dayofweek, unix_timestamp, rank
from pyspark.sql.window import Window
import os
from datetime import datetime, timedelta


def trip_duration(df):
    df = df.withColumn("start_time", unix_timestamp("start_time", "yyyy-MM-dd HH:mm:ss").cast("timestamp"))
    df = df.withColumn("end_time", unix_timestamp("end_time", "yyyy-MM-dd HH:mm:ss").cast("timestamp"))
    df = df.withColumn("tripduration", col("tripduration").cast("double"))
    avg_trip_duration = df.groupBy(to_date("start_time").alias("date")).agg(avg("tripduration").alias("avg_tripduration"))
    avg_trip_duration = avg_trip_duration.orderBy("date")
    return avg_trip_duration

def trips_per_day(df):
    trips_per_day = df.groupBy(to_date("start_time").alias("date")).agg(count("trip_id").alias("trips_count"))
    trips_per_day = trips_per_day.orderBy("date")
    return trips_per_day

def popular_start_station_per_month(df):
    trips_per_station = df.groupBy(month("start_time").alias("month"), "from_station_name").agg(
        count("trip_id").alias("trips_count")
    )
    window_spec = Window.partitionBy("month").orderBy(col("trips_count").desc())
    result = trips_per_station.withColumn("rank", rank().over(window_spec)).filter(col("rank") == 1).drop("rank")
    return result

def top3_stations_per_day_last_2_weeks(df):
    two_weeks_ago = datetime(year=2019,month=12,day=31) - timedelta(weeks=2)
    two_weeks_ago_str = two_weeks_ago.strftime("%Y-%m-%d")
    recent_trips = df.filter(col("start_time") >= two_weeks_ago_str)
    top3_stations_per_day = recent_trips.groupBy(to_date("start_time").alias("date"), "from_station_name").agg(
        count("trip_id").alias("trips_count")
    ).orderBy("date", col("trips_count").desc())
    window_spec = Window.partitionBy("date").orderBy(col("trips_count").desc())
    top3_stations_per_day = top3_stations_per_day.withColumn("rank", rank().over(window_spec)) \
        .filter(col("rank") <= 3) \
        .drop("rank")
    return top3_stations_per_day

def man_vs_woman(df):
    df_filtered = df.filter(df["gender"].isNotNull())
    avg_trip_duration_gender = df_filtered.groupBy("gender").agg(
    avg("tripduration").alias("avg_tripduration"))
    return avg_trip_duration_gender


def main():
    spark = SparkSession.builder \
        .appName("spark") \
        .master("spark://spark-master:7077") \
        .getOrCreate()
    
    spark = SparkSession.builder.appName("spark").getOrCreate()



    file_path = "jobs/data.csv"
    df = spark.read.option("header", "true").csv(file_path, inferSchema=True)

    trip_duration_data = trip_duration(df)
    trips_per_day_data = trips_per_day(df)
    popular_start_station_per_month_data = popular_start_station_per_month(df)
    top3_stations_per_day_last_2_weeks_data = top3_stations_per_day_last_2_weeks(df)
    mans_vs_womans_data = man_vs_woman(df)

    output_dir = "jobs/out"
    os.makedirs(output_dir, exist_ok=True)

    trip_duration_data.write.csv(os.path.join(output_dir, "avg_trip_duration_per_day.csv"), header=True)
    trips_per_day_data.write.csv(os.path.join(output_dir, "trips_per_day.csv"), header=True)
    popular_start_station_per_month_data.write.csv(os.path.join(output_dir, "popular_start_station_per_month.csv"), header=True)
    top3_stations_per_day_last_2_weeks_data.write.csv(os.path.join(output_dir, "top3_stations_per_day_last_2_weeks.csv"), header=True)
    mans_vs_womans_data.write.csv(os.path.join(output_dir, "avg_trip_duration_by_gender.csv"), header=True)

    spark.stop()

if __name__ == "__main__":
    print('App started')
    main()









