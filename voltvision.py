import os

# Java path
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot"

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Start Spark
spark = SparkSession.builder \
    .appName("EV Charging Analysis") \
    .master("local[*]") \
    .getOrCreate()

print("Spark started successfully!")

# Load dataset
df = spark.read.csv(
    "fixed_ev_charging_data.csv",
    header=True,
    inferSchema=True
)

print("Dataset loaded successfully!")
df.show(5)
df.printSchema()

# Data preprocessing
df = df.withColumn(
        "start_time", to_timestamp("start_time")
    ).withColumn(
        "end_time", to_timestamp("end_time")
    ).withColumn(
        "hour", hour("start_time")
    ).withColumn(
        "date", to_date("start_time")
    ).withColumn(
        "day", date_format("start_time", "EEEE")
    ).withColumn(
        "duration_mins",
        (unix_timestamp("end_time") - unix_timestamp("start_time")) / 60
    )

# 1. Boxplot for charging duration by city
dur_city = df.select("city", "duration_mins").toPandas()

fig = px.box(
    dur_city,
    x="city",
    y="duration_mins",
    color="city",
    title="Charging Duration Distribution by City"
)
fig.show()

# 2. Peak hour analysis
peak = df.groupBy("hour").count().orderBy("hour")
peak_pd = peak.toPandas()

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=peak_pd["hour"],
        y=peak_pd["count"],
        mode="lines+markers",
        fill="tozeroy"
    )
)

fig.update_layout(
    title="EV Charging Demand by Hour",
    xaxis_title="Hour of Day",
    yaxis_title="Number of Sessions"
)

fig.show()

# 3. Heatmap day vs hour
heat = df.groupBy("day", "hour").count().toPandas()

pivot = heat.pivot(
    index="day",
    columns="hour",
    values="count"
).fillna(0)

days_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

pivot = pivot.reindex(days_order)

plt.figure(figsize=(12, 6))
sns.heatmap(pivot, cmap="coolwarm", annot=True)
plt.title("Charging Demand Heatmap (Day vs Hour)")
plt.show(block=False)

# 4. City-wise demand
city = df.groupBy("city") \
         .count() \
         .orderBy(desc("count")) \
         .toPandas()

fig = px.bar(
    city,
    x="count",
    y="city",
    orientation="h",
    title="City-wise Charging Demand",
    color="count"
)

fig.show()

# 5. Daily energy consumption
energy = df.groupBy("date") \
           .agg(sum("energy_kwh").alias("energy")) \
           .orderBy("date") \
           .toPandas()

fig = px.line(
    energy,
    x="date",
    y="energy",
    markers=True,
    title="Daily Energy Consumption Trend"
)

fig.show()

# 6. Charging duration histogram
dur = df.select("duration_mins").toPandas()

fig = px.histogram(
    dur,
    x="duration_mins",
    nbins=30,
    title="Charging Duration Distribution",
    marginal="box"
)

fig.show()

# 7. Linear regression prediction
model_data = df.select("hour", "energy_kwh").toPandas()

X = model_data[["hour"]]
y = model_data["energy_kwh"]

model = LinearRegression()
model.fit(X, y)

print("Prediction model trained successfully!")

# User prediction
while True:
    user = input("Enter hour (0-23) or type 'exit': ")

    if user.lower() == "exit":
        print("Exited prediction.")
        break

    try:
        hour_val = int(user)

        if hour_val < 0 or hour_val > 23:
            print("Please enter a valid hour between 0 and 23.\n")
            continue

        pred = model.predict(
            pd.DataFrame({"hour": [hour_val]})
        )

        print(
            f"⚡ Predicted Energy at {hour_val}:00 → {float(pred[0]):.2f} kWh\n"
        )

    except:
        print("Invalid input! Enter a number.\n")

spark.stop()