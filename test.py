import time
import datetime

# Get the current time as a timestamp
timestamp = time.time()

# Convert timestamp to a datetime object
dt_object = datetime.datetime.fromtimestamp(timestamp)

# Format the datetime object as a string
formatted_time = dt_object.strftime("%Y-%m-%d_%H:%M:%S")

print(formatted_time)