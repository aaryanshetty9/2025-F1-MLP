import fastf1

# Enable cache
fastf1.Cache.enable_cache('cache')

# Fetch the most recent 2024 event
event_schedule = fastf1.get_event_schedule(2024)
latest_event = event_schedule.iloc[-1]  # Get the last race that has data

# Load the latest race session to extract the correct driver lineup
latest_race = fastf1.get_session(2024, latest_event['EventName'], 'R')
latest_race.load()

# Get the official 2024 driver roster
current_drivers = latest_race.drivers
print("Current 2024 Driver Roster:", current_drivers)
