from fetch.download_data import get_positive_star_list
from fetch.download_data import run_query
import lightkurve as lk

query = """
SELECT DISTINCT hostname
FROM ps
WHERE disc_facility = 'Kepler'
"""

df = run_query(query)

print(len(df))
print(df.head())
print(df["hostname"].nunique())

positive = get_positive_star_list(1000)

print(len(positive))
print(positive[:10])