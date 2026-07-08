import lightkurve as lk

search = lk.search_lightcurve(
    "Kepler-10",
    mission="Kepler",
    author="Kepler",
    exptime=1800
)

print(search)

lc = search[0].download()

print(lc)