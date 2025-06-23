import pandas as pd
import numpy as np
import dask.dataframe as dd

n = 5_000_000

df = pd.DataFrame(
    {
        "id":np.arange(n),
        "value":np.random.normal(100,20,size=n),
        "category":np.random.choice(["A","B","C"],size=n)
    })

#zapis wygenerowanej taablicy do pliku csv
df.to_csv("big_data.csv",index=False)

#wczytanie danych z użyciem dask
ddf = dd.read_csv("big_data.csv")
print(df.head(8))

#proste operacje
mean_value = ddf['value'].mean()
print(f"Średnia leniwa: {mean_value}")

print(f"Średnia obliczona: {mean_value.compute():.2f}")

#grupowanie
grouped_df = ddf.groupby('category')["value"].mean()
print(f"Średnia leniwa dla każdej kategorii: {grouped_df}")
print(f"Średnia obliczona dla każdej kategorii: {grouped_df.compute()}")

print("__________________________________________")

# #proste operacje
# mean_value = df['value'].mean()
#
# print(f"Średnia obliczona: {mean_value:.2f}")
#
# #grupowanie
# grouped_df = df.groupby('category')["value"].mean()
# print(f"Średnia obliczona dla każdej kategorii: {grouped_df}")

