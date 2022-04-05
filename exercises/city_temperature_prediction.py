from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=["Date"])
    data = data[data.Temp > -15]
    data["DayOfYear"] = data.Date.dt.day_of_year

    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    il_data = data[data.Country == "Israel"]

    plt.scatter(il_data.DayOfYear, il_data.Temp, c=il_data.Year, cmap=cm.get_cmap("gist_rainbow",
                                                                                  il_data.Year.nunique()))
    plt.xlabel("Day of year")
    plt.ylabel("Temperature")
    plt.title("Temperature vs Day of year")
    # add a legend
    plt.colorbar(label="Year", ticks=np.unique(il_data.Year))
    plt.savefig("./plots/city_temperature_scatter.png")
    plt.close()

    grouped_by_month = il_data.groupby("Month")
    temp_stds_by_month = grouped_by_month.agg(np.std).Temp
    plt.bar(temp_stds_by_month.index, temp_stds_by_month)
    plt.xlabel("Month")
    plt.ylabel("Standard deviation of temperature")
    plt.title("Standard deviation of temperature by month")
    plt.savefig("./plots/temp_stds_by_month.png")
    plt.close()

    # Question 3 - Exploring differences between countries
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    grouped_by_country_month = data.groupby(["Country", "Month"])
    temp_avgs_by_country_month = grouped_by_country_month.agg(np.mean).Temp
    temp_stds_by_country_month = grouped_by_country_month.agg(np.std).Temp

    for i, country in enumerate(data.Country.unique()):
        ax.plot(temp_avgs_by_country_month[country].index,
                temp_avgs_by_country_month[country],
                label=country + " mean", c="bgrm"[i], linewidth=3)
        ax.fill_between(temp_avgs_by_country_month[country].index,
                        temp_avgs_by_country_month[country] - temp_stds_by_country_month[country],
                        temp_avgs_by_country_month[country] + temp_stds_by_country_month[country],
                        alpha=0.15,
                        label=country + " std",
                        color="bgrm"[i])

    plt.xlabel("Month")
    plt.ylabel("Average temperature")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    plt.title("Average monthly temperature by country")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.savefig("./plots/temp_avgs_by_country_month.png")
    plt.close()

    # Question 4 - Fitting model for different values of `k`
    il_data_X, il_data_y = il_data.DayOfYear, il_data.Temp
    train_X, train_y, test_X, test_y = split_train_test(il_data_X, il_data_y)
    k_values = np.arange(1, 11)
    test_errors = []
    best_k = None
    best_error = np.inf
    for k in k_values:
        model = PolynomialFitting(k)
        model.fit(train_X, train_y)
        test_error = np.round(model.loss(test_X, test_y), 2)
        test_errors.append(test_error)
        print(f"k={k}, test_error={test_error}")

        if test_error < best_error:
            best_error = test_error
            best_k = k

    plt.bar(k_values, test_errors)
    plt.xlabel("k")
    plt.ylabel("Test error")
    plt.title("Test error by k")
    plt.savefig("./plots/test_error_by_k.png")
    plt.close()

    # Question 5 - Evaluating fitted model on different countries
    model = PolynomialFitting(best_k)
    model.fit(il_data_X, il_data_y)

    countries = sorted(set(data.Country.unique()) - {"Israel"})
    model_errors = []
    for country in countries:
        country_data = data[data.Country == country]
        model_errors.append(model.loss(country_data.DayOfYear, country_data.Temp))

    plt.bar(countries, model_errors)
    plt.xlabel("Country")
    plt.ylabel("Model error")
    plt.title(f"Model error by country (k={best_k})")
    plt.savefig("./plots/model_error_by_country.png")
    plt.close()
