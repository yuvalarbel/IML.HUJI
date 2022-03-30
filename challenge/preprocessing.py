#!%PYTHON_HOME%\python.exe
# coding: utf-8

# Standard Library Imports
import re
import json

import numpy as np
import pandas as pd
from sklearn import preprocessing

# Project Imports
import challenge.consts as consts


# TODO: Columns:
#       - datetime columns - Seconds since epoch
#       - accommadation_type_name


class Preprocess(object):
    ADD_POLYNOMIAL_FEATURES = False
    STANDARDIZE_FEATURES = True
    POLYNOMIAL_FEATURE_DEGREE = 2

    Y_COLUMNS = ['time_to_cancel', 'cancel_time_to_checkin', 'real_cancellation_datetime']

    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
    DATE_COLS = ['month', 'day', 'weekday']
    TIME_COLS = ['hour', 'minute']
    DATETIME_COLUMNS = {'booking_datetime':  DATE_COLS + TIME_COLS,
                        'checkin_date':  DATE_COLS,
                        'checkout_date': DATE_COLS}
    DATETIME_DIFFERENCES = [('hotel_live_date', 'booking_datetime'),
                            ('booking_datetime', 'checkin_date'),
                            ('checkin_date', 'checkout_date')]
    BINARY_CLASSES = {'original_payment_type': ['Invoice', 'Credit Card', 'Gift Card'],
                      'charge_option': ['Pay Now', 'Pay Later', 'Pay at Check-in']}
    NUMBER_COLS = ['hotel_star_rating',
                   'guest_is_not_the_customer',
                   'no_of_adults',
                   'no_of_children',
                   'no_of_extra_bed',
                   'no_of_room',
                   'is_user_logged_in',
                   'is_first_booking',
                   'hotel_area_code',
                   'hotel_city_code']
    BINARY_NULLS_COLS = ['request_nonesmoke',
                         'request_latecheckin',
                         'request_highfloor',
                         'request_largebed',
                         'request_twinbeds',
                         'request_airport',
                         'request_earlycheckin']
    SPECIAL_COLUMN_GROUPS_FILENAME = 'special_column_groups.json'

    FEATURES_TO_POLYNOMIALIZE = []

    def __init__(self, data):
        self.data = data
        self.unnormalized_features = data[[]]
        self.features = None
        self.labels = None

        self.feature_funcs = [
            self.datetimes,
            self.datetime_differences,
            self.binary_classes,
            self.numbers,
            self.misc,
            self.binary_with_nulls,
            self.payment_amount_usd_equivalence,
            self.cancellation_policy,
            self.create_country_dummies,
            self.special_column_groups
        ]

    def run(self):
        self.create_features()
        self.create_labels()
        return self.features, self.labels

    def run_final(self):
        self.create_features()
        return self.features

    def create_features(self):
        for func in self.feature_funcs:
            func()
        self.features = self.unnormalized_features
        if self.ADD_POLYNOMIAL_FEATURES:
            poly = preprocessing.PolynomialFeatures(degree=self.POLYNOMIAL_FEATURE_DEGREE)
            self.features = poly.fit_transform(self.features)
        if self.STANDARDIZE_FEATURES:
            scaler = preprocessing.StandardScaler()
            scaler.fit(self.features)
            self.features = scaler.transform(self.features)
        self.features = pd.DataFrame(self.features, columns=self.unnormalized_features.columns)
        self.features.insert(self.features.shape[1], 'X_booking_datetime_original',
                             self.col_to_datetime('booking_datetime'))
        self.features.insert(self.features.shape[1], 'X_checkin_date_original', self.col_to_datetime('checkin_date'))

    def create_labels(self):
        time_to_cancel = self.get_datetime_diff_in_secs("booking_datetime", "cancellation_datetime")
        cancel_time_to_booking = self.get_datetime_diff_in_secs("cancellation_datetime", "checkin_date")
        self.labels = pd.concat([time_to_cancel,
                                 pd.concat([cancel_time_to_booking,
                                            self.col_to_datetime("cancellation_datetime")], axis=1)], axis=1)
        self.labels.columns = self.Y_COLUMNS

    def add_new_feature(self, name, feature):
        assert feature.size == self.data.shape[0]
        assert name not in self.unnormalized_features.columns

        self.unnormalized_features.insert(self.unnormalized_features.shape[1], name, feature)

    def col_to_datetime(self, col):
        return pd.to_datetime(self.data[col], format=self.DATETIME_FORMAT)

    ###### Feature functions ######

    def datetimes(self):
        for col_name, cols in self.DATETIME_COLUMNS.items():
            datetime_col = self.col_to_datetime(col_name)
            for col in cols:
                self.add_new_feature(col_name + '_' + col, getattr(datetime_col.dt, col))

    def datetime_differences(self):
        for earlier, later in self.DATETIME_DIFFERENCES:
            diff = self.get_datetime_diff_in_secs(earlier, later)
            self.add_new_feature('_'.join([earlier, later, 'diff']), diff)

    def get_datetime_diff_in_secs(self, earlier, later):
        later_dt = self.col_to_datetime(later)
        earlier_dt = self.col_to_datetime(earlier)
        diff = (later_dt - earlier_dt) / np.timedelta64(1, 's')
        return diff

    def binary_classes(self):
        for col, classes in self.BINARY_CLASSES.items():
            for class_name in classes:
                class_tag = '_'.join(class_name.lower().split())
                self.add_new_feature(col + '_' + class_tag,
                                     (self.data[col] == class_name).astype(int))

    def numbers(self):
        for col in self.NUMBER_COLS:
            self.add_new_feature(col, self.data[col])

    def misc(self):
        col = self.data.original_payment_method == 'UNKNOWN'
        self.add_new_feature('payment_method_unknown', col.astype(int))

    def binary_with_nulls(self):
        for col in self.BINARY_NULLS_COLS:
            self.add_new_feature(col, self.data[col].fillna(0))

    def payment_amount_usd_equivalence(self):
        usd_amounts = np.empty(len(self.data))
        selling_amounts = self.data["original_selling_amount"]
        currencies = self.data["original_payment_currency"]
        for i in range(len(self.data)):
            usd_amounts[i] = self.convert_amount_to_usd(consts.usd_conversion_rates, float(selling_amounts[i]),
                                                        currencies[i])
        self.add_new_feature("payment_amount_usd_equivalence", usd_amounts)

    @staticmethod
    def convert_amount_to_usd(conversion_rates, amount: float, currency: str) -> float:
        # Where USD is the base currency you want to use
        rate = conversion_rates[currency]
        return amount / rate

    def cancellation_policy(self):
        cancellation_policy = self.data["cancellation_policy_code"]

        data_first_cancellation_policy_days = np.zeros(len(self.data))
        data_first_cancellation_policy_payment_percentage = np.empty(len(self.data))
        data_first_cancellation_policy_nights = np.empty(len(self.data))
        data_second_cancellation_policy_days = np.empty(len(self.data))
        data_second_cancellation_policy_payment_percentage = np.empty(len(self.data))
        data_second_cancellation_policy_nights = np.empty(len(self.data))
        data_no_show_cancellation_policy_payment_percentage = np.empty(len(self.data))
        data_no_show_cancellation_policy_nights = np.empty(len(self.data))

        for i in range(len(cancellation_policy)):
            first_cancellation_policy_days = 0
            first_cancellation_policy_payment_percentage = 0
            first_cancellation_policy_nights = 0
            second_cancellation_policy_days = 0
            second_cancellation_policy_payment_percentage = 0
            second_cancellation_policy_nights = 0
            no_show_cancellation_policy_payment_percentage = 0
            no_show_cancellation_policy_nights = 0

            cancellation_data = cancellation_policy[i].split("_")
            for policy in cancellation_data:
                first_result = re.match('(\\d*)D(\\d*)N', policy)
                second_result = re.match('(\\d*)D(\\d*)P', policy)
                third_result = re.match('(\\d*)P', policy)
                fourth_result = re.match('(\\d*)N', policy)

                if first_result:
                    if not first_cancellation_policy_days:
                        first_cancellation_policy_days = float(first_result.group(1))
                        first_cancellation_policy_nights = float(first_result.group(2))
                    else:
                        second_cancellation_policy_days = float(first_result.group(1))
                        second_cancellation_policy_nights = float(first_result.group(2))
                elif second_result:
                    if not first_cancellation_policy_days:
                        first_cancellation_policy_days = float(second_result.group(1))
                        first_cancellation_policy_payment_percentage = float(second_result.group(2))
                    else:
                        second_cancellation_policy_days = float(second_result.group(1))
                        second_cancellation_policy_payment_percentage = float(second_result.group(2))
                elif third_result:
                    no_show_cancellation_policy_payment_percentage = float(third_result.group(1))
                elif fourth_result:
                    no_show_cancellation_policy_nights = float(fourth_result.group(1))

                data_first_cancellation_policy_days[i] = first_cancellation_policy_days
                data_first_cancellation_policy_nights[i] = first_cancellation_policy_nights
                data_first_cancellation_policy_payment_percentage[i] = first_cancellation_policy_payment_percentage
                data_second_cancellation_policy_days[i] = second_cancellation_policy_days
                data_second_cancellation_policy_nights[i] = second_cancellation_policy_nights
                data_second_cancellation_policy_payment_percentage[i] = second_cancellation_policy_payment_percentage
                data_no_show_cancellation_policy_nights[i] = no_show_cancellation_policy_nights
                data_no_show_cancellation_policy_payment_percentage[i] = no_show_cancellation_policy_payment_percentage

        self.add_new_feature("first_cancellation_policy_days", data_first_cancellation_policy_days)
        self.add_new_feature("first_cancellation_policy_nights", data_first_cancellation_policy_nights)
        self.add_new_feature("first_cancellation_policy_payment_percentage",
                             data_first_cancellation_policy_payment_percentage)
        self.add_new_feature("second_cancellation_policy_days", data_second_cancellation_policy_days)
        self.add_new_feature("second_cancellation_policy_nights", data_second_cancellation_policy_nights)
        self.add_new_feature("second_cancellation_policy_payment_percentage",
                             data_second_cancellation_policy_payment_percentage)
        self.add_new_feature("no_show_cancellation_policy_nights", data_no_show_cancellation_policy_nights)
        self.add_new_feature("no_show_cancellation_policy_payment_percentage",
                             data_no_show_cancellation_policy_payment_percentage)

    def create_country_dummies(self):
        numeric_customer_country = self.data[
            'customer_nationality'].map(
            consts.gdp_dict)
        self.add_new_feature("numeric_customer_country",
                             numeric_customer_country)
        numeric_guest_country = self.data[
            'guest_nationality_country_name'].map(
            consts.gdp_dict)
        self.add_new_feature("numeric_guest_country",
                             numeric_guest_country)

    def special_column_groups(self):
        with open(self.SPECIAL_COLUMN_GROUPS_FILENAME, 'r') as f:
            special_column_groups = json.load(f)

        for col, groups in special_column_groups.items():
            for i, group in enumerate(groups):
                label = col + '_class_' + str(i)
                self.add_new_feature(label, (self.data[col] == group).astype(int))
