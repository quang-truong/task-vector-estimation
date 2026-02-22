import duckdb
import pandas as pd

from relbench.base import Database, EntityTask, RecommendationTask, Table, TaskType
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
    mae,
    r2,
    rmse,
    roc_auc,
)


class UserChurnTask(EntityTask):
    r"""Churn for a customer is 1 if the customer does not review any product in the
    time window, else 0."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = "churn"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        product = db.table_dict["product"].df
        customer = db.table_dict["customer"].df
        review = db.table_dict["review"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                timestamp,
                customer_id,
                CAST(
                    NOT EXISTS (
                        SELECT 1
                        FROM review
                        WHERE
                            review.customer_id = customer.customer_id AND
                            review_time > timestamp AND
                            review_time <= timestamp + INTERVAL '{self.timedelta}'
                    ) AS INTEGER
                ) AS churn
            FROM
                timestamp_df,
                customer,
            WHERE
                EXISTS (
                    SELECT 1
                    FROM review
                    WHERE
                        review.customer_id = customer.customer_id AND
                        review_time > timestamp - INTERVAL '{self.timedelta}' AND
                        review_time <= timestamp
                )
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class UserLTVTask(EntityTask):
    r"""LTV (life-time value) for a customer is the sum of prices of products that the
    customer reviews in the time window."""

    task_type = TaskType.REGRESSION
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = "ltv"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        product = db.table_dict["product"].df
        customer = db.table_dict["customer"].df
        review = db.table_dict["review"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                timestamp,
                customer_id,
                ltv,
            FROM
                timestamp_df,
                customer,
                (
                    SELECT
                        COALESCE(SUM(price), 0) as ltv,
                    FROM
                        review,
                        product
                    WHERE
                        review.customer_id = customer.customer_id AND
                        review.product_id = product.product_id AND
                        review_time > timestamp AND
                        review_time <= timestamp + INTERVAL '{self.timedelta}'
                )
            WHERE
                EXISTS (
                    SELECT 1
                    FROM review
                    WHERE
                        review.customer_id = customer.customer_id AND
                        review_time > timestamp - INTERVAL '{self.timedelta}' AND
                        review_time <= timestamp
                )
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"customer_id": "customer"},
            pkey_col=None,
            time_col="timestamp",
        )


class ItemChurnTask(EntityTask):
    r"""Churn for a product is 1 if the product recieves at least one review in the time
    window, else 0."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "product_id"
    entity_table = "product"
    time_col = "timestamp"
    target_col = "churn"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        product = db.table_dict["product"].df
        customer = db.table_dict["customer"].df
        review = db.table_dict["review"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                timestamp,
                product_id,
                CAST(
                    NOT EXISTS (
                        SELECT 1
                        FROM review
                        WHERE
                            review.product_id = product.product_id AND
                            review_time > timestamp AND
                            review_time <= timestamp + INTERVAL '{self.timedelta}'
                    ) AS INTEGER
                ) AS churn
            FROM
                timestamp_df,
                product,
            WHERE
                EXISTS (
                    SELECT 1
                    FROM review
                    WHERE
                        review.product_id = product.product_id AND
                        review_time > timestamp - INTERVAL '{self.timedelta}' AND
                        review_time <= timestamp
                )
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class ItemLTVTask(EntityTask):
    r"""LTV (life-time value) for a product is the numer of times the product is
    purchased in the time window multiplied by price."""

    task_type = TaskType.REGRESSION
    entity_col = "product_id"
    entity_table = "product"
    time_col = "timestamp"
    target_col = "ltv"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        product = db.table_dict["product"].df
        customer = db.table_dict["customer"].df
        review = db.table_dict["review"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                timestamp,
                product.product_id,
                COALESCE(SUM(price), 0) AS ltv,
            FROM
                timestamp_df,
                product,
                review
            WHERE
                review.product_id = product.product_id AND
                review_time > timestamp AND
                review_time <= timestamp + INTERVAL '{self.timedelta}'
            GROUP BY
                timestamp,
                product.product_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col="timestamp",
        )


class UserItemPurchaseTask(RecommendationTask):
    r"""Predict the list of distinct items each customer will purchase in the next two
    years."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "customer_id"
    src_entity_table = "customer"
    dst_entity_col = "product_id"
    dst_entity_table = "product"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        customer = db.table_dict["customer"].df
        review = db.table_dict["review"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                review.customer_id,
                LIST(DISTINCT review.product_id) AS product_id
            FROM
                timestamp_df t
            LEFT JOIN
                review
            ON
                review.review_time > t.timestamp AND
                review.review_time <= t.timestamp + INTERVAL '{self.timedelta} days'
            WHERE
                review.customer_id is not null and review.product_id is not null
            GROUP BY
                t.timestamp,
                review.customer_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )


class UserItemRateTask(RecommendationTask):
    r"""Predict the list of distinct items each customer will purchase and give a 5 star
    review in the next two years."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "customer_id"
    src_entity_table = "customer"
    dst_entity_col = "product_id"
    dst_entity_table = "product"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        customer = db.table_dict["customer"].df
        review = db.table_dict["review"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp,
                    review.customer_id,
                    LIST(DISTINCT review.product_id) AS product_id
                FROM
                    timestamp_df t
                LEFT JOIN
                    review
                ON
                    review.review_time > t.timestamp AND
                    review.review_time <= t.timestamp + INTERVAL '{self.timedelta} days'
                WHERE
                    review.customer_id IS NOT NULL
                    AND review.product_id IS NOT NULL
                    AND review.rating = 5.0
                GROUP BY
                    t.timestamp,
                    review.customer_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )


class UserItemReviewTask(RecommendationTask):
    r"""Predict the list of distinct items each customer will purchase and give a
    detailed review in the next two years."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "customer_id"
    src_entity_table = "customer"
    dst_entity_col = "product_id"
    dst_entity_table = "product"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        customer = db.table_dict["customer"].df
        review = db.table_dict["review"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        REVIEW_LENGTH = (
            300  # minimum length of review to be considered as detailed review
        )

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp,
                    review.customer_id,
                    LIST(DISTINCT review.product_id) AS product_id
                FROM
                    timestamp_df t
                LEFT JOIN
                    review
                ON
                    review.review_time > t.timestamp AND
                    review.review_time <= t.timestamp + INTERVAL '{self.timedelta} days'
                WHERE
                    review.customer_id IS NOT NULL
                    AND review.product_id IS NOT NULL
                    AND (LENGTH(review.review_text) > {REVIEW_LENGTH} AND review.review_text IS NOT NULL)
                GROUP BY
                    t.timestamp,
                    review.customer_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )

###############################################
################ New Tasks ####################
###############################################

class UserSSLTask(EntityTask):
    task_type = None
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = None
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = None

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        pass

class UserTVE1HopTask(EntityTask):
    task_type = None
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = ['customer_review_0__mean__review__0__rating',
       'customer_review_0__min__review__0__rating',
       'customer_review_0__max__review__0__rating',
       'customer_review_0__sum__review__0__rating',
       'customer_review_0__count__review__0__rating',
       'customer_review_0__stddev__review__0__rating',
       'customer_review_0__count__review__0__verified',
       'customer_review_0__count_distinct__review__0__verified',
       'customer_review_0__mode__review__0__verified_False',
       'customer_review_0__mode__review__0__verified_True', 'label']
    # target_col = ['pca_component_1', 'pca_component_2', 'pca_component_3']
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = None

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        pass
        
class UserTVE2HopTask(EntityTask):
    task_type = None
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = ['customer_review_0__mean__review__0__rating',
       'customer_review_0__min__review__0__rating',
       'customer_review_0__max__review__0__rating',
       'customer_review_0__sum__review__0__rating',
       'customer_review_0__count__review__0__rating',
       'customer_review_0__stddev__review__0__rating',
       'customer_review_0__count__review__0__verified',
       'customer_review_0__count_distinct__review__0__verified',
       'customer_review_product_0__mean__product__0__price',
       'customer_review_product_0__min__product__0__price',
       'customer_review_product_0__max__product__0__price',
       'customer_review_product_0__sum__product__0__price',
       'customer_review_product_0__count__product__0__price',
       'customer_review_product_0__stddev__product__0__price',
       'customer_review_review_0__mean__review__1__rating',
       'customer_review_review_0__min__review__1__rating',
       'customer_review_review_0__max__review__1__rating',
       'customer_review_review_0__sum__review__1__rating',
       'customer_review_review_0__count__review__1__rating',
       'customer_review_review_0__stddev__review__1__rating',
       'customer_review_review_0__count__review__1__verified',
       'customer_review_review_0__count_distinct__review__1__verified',
       'customer_review_review_1__mean__review__1__rating',
       'customer_review_review_1__min__review__1__rating',
       'customer_review_review_1__max__review__1__rating',
       'customer_review_review_1__sum__review__1__rating',
       'customer_review_review_1__count__review__1__rating',
       'customer_review_review_1__stddev__review__1__rating',
       'customer_review_review_1__count__review__1__verified',
       'customer_review_review_1__count_distinct__review__1__verified',
       'customer_review_0__mode__review__0__verified_False',
       'customer_review_0__mode__review__0__verified_True',
       'customer_review_review_0__mode__review__1__verified_False',
       'customer_review_review_0__mode__review__1__verified_True',
       'customer_review_review_1__mode__review__1__verified_False',
       'customer_review_review_1__mode__review__1__verified_True', 'label']
    # target_col = ['pca_component_1', 'pca_component_2', 'pca_component_3']
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = None

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        pass

class ItemSSLTask(EntityTask):
    task_type = None
    entity_col = "product_id"
    entity_table = "product"
    time_col = "timestamp"
    target_col = None
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = None

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        pass
    
class ItemTVE1HopTask(EntityTask):
    task_type = None
    entity_col = "product_id"
    entity_table = "product"
    time_col = "timestamp"
    target_col = [
        'product_review_0__mean__review__0__rating',
        'product_review_0__min__review__0__rating',
        'product_review_0__max__review__0__rating',
        'product_review_0__sum__review__0__rating',
        'product_review_0__count__review__0__rating',
        'product_review_0__stddev__review__0__rating',
        'product_review_0__count__review__0__verified',
        'product_review_0__count_distinct__review__0__verified',
        'product_review_0__mode__review__0__verified_False',
        'product_review_0__mode__review__0__verified_True', 'label'
    ]
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = None
    
    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        pass

class ItemTVE2HopTask(EntityTask):
    task_type = None
    entity_col = "product_id"
    entity_table = "product"
    time_col = "timestamp"
    target_col = [
        'product_review_0__mean__review__0__rating',
        'product_review_0__min__review__0__rating',
        'product_review_0__max__review__0__rating',
        'product_review_0__sum__review__0__rating',
        'product_review_0__count__review__0__rating',
        'product_review_0__stddev__review__0__rating',
        'product_review_0__count__review__0__verified',
        'product_review_0__count_distinct__review__0__verified',
        'product_review_review_0__mean__review__1__rating',
        'product_review_review_0__min__review__1__rating',
        'product_review_review_0__max__review__1__rating',
        'product_review_review_0__sum__review__1__rating',
        'product_review_review_0__count__review__1__rating',
        'product_review_review_0__stddev__review__1__rating',
        'product_review_review_0__count__review__1__verified',
        'product_review_review_0__count_distinct__review__1__verified',
        'product_review_review_1__mean__review__1__rating',
        'product_review_review_1__min__review__1__rating',
        'product_review_review_1__max__review__1__rating',
        'product_review_review_1__sum__review__1__rating',
        'product_review_review_1__count__review__1__rating',
        'product_review_review_1__stddev__review__1__rating',
        'product_review_review_1__count__review__1__verified',
        'product_review_review_1__count_distinct__review__1__verified',
        'product_review_0__mode__review__0__verified_False',
        'product_review_0__mode__review__0__verified_True',
        'product_review_review_0__mode__review__1__verified_False',
        'product_review_review_0__mode__review__1__verified_True',
        'product_review_review_1__mode__review__1__verified_False',
        'product_review_review_1__mode__review__1__verified_True', 'label'
    ]
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = None
    
    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        pass
        
class UserChurnTop100SpendingTask(EntityTask):
    r"""Churn for top 100 spending customers. A customer churns (1) if they do not review 
    any product in the next time window."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = "churn"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [average_precision, accuracy, f1, roc_auc]
    k = 100

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        product = db.table_dict["product"].df
        customer = db.table_dict["customer"].df
        review = db.table_dict["review"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            WITH customer_spending AS (
                -- Calculate total spending per customer in previous period
                SELECT 
                    t.timestamp,
                    r.customer_id,
                    SUM(p.price) as total_spending
                FROM timestamp_df t
                CROSS JOIN review r
                JOIN product p ON r.product_id = p.product_id
                WHERE r.review_time > t.timestamp - INTERVAL '{self.timedelta}'
                    AND r.review_time <= t.timestamp
                GROUP BY t.timestamp, r.customer_id
            ),
            top_spenders AS (
                -- Select top {self.k} spenders per timestamp
                SELECT 
                    timestamp,
                    customer_id
                FROM (
                    SELECT 
                        timestamp,
                        customer_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY timestamp 
                            ORDER BY total_spending DESC
                        ) as spending_rank
                    FROM customer_spending
                )
                WHERE spending_rank <= {self.k}
            )
            -- Calculate churn for top spenders
            SELECT
                ts.timestamp,
                ts.customer_id,
                CAST(
                    NOT EXISTS (
                        SELECT 1
                        FROM review r
                        WHERE r.customer_id = ts.customer_id
                            AND r.review_time > ts.timestamp
                            AND r.review_time <= ts.timestamp + INTERVAL '{self.timedelta}'
                    ) AS INTEGER
                ) AS churn
            FROM top_spenders ts
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )
        
class UserChurnTop50SpendingTask(EntityTask):
    r"""Churn for top 50 spending customers. A customer churns (1) if they do not review 
    any product in the next time window."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = "churn"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [average_precision, accuracy, f1, roc_auc]
    k = 50

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        product = db.table_dict["product"].df
        customer = db.table_dict["customer"].df
        review = db.table_dict["review"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            WITH customer_spending AS (
                -- Calculate total spending per customer in previous period
                SELECT 
                    t.timestamp,
                    r.customer_id,
                    SUM(p.price) as total_spending
                FROM timestamp_df t
                CROSS JOIN review r
                JOIN product p ON r.product_id = p.product_id
                WHERE r.review_time > t.timestamp - INTERVAL '{self.timedelta}'
                    AND r.review_time <= t.timestamp
                GROUP BY t.timestamp, r.customer_id
            ),
            top_spenders AS (
                -- Select top {self.k} spenders per timestamp
                SELECT 
                    timestamp,
                    customer_id
                FROM (
                    SELECT 
                        timestamp,
                        customer_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY timestamp 
                            ORDER BY total_spending DESC
                        ) as spending_rank
                    FROM customer_spending
                )
                WHERE spending_rank <= {self.k}
            )
            -- Calculate churn for top spenders
            SELECT
                ts.timestamp,
                ts.customer_id,
                CAST(
                    NOT EXISTS (
                        SELECT 1
                        FROM review r
                        WHERE r.customer_id = ts.customer_id
                            AND r.review_time > ts.timestamp
                            AND r.review_time <= ts.timestamp + INTERVAL '{self.timedelta}'
                    ) AS INTEGER
                ) AS churn
            FROM top_spenders ts
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

class ItemChurnLeast100SpendingTask(EntityTask):
    r"""Churn for least 100 spending products. A product churns (1) if it does not receive 
    any reviews in the next time window."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "product_id"
    entity_table = "product"
    time_col = "timestamp"
    target_col = "churn"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [average_precision, accuracy, f1, roc_auc]
    k = 100

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        product = db.table_dict["product"].df
        customer = db.table_dict["customer"].df
        review = db.table_dict["review"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            WITH product_spending AS (
                -- Calculate total spending per product in previous period
                SELECT 
                    t.timestamp,
                    r.product_id,
                    SUM(p.price) as total_spending
                FROM timestamp_df t
                CROSS JOIN review r
                JOIN product p ON r.product_id = p.product_id
                WHERE r.review_time > t.timestamp - INTERVAL '{self.timedelta}'
                    AND r.review_time <= t.timestamp
                GROUP BY t.timestamp, r.product_id
            ),
            top_products AS (
                -- Select top {self.k} products per timestamp by spending
                SELECT 
                    timestamp,
                    product_id
                FROM (
                    SELECT 
                        timestamp,
                        product_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY timestamp 
                            ORDER BY total_spending ASC
                        ) as spending_rank
                    FROM product_spending
                )
                WHERE spending_rank <= {self.k}
            )
            -- Calculate churn for top products
            SELECT
                tp.timestamp,
                tp.product_id,
                CAST(
                    NOT EXISTS (
                        SELECT 1
                        FROM review r
                        WHERE r.product_id = tp.product_id
                            AND r.review_time > tp.timestamp
                            AND r.review_time <= tp.timestamp + INTERVAL '{self.timedelta}'
                    ) AS INTEGER
                ) AS churn
            FROM top_products tp
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )
        
class ItemChurnLeast50SpendingTask(EntityTask):
    r"""Churn for least 50 spending products. A product churns (1) if it does not receive 
    any reviews in the next time window."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "product_id"
    entity_table = "product"
    time_col = "timestamp"
    target_col = "churn"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [average_precision, accuracy, f1, roc_auc]
    k = 50

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        product = db.table_dict["product"].df
        customer = db.table_dict["customer"].df
        review = db.table_dict["review"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            WITH product_spending AS (
                -- Calculate total spending per product in previous period
                SELECT 
                    t.timestamp,
                    r.product_id,
                    SUM(p.price) as total_spending
                FROM timestamp_df t
                CROSS JOIN review r
                JOIN product p ON r.product_id = p.product_id
                WHERE r.review_time > t.timestamp - INTERVAL '{self.timedelta}'
                    AND r.review_time <= t.timestamp
                GROUP BY t.timestamp, r.product_id
            ),
            top_products AS (
                -- Select top {self.k} products per timestamp by spending
                SELECT 
                    timestamp,
                    product_id
                FROM (
                    SELECT 
                        timestamp,
                        product_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY timestamp 
                            ORDER BY total_spending ASC
                        ) as spending_rank
                    FROM product_spending
                )
                WHERE spending_rank <= {self.k}
            )
            -- Calculate churn for top products
            SELECT
                tp.timestamp,
                tp.product_id,
                CAST(
                    NOT EXISTS (
                        SELECT 1
                        FROM review r
                        WHERE r.product_id = tp.product_id
                            AND r.review_time > tp.timestamp
                            AND r.review_time <= tp.timestamp + INTERVAL '{self.timedelta}'
                    ) AS INTEGER
                ) AS churn
            FROM top_products tp
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class ItemLTVLeast100SpendingTask(EntityTask):
    r"""LTV (life-time value) for least 100 spending products. LTV is the number of times 
    the product is purchased in the future time window multiplied by price."""

    task_type = TaskType.REGRESSION
    entity_col = "product_id"
    entity_table = "product"
    time_col = "timestamp"
    target_col = "ltv"
    timedelta = pd.Timedelta(days=365 // 4)
    metrics = [r2, mae, rmse]
    k = 100

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        product = db.table_dict["product"].df
        customer = db.table_dict["customer"].df
        review = db.table_dict["review"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            WITH product_spending AS (
                -- Calculate total spending per product in previous period
                SELECT 
                    t.timestamp,
                    r.product_id,
                    SUM(p.price) as total_spending
                FROM timestamp_df t
                CROSS JOIN review r
                JOIN product p ON r.product_id = p.product_id
                WHERE r.review_time > t.timestamp - INTERVAL '{self.timedelta}'
                    AND r.review_time <= t.timestamp
                GROUP BY t.timestamp, r.product_id
            ),
            top_products AS (
                -- Select top {self.k} products per timestamp by spending
                SELECT 
                    timestamp,
                    product_id
                FROM (
                    SELECT 
                        timestamp,
                        product_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY timestamp 
                            ORDER BY total_spending ASC
                        ) as spending_rank
                    FROM product_spending
                )
                WHERE spending_rank <= {self.k}
            )
            -- Calculate future LTV for top products
            SELECT
                tp.timestamp,
                tp.product_id,
                COALESCE(SUM(p.price), 0) AS ltv
            FROM top_products tp
            LEFT JOIN review r ON 
                r.product_id = tp.product_id
                AND r.review_time > tp.timestamp
                AND r.review_time <= tp.timestamp + INTERVAL '{self.timedelta}'
            LEFT JOIN product p ON r.product_id = p.product_id
            GROUP BY tp.timestamp, tp.product_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

class UserLTVBadReviewsTask(EntityTask):
    r"""Predict the lifetime value (sum of all product prices from purchases) for 
    users who only submitted bad reviews (rating of 1.0 or 2.0 or 3.0) in the previous period."""

    task_type = TaskType.REGRESSION
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = "ltv"
    timedelta = pd.Timedelta(days=90)  # Assuming same timedelta as the original task
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        customer = db.table_dict["customer"].df
        review = db.table_dict["review"].df
        product = db.table_dict["product"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            WITH previous_period_customers AS (
                -- Find customers who only had bad reviews in previous period
                SELECT 
                    t.timestamp,
                    r.customer_id
                FROM timestamp_df t
                JOIN review r ON r.review_time > t.timestamp - INTERVAL '{self.timedelta}'
                             AND r.review_time <= t.timestamp
                GROUP BY t.timestamp, r.customer_id
                HAVING 
                    -- Check that all ratings are 1.0 or 2.0
                    MAX(CASE WHEN r.rating > 3.0 THEN 1 ELSE 0 END) = 0
                    -- Ensure they submitted at least one review
                    AND COUNT(*) > 0
            )
            
            SELECT
                pc.timestamp,
                pc.customer_id,
                COALESCE(
                    (
                        SELECT SUM(p.price)
                        FROM review r
                        JOIN product p ON r.product_id = p.product_id
                        WHERE r.customer_id = pc.customer_id
                          AND r.review_time > pc.timestamp
                          AND r.review_time <= pc.timestamp + INTERVAL '{self.timedelta}'
                    ),
                    0
                ) AS ltv
            FROM previous_period_customers pc
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )