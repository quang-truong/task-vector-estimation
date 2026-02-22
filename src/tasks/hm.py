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


class UserItemPurchaseTask(RecommendationTask):
    r"""Predict the list of articles each customer will purchase in the next seven
    days."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "customer_id"
    src_entity_table = "customer"
    dst_entity_col = "article_id"
    dst_entity_table = "article"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=7)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 12
    
    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        customer = db.table_dict["customer"].df
        transactions = db.table_dict["transactions"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                transactions.customer_id,
                LIST(DISTINCT transactions.article_id) AS article_id
            FROM
                timestamp_df t
            LEFT JOIN
                transactions
            ON
                transactions.t_dat > t.timestamp AND
                transactions.t_dat <= t.timestamp + INTERVAL '{self.timedelta} days'
            GROUP BY
                t.timestamp,
                transactions.customer_id
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


class UserChurnTask(EntityTask):
    r"""Predict the churn for a customer (no transactions) in the next week."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = "churn"
    timedelta = pd.Timedelta(days=7)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        customer = db.table_dict["customer"].df
        transactions = db.table_dict["transactions"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                timestamp,
                customer_id,
                CAST(
                    NOT EXISTS (
                        SELECT 1
                        FROM transactions
                        WHERE
                            transactions.customer_id = customer.customer_id AND
                            t_dat > timestamp AND
                            t_dat <= timestamp + INTERVAL '{self.timedelta}'
                    ) AS INTEGER
                ) AS churn
            FROM
                timestamp_df,
                customer,
            WHERE
                EXISTS (
                    SELECT 1
                    FROM transactions
                    WHERE
                        transactions.customer_id = customer.customer_id AND
                        t_dat > timestamp - INTERVAL '{self.timedelta}' AND
                        t_dat <= timestamp
                )
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class ItemSalesTask(EntityTask):
    r"""Predict the total sales for an article (the sum of prices of the associated
    transactions) in the next week."""

    task_type = TaskType.REGRESSION
    entity_col = "article_id"
    entity_table = "article"
    time_col = "timestamp"
    target_col = "sales"
    timedelta = pd.Timedelta(days=7)
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        transactions = db.table_dict["transactions"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        article = db.table_dict["article"].df

        df = duckdb.sql(
            f"""
            SELECT
                timestamp,
                article_id,
                sales
            FROM
                timestamp_df,
                article,
                (
                    SELECT
                        COALESCE(SUM(price), 0) as sales
                    FROM
                        transactions,
                    WHERE
                        transactions.article_id = article.article_id AND
                        t_dat > timestamp AND
                        t_dat <= timestamp + INTERVAL '{self.timedelta}'
                )
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"article_id": "article"},
            pkey_col=None,
            time_col="timestamp",
        )

###############################################
################ New Tasks ####################
###############################################

class UserChurnTop50SpendingTask(EntityTask):
    r"""Churn for top 50 spending customers. A customer churns (1) if they do not purchase 
    any product in the next time window."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = "churn"
    timedelta = pd.Timedelta(days=7)
    metrics = [average_precision, accuracy, f1, roc_auc]
    k = 50

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        customer = db.table_dict["customer"].df
        transactions = db.table_dict["transactions"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        
        df = duckdb.sql(
            f"""
            WITH customer_spending AS (
                -- Calculate total spending per customer in previous period
                SELECT 
                    t.timestamp,
                    tr.customer_id,
                    SUM(tr.price) as total_spending
                FROM timestamp_df t
                CROSS JOIN transactions tr
                WHERE tr.t_dat > t.timestamp - INTERVAL '{self.timedelta}'
                    AND tr.t_dat <= t.timestamp
                GROUP BY t.timestamp, tr.customer_id
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
                        FROM transactions tr
                        WHERE tr.customer_id = ts.customer_id
                            AND tr.t_dat > ts.timestamp
                            AND tr.t_dat <= ts.timestamp + INTERVAL '{self.timedelta}'
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
        
class UserChurnTop100SpendingTask(EntityTask):
    r"""Churn for top 100 spending customers. A customer churns (1) if they do not purchase 
    any product in the next time window."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = "churn"
    timedelta = pd.Timedelta(days=7)
    metrics = [average_precision, accuracy, f1, roc_auc]
    k = 100

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        customer = db.table_dict["customer"].df
        transactions = db.table_dict["transactions"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        
        df = duckdb.sql(
            f"""
            WITH customer_spending AS (
                -- Calculate total spending per customer in previous period
                SELECT 
                    t.timestamp,
                    tr.customer_id,
                    SUM(tr.price) as total_spending
                FROM timestamp_df t
                CROSS JOIN transactions tr
                WHERE tr.t_dat > t.timestamp - INTERVAL '{self.timedelta}'
                    AND tr.t_dat <= t.timestamp
                GROUP BY t.timestamp, tr.customer_id
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
                        FROM transactions tr
                        WHERE tr.customer_id = ts.customer_id
                            AND tr.t_dat > ts.timestamp
                            AND tr.t_dat <= ts.timestamp + INTERVAL '{self.timedelta}'
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
    
class ItemSalesTop200SpendingTask(EntityTask):
    r"""Predict the total sales for top 200 selling articles (the sum of prices of the associated
    transactions) in the next week."""

    task_type = TaskType.REGRESSION
    entity_col = "article_id"
    entity_table = "article"
    time_col = "timestamp"
    target_col = "sales"
    timedelta = pd.Timedelta(days=7)
    metrics = [r2, mae, rmse]
    k = 200

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        transactions = db.table_dict["transactions"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        article = db.table_dict["article"].df

        df = duckdb.sql(
            f"""
            WITH article_sales AS (
                -- Calculate total sales per article in previous period
                SELECT 
                    t.timestamp,
                    tr.article_id,
                    SUM(tr.price) as total_sales
                FROM timestamp_df t
                CROSS JOIN transactions tr
                WHERE tr.t_dat > t.timestamp - INTERVAL '{self.timedelta}'
                    AND tr.t_dat <= t.timestamp
                GROUP BY t.timestamp, tr.article_id
            ),
            top_sellers AS (
                -- Select top {self.k} selling articles per timestamp
                SELECT 
                    timestamp,
                    article_id
                FROM (
                    SELECT 
                        timestamp,
                        article_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY timestamp 
                            ORDER BY total_sales DESC
                        ) as sales_rank
                    FROM article_sales
                )
                WHERE sales_rank <= {self.k}
            )
            -- Calculate future sales for top sellers
            SELECT
                ts.timestamp,
                ts.article_id,
                COALESCE(
                    (
                        SELECT SUM(tr.price)
                        FROM transactions tr
                        WHERE tr.article_id = ts.article_id
                            AND tr.t_dat > ts.timestamp
                            AND tr.t_dat <= ts.timestamp + INTERVAL '{self.timedelta}'
                    ),
                    0
                ) AS sales
            FROM top_sellers ts
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

class UserSSLTask(EntityTask):
    task_type = None
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = None
    timedelta = pd.Timedelta(days=7)
    metrics = None

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        pass

class UserTVE1HopTask(EntityTask):
    task_type = None
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = ['customer_transactions_0__mean__transactions__0__price',
        'customer_transactions_0__min__transactions__0__price',
        'customer_transactions_0__max__transactions__0__price',
        'customer_transactions_0__sum__transactions__0__price',
        'customer_transactions_0__count__transactions__0__price',
        'customer_transactions_0__stddev__transactions__0__price',
        'customer_transactions_0__count__transactions__0__sales_channel_id',
        'customer_transactions_0__count_distinct__transactions__0__sales_channel_id',
        'customer_transactions_0__mode__transactions__0__sales_channel_id_1.0',
        'customer_transactions_0__mode__transactions__0__sales_channel_id_2.0',
        'label'
    ]
    # target_col = ['pca_component_1', 'pca_component_2', 'pca_component_3']
    timedelta = pd.Timedelta(days=7)
    metrics = None

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        pass
    
class UserTVE2HopTask(EntityTask):
    task_type = None
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = [
        'customer_transactions_0__mean__transactions__0__price',
        'customer_transactions_0__min__transactions__0__price',
        'customer_transactions_0__max__transactions__0__price',
        'customer_transactions_0__sum__transactions__0__price',
        'customer_transactions_0__count__transactions__0__price',
        'customer_transactions_0__stddev__transactions__0__price',
        'customer_transactions_0__count__transactions__0__sales_channel_id',
        'customer_transactions_0__count_distinct__transactions__0__sales_channel_id',
        'customer_transactions_article_0__count__article__0__product_type_no',
        'customer_transactions_article_0__count_distinct__article__0__product_type_no',
        'customer_transactions_article_0__count__article__0__product_group_name',
        'customer_transactions_article_0__count_distinct__article__0__product_group_name',
        'customer_transactions_article_0__count__article__0__graphical_appearance_no',
        'customer_transactions_article_0__count_distinct__article__0__graphical_appearance_no',
        'customer_transactions_article_0__count__article__0__colour_group_code',
        'customer_transactions_article_0__count_distinct__article__0__colour_group_code',
        'customer_transactions_article_0__count__article__0__perceived_colour_value_id',
        'customer_transactions_article_0__count_distinct__article__0__perceived_colour_value_id',
        'customer_transactions_article_0__count__article__0__perceived_colour_master_id',
        'customer_transactions_article_0__count_distinct__article__0__perceived_colour_master_id',
        'customer_transactions_article_0__count__article__0__department_no',
        'customer_transactions_article_0__count_distinct__article__0__department_no',
        'customer_transactions_article_0__count__article__0__department_name',
        'customer_transactions_article_0__count_distinct__article__0__department_name',
        'customer_transactions_article_0__count__article__0__index_code',
        'customer_transactions_article_0__count_distinct__article__0__index_code',
        'customer_transactions_article_0__count__article__0__index_group_no',
        'customer_transactions_article_0__count_distinct__article__0__index_group_no',
        'customer_transactions_article_0__count__article__0__section_no',
        'customer_transactions_article_0__count_distinct__article__0__section_no',
        'customer_transactions_article_0__count__article__0__garment_group_no',
        'customer_transactions_article_0__count_distinct__article__0__garment_group_no',
        'customer_transactions_transactions_0__mean__transactions__1__price',
        'customer_transactions_transactions_0__min__transactions__1__price',
        'customer_transactions_transactions_0__max__transactions__1__price',
        'customer_transactions_transactions_0__sum__transactions__1__price',
        'customer_transactions_transactions_0__count__transactions__1__price',
        'customer_transactions_transactions_0__stddev__transactions__1__price',
        'customer_transactions_transactions_0__count__transactions__1__sales_channel_id',
        'customer_transactions_transactions_0__count_distinct__transactions__1__sales_channel_id',
        'customer_transactions_transactions_1__mean__transactions__1__price',
        'customer_transactions_transactions_1__min__transactions__1__price',
        'customer_transactions_transactions_1__max__transactions__1__price',
        'customer_transactions_transactions_1__sum__transactions__1__price',
        'customer_transactions_transactions_1__count__transactions__1__price',
        'customer_transactions_transactions_1__stddev__transactions__1__price',
        'customer_transactions_transactions_1__count__transactions__1__sales_channel_id',
        'customer_transactions_transactions_1__count_distinct__transactions__1__sales_channel_id',
        '0',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        '10',
        '11',
        '12',
        '13',
        '14',
        '15',
        '16',
        '17',
        '18',
        '19',
        '20',
        '21',
        '22',
        '23',
        '24',
        '25',
        '26',
        '27',
        '28',
        '29',
        '30',
        '31',
        '32',
        '33',
        '34',
        '35',
        '36',
        '37',
        '38',
        '39',
        '40',
        '41',
        '42',
        '43',
        '44',
        '45',
        '46',
        '47',
        '48',
        '49',
        '50',
        '51',
        '52',
        '53',
        '54',
        '55',
        '56',
        '57',
        '58',
        '59',
        '60',
        '61',
        '62',
        '63',
        'label'
    ]
    # target_col = ['pca_component_1', 'pca_component_2', 'pca_component_3']
    timedelta = pd.Timedelta(days=7)
    metrics = None

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        pass

class ItemSSLTask(EntityTask):
    task_type = None
    entity_col = "article_id"
    entity_table = "article"
    time_col = "timestamp"
    target_col = None
    timedelta = pd.Timedelta(days=7)
    metrics = None

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        pass
    
class ItemTVE1HopTask(EntityTask):
    task_type = None
    entity_col = "article_id"
    entity_table = "article"
    time_col = "timestamp"
    target_col = [
        'article_transactions_0__mean__transactions__0__price',
        'article_transactions_0__min__transactions__0__price',
        'article_transactions_0__max__transactions__0__price',
        'article_transactions_0__sum__transactions__0__price',
        'article_transactions_0__count__transactions__0__price',
        'article_transactions_0__stddev__transactions__0__price',
        'article_transactions_0__count__transactions__0__sales_channel_id',
        'article_transactions_0__count_distinct__transactions__0__sales_channel_id',
        'article_transactions_0__mode__transactions__0__sales_channel_id_1.0',
        'article_transactions_0__mode__transactions__0__sales_channel_id_2.0',
        'label'
    ]
    timedelta = pd.Timedelta(days=7)
    metrics = None

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        pass
    
class ItemTVE2HopTask(EntityTask):
    task_type = None
    entity_col = "article_id"
    entity_table = "article"
    time_col = "timestamp"
    target_col = [
        'article_transactions_0__mean__transactions__0__price',
        'article_transactions_0__min__transactions__0__price',
        'article_transactions_0__max__transactions__0__price',
        'article_transactions_0__sum__transactions__0__price',
        'article_transactions_0__count__transactions__0__price',
        'article_transactions_0__stddev__transactions__0__price',
        'article_transactions_0__count__transactions__0__sales_channel_id',
        'article_transactions_0__count_distinct__transactions__0__sales_channel_id',
        'article_transactions_customer_0__count__customer__0__FN',
        'article_transactions_customer_0__count_distinct__customer__0__FN',
        'article_transactions_customer_0__count__customer__0__Active',
        'article_transactions_customer_0__count_distinct__customer__0__Active',
        'article_transactions_customer_0__count__customer__0__club_member_status',
        'article_transactions_customer_0__count_distinct__customer__0__club_member_status',
        'article_transactions_customer_0__count__customer__0__fashion_news_frequency',
        'article_transactions_customer_0__count_distinct__customer__0__fashion_news_frequency',
        'article_transactions_customer_0__mean__customer__0__age',
        'article_transactions_customer_0__min__customer__0__age',
        'article_transactions_customer_0__max__customer__0__age',
        'article_transactions_customer_0__sum__customer__0__age',
        'article_transactions_customer_0__count__customer__0__age',
        'article_transactions_customer_0__stddev__customer__0__age',
        'article_transactions_transactions_0__mean__transactions__1__price',
        'article_transactions_transactions_0__min__transactions__1__price',
        'article_transactions_transactions_0__max__transactions__1__price',
        'article_transactions_transactions_0__sum__transactions__1__price',
        'article_transactions_transactions_0__count__transactions__1__price',
        'article_transactions_transactions_0__stddev__transactions__1__price',
        'article_transactions_transactions_0__count__transactions__1__sales_channel_id',
        'article_transactions_transactions_0__count_distinct__transactions__1__sales_channel_id',
        'article_transactions_transactions_1__mean__transactions__1__price',
        'article_transactions_transactions_1__min__transactions__1__price',
        'article_transactions_transactions_1__max__transactions__1__price',
        'article_transactions_transactions_1__sum__transactions__1__price',
        'article_transactions_transactions_1__count__transactions__1__price',
        'article_transactions_transactions_1__stddev__transactions__1__price',
        'article_transactions_transactions_1__count__transactions__1__sales_channel_id',
        'article_transactions_transactions_1__count_distinct__transactions__1__sales_channel_id',
        'article_transactions_0__mode__transactions__0__sales_channel_id_1.0',
        'article_transactions_0__mode__transactions__0__sales_channel_id_2.0',
        'article_transactions_customer_0__mode__customer__0__FN_1.0',
        'article_transactions_customer_0__mode__customer__0__FN_nan',
        'article_transactions_customer_0__mode__customer__0__Active_1.0',
        'article_transactions_customer_0__mode__customer__0__Active_nan',
        'article_transactions_customer_0__mode__customer__0__club_member_status_ACTIVE',
        'article_transactions_customer_0__mode__customer__0__club_member_status_LEFT CLUB',
        'article_transactions_customer_0__mode__customer__0__club_member_status_PRE-CREATE',
        'article_transactions_customer_0__mode__customer__0__club_member_status_None',
        'article_transactions_customer_0__mode__customer__0__fashion_news_frequency_Monthly',
        'article_transactions_customer_0__mode__customer__0__fashion_news_frequency_NONE',
        'article_transactions_customer_0__mode__customer__0__fashion_news_frequency_Regularly',
        'article_transactions_customer_0__mode__customer__0__fashion_news_frequency_None',
        'article_transactions_transactions_0__mode__transactions__1__sales_channel_id_1.0',
        'article_transactions_transactions_0__mode__transactions__1__sales_channel_id_2.0',
        'article_transactions_transactions_1__mode__transactions__1__sales_channel_id_1.0',
        'article_transactions_transactions_1__mode__transactions__1__sales_channel_id_2.0',
        'label'
    ]
    timedelta = pd.Timedelta(days=7)
    metrics = None

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        pass
    