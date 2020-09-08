from random import randint

import psycopg2
import time
from psycopg2.extras import DictCursor


class DbConnector:
    def __init__(self, user, password, host, database, schema):
        """
        Creates a database connector responsible for communicating with the experiment database.

        :param user: Database user
        :param password: Database password
        :param host: Database host
        :param database: Database name
        :param schema: Database schema
        """

        self.user = user
        self.password = password
        self.host = host
        self.database = database
        self.schema = schema

        self.connection = None
        self.cursor = None

    def init_connection(self):
        """
        Initializes the connection to the database.

        """
        self.connection = psycopg2.connect(user=self.user, database=self.database, password=self.password,
                                           host=self.host)
        self.connection.set_session(isolation_level='SERIALIZABLE')
        self.cursor = self.connection.cursor(cursor_factory=DictCursor)

    def close_connection(self):
        """
        Closes the database connection.

        """
        self.cursor.close()
        self.connection.close()

    def set_result_success(self, result, e2e=False):
        """
        Adds the result data to the result table in case of a successful experiment.

        :param result: Dict with the experiment results
        :param e2e: True, if the results are to be written to the e2e result table. Default: False
        """
        result_table = "e2e_result" if e2e else "result"
        try:
            self.init_connection()
            sql = """UPDATE {0}.{1} SET kendalls_tau=%s, spearman=%s, 
            zero_one_accuracy=%s, object_detection_precision=%s,
            object_detection_recall=%s, average_ranking_size=%s, label_accuracy=%s, duration=%s, 
            finished=now() where id = %s""".format(
                self.schema, result_table)
            self.cursor.execute(sql
                                , (float(result.get("kendalls_tau")), float(result.get("spearman")),
                                   float(result.get("zero_one_accuracy")),
                                   float(result.get("object_detection_precision")),
                                   float(result.get("object_detection_recall")),
                                   float(result.get("avg_ranking_size")),
                                   float(result.get("label_accuracy")),
                                   result.get("duration"),
                                   result.get("result_id")))
            self.connection.commit()
        finally:
            self.close_connection()

    def set_result_exception(self, result, e2e=False):
        """
        Adds the result data to the result table in case of an experiment with an exception.

        :param result: Dict with the experiment results
        :param e2e: True, if the results are to be written to the e2e result table. Default: False
        """
        result_table = "e2e_result" if e2e else "result"
        try:
            self.init_connection()
            sql = """UPDATE {0}.{1} SET exception=%s, duration=%s, finished=now() where id = %s""".format(self.schema,
                                                                                                          result_table)
            self.cursor.execute(sql
                                , (result.get("exception"), result.get("duration"), result.get("result_id")))
            self.connection.commit()
        finally:
            self.close_connection()

    def get_next_experiment(self, host, retries=3, e2e=False):
        """
        Tries to fetch another experiment from the configuration table.

        If an experiment was found, a result entry with the provided hostname is created.

        :param host: Hostname of the server the program is executed on
        :param retries: Number of retries for fetching the next experiment. Needed because sometimes exceptions occur
        due to concurrent database accesses. Default: 3
        :param e2e: True, if an e2e configuration has to be fetched. Default: False
        :return: An experiment configuration or None, if no more experiments are to be executed
        """
        if e2e:
            configuration_table = "e2e_configuration"
            result_table = "e2e_result"
        else:
            configuration_table = "configuration"
            result_table = "result"
        try:
            sql = """select * from {0}.{1} where id = (
                    select
                    c.id
                    from
                    {0}.{1} c left outer join {0}.{2} r
                    on (r.configuration_id = c.id)
                    group by c.id, c.n_experiments
                    having count(r.id) < c.n_experiments
                    order by id asc
                    limit 1);""".format(self.schema, configuration_table, result_table)
            self.init_connection()
            self.cursor.execute(sql)
            experiment_config = self.cursor.fetchone()
            if experiment_config is None:
                return None, None
            # Insert empty result entry
            sql = """INSERT INTO {0}.{1} (configuration_id, hostname) values 
                                            (%s,%s) returning id""".format(self.schema, result_table)
            self.cursor.execute(sql
                                , [experiment_config["id"], host])
            result_id = self.cursor.fetchone()[0]
            self.connection.commit()
        except psycopg2.OperationalError:
            if retries > 0:
                # Sleep random time and try again
                time.sleep(randint(1000, 5000) / 1000)
                return self.get_next_experiment(host, retries - 1, e2e=e2e)
            else:
                raise RuntimeError("Database error when trying to get and insert new experiment")
        finally:
            self.close_connection()
        return experiment_config, result_id
