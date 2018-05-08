#!/usr/bin/env python2.7
#encoding=utf-8
import os, sys
reload(sys)
sys.setdefaultencoding("utf-8")

class Storage:
    def connect(self):
        pass

    def close(self):
        pass

    def create_job_table(self):
        raise NotImplementedError

    def insert_job(self, job):
        raise NotImplementedError

    def update_job(self, job):
        raise NotImplementedError

    def job_guid_to_job_id(self, job_guid):
        return True, 0

    def create_task_table(self):
        raise NotImplementedError

    def insert_task(self, task):
        raise NotImplementedError

    def update_task(self, task):
        raise NotImplementedError

class LogStorage(Storage):
    """LogStorage, print log only"""

    def __init__(self, log_file):
        import tensorgou.server.scheduler.syspath_autofiller
        from tensorgou.server.common.file_logger import RotatingFileLogger
        self._logger = RotatingFileLogger(log_file, maxBytes=20*1024*1024, backupCount=5)
        pass

    def connect(self):
        self._logger.info("log_storage: connect")
        pass

    def close(self):
        self._logger.info("log_storage: close")
        pass

    def create_job_table(self):
        self._logger.info("log_storage: create job table")
        pass

    def insert_job(self, job):
        self._logger.info("log_storage: insert job [job:%s]" % str(job))
        pass

    def update_job(self, job):
        self._logger.info("log_storage: update job [job:%s]" % str(job))
        pass

    def create_task_table(self):
        self._logger.info("log_storage: create task table")
        pass

    def insert_task(self, task):
        self._logger.info("log_storage: insert task [task:%s]" % str(task))
        pass

    def update_task(self, task):
        self._logger.info("log_storage: update task [task:%s]" % str(task))
        pass

def _sql_wrapper(f):
    def _f(self, *args, **kwargs):
        try:
            r = f(self, *args, **kwargs)
        except:
            self.close()
            self.connect()
            try:
                r = f(self, *args, **kwargs)
            except:
                return False, None
        return True, r
    return _f

class SqliteStorage(Storage):
    def __init__(self, file_path):
        self._file_path = file_path
        self._conn = None
        self.connect()
        pass

    def connect(self):
        import sqlite3
        self._conn = sqlite3.connect(self._file_path, isolation_level=None)
        pass

    def close(self):
        if self._conn != None:
            self._conn.close()
            self._conn = None
        pass

    @_sql_wrapper
    def create_job_table(self):
        sql = '''
create table if not exists ts_job (
    job_id integer primary key autoincrement,
    job_guid varchar(128) not null default '',
    job_name varchar(128) not null default '',
    job_all_stage integer not null default 0,
    job_now_stage integer not null default 0,
    job_status integer not null default 0,
    job_brief text)
'''
        self._conn.cursor().execute(sql)
        self._conn.commit()
        pass

    @_sql_wrapper
    def insert_job(self, job):
        sql = "insert into ts_job(job_guid, job_name, job_all_stage, job_now_stage, job_status, job_brief) values(?,?,?,?,?,?)"
        self._conn.cursor().execute(sql, job[1:])
        self._conn.commit()
        pass
    
    @_sql_wrapper
    def update_job(self, job):
        sql = "update ts_job set job_guid=?, job_name=?, job_all_stage=?, job_now_stage=?, job_status=?, job_brief=? where job_id=?"
        self._conn.cursor().execute(sql, (job[1:]+(job[0],)))
        self._conn.commit()
        pass

    @_sql_wrapper
    def job_guid_to_job_id(self, job_guid):
        sql = "select job_id from ts_job where job_guid=?"
        c = self._conn.cursor()
        c.execute(sql, (job_guid,))
        r = c.fetchall()
        if len(r) == 1:
            return r[0][0]
        raise Exception("No record in db [job_guid:%s]" % job_guid)

    @_sql_wrapper
    def create_task_table(self):
        sql = '''
create table if not exists ts_task (
    task_id integer primary key autoincrement,
    task_guid varchar(128) not null default '',
    job_id integer not null default 0,
    job_guid varchar(128) not null default '',
    task_num integer not null default 0,
    task_sub_num integer not null default 0,
    task_status integer not null default 0,
    task_brief text)
'''
        self._conn.cursor().execute(sql)
        self._conn.commit()
        pass

    @_sql_wrapper
    def insert_task(self, task):
        sql = "insert into ts_task(task_guid, job_id, job_guid, task_num, task_sub_num, task_status, task_brief) values(?,?,?,?,?,?,?)"
        self._conn.cursor().execute(sql, task)
        self._conn.commit()
        pass
    
    @_sql_wrapper
    def update_task(self, task):
        sql = "update ts_task set job_id=?, job_guid=?, task_num=?, task_sub_num=?, task_status=?, task_brief=? where task_guid=?"
        self._conn.cursor().execute(sql, (task[1:]+(task[0],)))
        self._conn.commit()
        pass


class MysqlStorage(Storage):
    def __init__(self, host, port, user, passwd, db):
        self._mysql_host = host
        self._mysql_port = port
        self._mysql_user = user
        self._mysql_passwd = passwd
        self._mysql_db = db
        self._conn = None
        self.connect()
        pass

    def connect(self):
        import MySQLdb
        self._conn = MySQLdb.connect(host=self._mysql_host,
                port=self._mysql_port,
                user=self._mysql_user,
                passwd = self._mysql_passwd,
                db = self._mysql_db,
                charset = "utf8")
        self._conn.autocommit(1)
        pass

    def close(self):
        if self._conn != None:
            self._conn.close()
            self._conn = None
        pass

    @_sql_wrapper
    def create_job_table(self):
        sql = '''
create table if not exists ts_job (
    job_id integer primary key auto_increment,
    job_guid varchar(128) not null default '',
    job_name varchar(128) not null default '',
    job_all_stage integer not null default 0,
    job_now_stage integer not null default 0,
    job_status integer not null default 0,
    job_brief text,
    key `idx_job_guid` (job_guid)) charset=utf8 collate=utf8_bin
'''
        self._conn.cursor().execute(sql)
        self._conn.commit()
        pass

    @_sql_wrapper
    def insert_job(self, job):
        sql = "insert into ts_job(job_guid, job_name, job_all_stage, job_now_stage, job_status, job_brief) values(%s,%s,%s,%s,%s,%s)"
        self._conn.cursor().execute(sql, job[1:])
        self._conn.commit()
        pass
    
    @_sql_wrapper
    def update_job(self, job):
        sql = "update ts_job set job_guid=%s, job_name=%s, job_all_stage=%s, job_now_stage=%s, job_status=%s, job_brief=%s where job_id=%s"
        self._conn.cursor().execute(sql, (job[1:]+(job[0],)))
        self._conn.commit()
        pass

    @_sql_wrapper
    def job_guid_to_job_id(self, job_guid):
        sql = "select job_id from ts_job where job_guid=%s"
        c = self._conn.cursor()
        c.execute(sql, (job_guid,))
        r = c.fetchall()
        if len(r) == 1:
            return r[0][0]
        raise Exception("No record in db [job_guid:%s]" % job_guid)

    @_sql_wrapper
    def create_task_table(self):
        sql = '''
create table if not exists ts_task (
    task_id integer primary key auto_increment,
    task_guid varchar(128) not null default '',
    job_id integer not null default 0,
    job_guid varchar(128) not null default '',
    task_num integer not null default 0,
    task_sub_num integer not null default 0,
    task_status integer not null default 0,
    task_brief text,
    key `idx_task_guid` (task_guid)) charset=utf8 collate=utf8_bin
'''
        self._conn.cursor().execute(sql)
        self._conn.commit()
        pass

    @_sql_wrapper
    def insert_task(self, task):
        sql = "insert into ts_task(task_guid, job_id, job_guid, task_num, task_sub_num, task_status, task_brief) values(%s,%s,%s,%s,%s,%s,%s)"
        self._conn.cursor().execute(sql, task)
        self._conn.commit()
        pass
    
    @_sql_wrapper
    def update_task(self, task):
        sql = "update ts_task set job_id=%s, job_guid=%s, task_num=%s, task_sub_num=%s, task_status=%s, task_brief=%s where task_guid=%s"
        self._conn.cursor().execute(sql, (task[1:]+(task[0],)))
        self._conn.commit()
        pass

