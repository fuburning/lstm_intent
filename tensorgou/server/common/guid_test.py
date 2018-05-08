#!/usr/bin/env python2.7
#encoding=utf-8
import os, sys
reload(sys)
sys.setdefaultencoding("utf-8")

import unittest
import tensorgou.server.common.guid as guid

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-11-21$"


class GuidTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_JobId(self):
        id1, id2 = guid.JobId(), guid.JobId()
        self.assertEqual(id1._seq, id2._seq-1)
        self.assertNotEqual(id1, id2)

        self.assertEqual(guid.JobId(0,0), guid.JobId(0,0))
        self.assertNotEqual(guid.JobId(0,0), guid.JobId(0,1))
        self.assertNotEqual(guid.JobId(0,0), guid.JobId(1,0))
        self.assertNotEqual(guid.JobId(0,0), guid.JobId(1,1))
    
        id3 = guid.JobId(0,0)
        self.assertEqual(guid.JobId.job_id(id3.string_id()), id3)
        pass

    def test_TaskId(self):
        jid = guid.JobId()
        tid = guid.TaskId(jid.string_id())
        self.assertEqual(tid, guid.TaskId.task_id(tid.string_id()))
        pass

    def test_ResourceId(self):
        rid = guid.ResourceId()
        self.assertEqual(rid, guid.ResourceId.resource_id(rid.string_id()))
        self.assertTrue(None == guid.ResourceId.resource_id("res_1_2_3"))
        self.assertEqual(guid.ResourceId(5, 5), guid.ResourceId.resource_id("res_5_5"))
        pass

if __name__ == "__main__":
    unittest.main()
    
