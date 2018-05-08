#!/usr/bin/env python2.7
#encoding=utf-8
import os, sys
reload(sys)
sys.setdefaultencoding("utf-8")

import urllib
import json

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-11-21$"


_SMS_URL = "http://sms.sogou-op.org/portal/mobile/smsproxy.php"


def send_message(appid, number, desc, rettype="json"):
    raw_params = { "appid": appid, "number": number, "desc": desc, "type": rettype }
    str_params = urllib.urlencode(raw_params)
    url = "%s?%s" % (_SMS_URL, str_params)

    try:
        r = json.loads(urllib.urlopen(url).read())
        return r["code"] == 0, r["desc"]
    except Exception as e:
        return False, "Exception:%s" % str(e)

if __name__ == "__main__":
    # only on certain ip
    print send_message("tensorgou", "13581834556", "test")

