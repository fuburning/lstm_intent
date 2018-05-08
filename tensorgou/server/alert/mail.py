#!/usr/bin/env python2.7
#encoding=utf-8
import os, sys
reload(sys)
sys.setdefaultencoding("utf-8")

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-11-21$"


import urllib

_MAIL_URL = "http://mail.portal.sogou/portal/tools/send_mail.php"

def send_mail(uid, fr_name, fr_addr, title, body, mail_list, mode="html"):
    raw_params = { "uid": uid, "fr_name": fr_name, "fr_addr": fr_addr,
            "title": title, "body": body, "maillist": ";".join(mail_list), "mode": mode }
    str_params = urllib.urlencode(raw_params)
    url = "%s?%s" % (_MAIL_URL, str_params)

    try:
        urllib.urlopen(url)
    except Exception as e:
        return False, "Exception:%s" % str(e)
    return True, "Sent mail successfully"

if __name__ == "__main__":
    print send_mail(uid="zhangyuanpeng@sogou-inc.com", fr_name="Tensorgou", fr_addr="zhangyuanpeng@sogou-inc.com",
            title="[Tensorgou]title", body="body:Tensorgou", mail_list=["zhangyuanpeng@sogou-inc.com"])

