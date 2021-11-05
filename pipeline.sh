#!/bin/bash
set -x
token='https://oapi.dingtalk.com/robot/send?access_token=9d02b894f4500e76907453a95aca4ae4330d7d49ba16e9ad6db7650e50f0bc28'
sudo docker restart fate-new-test && sleep 30 && sudo docker exec -i fate-new-test /bin/bash -c 'find /fate/auto-test -name "*.sh" |xargs -I {} bash {}'
if [ $? != 0 ];then
  curl $token -H 'Content-type: application/json' -d "{\"msgtype\": \"text\", \"text\": {\"content\":\"$* pipeline构建失败报警\"}}"
fi