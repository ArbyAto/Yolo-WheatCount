from django.http import JsonResponse
from django.conf import settings
import uuid
import os
# 该文件为一些封装好的函数,方便调用

# 返回json回应,输入为 返回id 返回信息 返回数据
def json_response(code=200, msg='请求成功', data=None):
    json_dict = {
        'code': code,
        'msg': msg,
    }
    if data is not None:
        json_dict.update({'data': data})
    return JsonResponse(json_dict, json_dumps_params={"ensure_ascii": False})

# 返回一个文件重命名
def rename(original_filename):
    extension = os.path.splitext(original_filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{extension}"
    return unique_filename

# 调用图片计数模型
def detect(fileid, filename):
    counts = settings.MAIN_WINDOW_MODEL.detect_img(f"app/static/raw/{fileid}.jpg", filename)
    return counts

# 调用视频计数模型
def detect_video(fileid, filename):
    counts = settings.MAIN_WINDOW_MODEL.detect_vid(f"app/static/raw/{fileid}.mp4", filename)
    return counts