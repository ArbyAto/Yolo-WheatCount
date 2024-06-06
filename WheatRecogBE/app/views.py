from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from utils.tools import json_response
from utils.tools import rename
from utils.mature import classify_maturity
from app.models import FieldInfo
import os

# 上传图片
def upload_picture(request):
    if request.method == 'POST':
        if 'image' in request.FILES:
            image = request.FILES['image']
            field_id = request.POST.get('fid')
            upload_time = request.POST.get('upload_time')

            # 保存图片到本地文件系统
            image_name = rename(image.name)
            save_path = os.path.join(settings.MEDIA_ROOT, image_name)
            with open(save_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)
            # 保存相关数据到数据库

            print(f'麦田号:{field_id},上传时间:{upload_time},重命名为:{image_name}')

            return json_response(200, '图片上传成功')
        else:
            return json_response(400, '上传资源缺失')
    return json_response(400, '无效的请求方法')

# 上传图片
def upload_picture_debug(request):
    if request.method == 'POST':
        image = request.POST.get('image')
        field_id = request.POST.get('fid')
        upload_time = request.POST.get('upload_time')

        # 保存图片到本地文件系统
        image_name = rename(image.name)
        save_path = os.path.join(settings.MEDIA_ROOT_RAW, image_name)
        with open(save_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)
        # 保存相关数据到数据库

        print(f'麦田号:{field_id},上传时间:{upload_time},重命名为:{image_name}')

        return json_response(200, '图片上传成功')
    return json_response(400, '无效的请求方法')

def upload_video(request):
    return HttpResponse("upload_video")

def history_query(request):
    return HttpResponse("history_query")
