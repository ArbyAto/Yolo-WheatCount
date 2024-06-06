from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from utils.tools import json_response
from utils.tools import rename
from utils.tools import detect
from utils.tools import detect_video
from utils.mature import classify_maturity
from app.models import FieldInfo, History
import os
import shutil

# 上传图片
def upload_picture(request):
    if request.method == 'POST':
        try:
            if 'image' in request.FILES:
                image = request.FILES['image']
                field_id = int(request.POST.get('fid'))
                upload_time = request.POST.get('upload_time')

                image_name = rename(image.name)
                save_path = os.path.join(settings.MEDIA_ROOT, image_name)

                field = FieldInfo.objects.get(field_id=field_id)
                # 保存相关数据到数据库
                new_record = History.objects.create(
                    field=field,
                    file_path=image_name,
                    wheat_counts=0,
                    maturity=0.00,
                    upload_time=upload_time,
                    file_type=0
                )
                # 保存图片到本地文件系统
                with open(save_path, 'wb+') as destination:
                    for chunk in image.chunks():
                        destination.write(chunk)
                print(f'麦田号:{field_id},上传时间:{upload_time},重命名为:{image_name}')

                return json_response(200, '图片上传成功', {'history_id':new_record.history_id})
            else:
                return json_response(400, '上传资源缺失')
        except Exception as e:
            return json_response(400, str(e))
    return json_response(400, '无效的请求方法')

# 上传视频
def upload_video(request):
    if request.method == 'POST':
        try:
            if 'video' in request.FILES:
                video = request.FILES['video']
                field_id = request.POST.get('fid')
                upload_time = request.POST.get('upload_time')

                # 保存视频到本地文件系统
                video_name = rename(video.name)
                save_path = os.path.join(settings.MEDIA_ROOT, video_name)
                
                field = FieldInfo.objects.get(field_id=field_id)
                # 保存相关数据到数据库
                new_record = History.objects.create(
                    field=field,
                    file_path=video_name,
                    wheat_counts=0,
                    maturity=0.00,
                    upload_time=upload_time,
                    file_type=1
                )
                with open(save_path, 'wb+') as destination:
                    for chunk in video.chunks():
                        destination.write(chunk)
                print(f'麦田号:{field_id},上传时间:{upload_time},重命名为:{video_name}')

                return json_response(200, '图片上传成功', {'history_id':new_record.history_id})
            else:
                return json_response(400, '上传资源缺失')
        except Exception as e:
            return json_response(400, str(e))
    return json_response(400, '无效的请求方法')

# 上传图片序号?
def upload_file(request):
    if request.method == 'POST':
        try:
            image = int(request.POST.get('iid'))
            field_id = int(request.POST.get('fid'))
            upload_time = request.POST.get('upload_time')

            image_name = rename("sample.jpg")
            image_path = os.path.join(settings.MEDIA_ROOT_RAW, f"{image}.jpg")
            save_path = os.path.join(settings.MEDIA_ROOT, image_name)
            with open(image_path, 'rb') as imagedata, open(save_path, 'wb') as outfile:
                # 将数据从源文件复制到目标文件
                shutil.copyfileobj(imagedata, outfile)

            wheat_counts = detect(image, image_name)
            # 预估产量(单位公斤) 和成熟度
            predict_yield = (9675 * wheat_counts) / ((60 + (1.4 * wheat_counts)) * 10)
            predict_mature = classify_maturity(image_path)

            field = FieldInfo.objects.get(field_id=field_id)
            field.predict_yield = predict_yield
            field.maturity = predict_mature
            field.save()
            # 保存相关数据到数据库
            new_record = History.objects.create(
                field=field,
                file_path=image_name,
                wheat_counts=wheat_counts,
                maturity=predict_mature,
                upload_time=upload_time,
                file_type=0
            )
            
            print(f'麦田号:{field_id},上传时间:{upload_time},重命名为:{image_name},识别数量为:{wheat_counts},成熟度为:{predict_mature}')

            return json_response(200, '图片上传成功', [{'history_id':new_record.history_id}])
        except Exception as e:
            return json_response(400, str(e))
    return json_response(400, '无效的请求方法')

# 上传视频序号?
def upload_file_video(request):
    if request.method == 'POST':
        try:
            video = int(request.POST.get('vid'))
            field_id = int(request.POST.get('fid'))
            upload_time = request.POST.get('upload_time')

            video_name = rename("sample.mp4")
            video_path = os.path.join(settings.MEDIA_ROOT_RAW, f"{video}.mp4")
            save_path = os.path.join(settings.MEDIA_ROOT, video_name)
            with open(video_path, 'rb') as videodata, open(save_path, 'wb') as outfile:
                # 将数据从源文件复制到目标文件
                shutil.copyfileobj(videodata, outfile)

            wheat_counts = detect_video(video, video_name)
            # 预估产量(单位公斤) 和成熟度

            txt_path = 'F:/Huawei/Internship/Backend/WheatRecogBE/app/static/raw/temp.txt'

            # 打开文件进行写入操作，模式为 'w' 表示覆盖写入
            with open(txt_path, 'w') as file:
                # 将整数数组的每个元素写入文件，每个整数占一行
                for count in wheat_counts:
                    file.write(f"{count}\n")

            field = FieldInfo.objects.get(field_id=field_id)
            # field.predict_yield = predict_yield
            # field.maturity = predict_mature
            # field.save()
            # 保存相关数据到数据库
            new_record = History.objects.create(
                field=field,
                file_path=video_name,
                wheat_counts=0,
                maturity=0.0,
                upload_time=upload_time,
                file_type=1
            )
            
            print(f'麦田号:{field_id},上传时间:{upload_time},重命名为:{video_name}')

            return json_response(200, '视频上传成功', [{'history_id':new_record.history_id}])
        except Exception as e:
            return json_response(400, str(e))
    return json_response(400, '无效的请求方法')


# 查询所有历史信息
def history_query(request):
    if request.method == 'GET':
        try:
            # 查询所有 History 数据项
            allinfos = History.objects.values('history_id', 'field_id', 'upload_time', 'file_path')
            # 构建 JSON 响应
            info_list = list(allinfos)
            return json_response(200, '返回成功', info_list)
        except Exception as e:
            return json_response(400, str(e))
    else:
        return json_response(400, '无效的请求方法')

# 根据记录id查询某个记录(如果是图片记录则同时返回图片)
def get_history(request, history_id):
    if request.method == 'GET':
        try:
            # 获取指定的 History 记录
            history = get_object_or_404(History, pk=int(history_id))
            if history.maturity == 0.0:
                mature_type = "乳熟期"
            elif history.maturity == 1.0:
                mature_type = "蜡熟期"
            elif history.maturity == 2.0:
                mature_type = "完熟期"
            else:
                mature_type = "中间阶段"

            if history.file_type == 0:
                # 构建响应数据
                data = [{
                    'history_id': history.history_id,
                    'field_id': history.field_id,  # 注意 ForeignKey 字段
                    'file_path': history.file_path, 
                    'wheat_counts': history.wheat_counts,
                    'upload_time': history.upload_time,
                    'file_type': history.file_type,
                    'maturity': mature_type,
                }]
            else:
                file_path = 'F:/Huawei/Internship/Backend/WheatRecogBE/app/static/raw/temp.txt'

                # 初始化一个空列表来存储读取的整数
                wheat_counts = []

                # 打开文件进行读取操作，模式为 'r' 表示读取
                with open(file_path, 'r') as file:
                    # 读取文件的每一行
                    for line in file:
                        # 去掉每行的换行符并转换为整数，添加到列表中
                        wheat_counts.append(int(line.strip()))
                
                # 构建响应数据
                data = [{
                    'history_id': history.history_id,
                    'field_id': history.field_id,  # 注意 ForeignKey 字段
                    'file_path': history.file_path, 
                    'wheat_counts': wheat_counts,
                    'upload_time': history.upload_time,
                    'file_type': history.file_type,
                    'maturity': mature_type,
                }]

            return json_response(200, '返回成功', data)
        except Exception as e:
            return json_response(400, str(e))
    else:
        return json_response(400, '无效的请求方法')

# 删除历史记录(同时删除对应文件)
def delete_history(request):
    if request.method == 'POST':
        history_id = request.POST.get('hid')
        if not history_id:
            return json_response(400, '该田区不存在')
        try:
            history_info = get_object_or_404(History, pk=history_id)
            file_path = os.path.join(settings.MEDIA_ROOT, history_info.file_path)
            outcome_path = os.path.join(settings.MEDIA_ROOT_OUTCOME, history_info.file_path)
            # file_path = history_info.file_path
            # 删除记录
            history_info.delete()
            # 删除图片
            if os.path.exists(file_path):
                os.remove(file_path)

            if os.path.exists(outcome_path):
                os.remove(outcome_path)

            return json_response(200, '删除成功')
        except Exception as e:
            return json_response(400, str(e))
    else:
        return json_response(400, '无效的请求方法')

def test(request):
    detect_video(0, "testname.mp4")
    return HttpResponse("upload_video")
