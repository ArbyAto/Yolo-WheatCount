from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from utils.tools import json_response
from utils.tools import rename
from app.models import FieldInfo
import os

# Create your views here.
# 访问根页面,测试用
def index(request):
    return HttpResponse("index")

# 创建一个麦田块
def create_field(request):
    if request.method == 'POST':
        try:
            acreage = request.POST.get('acreage')
            description = str(request.POST.get('description'))
            # 创建新的表项
            new_field = FieldInfo.objects.create(
                acreage=acreage,
                predict_yield=0.00,
                maturity=0.00,
                description=description
            )

            return json_response(200, '创建成功', {'field_id':new_field.field_id})
        except Exception as e:
            return json_response(400, str(e))
    return json_response(400, '无效的请求方法')

# 删除麦田块
def delete_field(request):
    if request.method == 'POST':
        field_id = request.POST.get('fid')
        if not field_id:
            return json_response(400, '该田区不存在')
        try:
            field_info = get_object_or_404(FieldInfo, pk=field_id)
            field_info.delete()
            return json_response(200, '删除成功')
        except Exception as e:
            return json_response(400, str(e))
    else:
        return json_response(400, '无效的请求方法')

# 查询所有麦田块
def get_all_field(request):
    if request.method == 'GET':
        try:
            # 查询所有 FieldInfo 数据项
            fieldinfos = FieldInfo.objects.all()
            # 构建 JSON 响应
            fieldinfo_list = list(fieldinfos.values())
            return json_response(200, '返回成功', fieldinfo_list)
        except Exception as e:
            return json_response(400, str(e))
    else:
        return json_response(400, '无效的请求方法')

def get_field(request, field_id):
    if request.method == 'GET':
        try:
            # 获取指定的 History 记录
            field = get_object_or_404(FieldInfo, pk=int(field_id))

            if field.maturity == 0.0:
                mature_type = "乳熟期"
            elif field.maturity == 1.0:
                mature_type = "蜡熟期"
            elif field.maturity == 2.0:
                mature_type = "完熟期"
            else:
                mature_type = "中间阶段"

            # 构建响应数据
            data = {
                'field_id': field.field_id,  # 注意 ForeignKey 字段
                'acreage': field.acreage, 
                'predict_yield': field.predict_yield,
                'maturity': mature_type,
                'description': field.description,
            }
            return json_response(200, '返回成功', data)
        except Exception as e:
            return json_response(400, str(e))
    else:
        return json_response(400, '无效的请求方法')