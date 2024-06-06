from django.db import models

# Create your models here.

#麦田详细信息:麦田ID 面积大小 预测产量 成熟度 说明
class FieldInfo(models.Model):
    field_id = models.AutoField(primary_key=True)
    acreage = models.DecimalField(max_digits=10, decimal_places=2)
    predict_yield = models.DecimalField(max_digits=10, decimal_places=2)
    maturity = models.DecimalField(max_digits=5, decimal_places=2)
    description = models.CharField(max_length=100, null=True, blank=True)

    class Meta:
        db_table = 'fieldinfo'

#历史记录:历史记录ID 所属麦田ID 原始图片数据路径 麦穗识别数量 识别成熟度 图片上传时间 数据类型(0图片,1视频)
class History(models.Model):
    history_id = models.AutoField(primary_key=True)
    field = models.ForeignKey(FieldInfo, on_delete=models.CASCADE)
    file_path = models.CharField(max_length=255)
    wheat_counts = models.IntegerField()
    maturity = models.DecimalField(max_digits=5, decimal_places=2)
    upload_time = models.CharField(max_length=50)
    file_type = models.IntegerField(choices=[(0, 'Picture'), (1, 'Video')], default=0)
    
    class Meta:
        db_table = 'history'