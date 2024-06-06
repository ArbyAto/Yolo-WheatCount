# Generated by Django 4.0 on 2024-05-21 06:53

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='FieldInfo',
            fields=[
                ('field_id', models.AutoField(primary_key=True, serialize=False)),
                ('acreage', models.DecimalField(decimal_places=2, max_digits=10)),
                ('predict_yield', models.DecimalField(decimal_places=2, max_digits=10)),
                ('maturity', models.DecimalField(decimal_places=2, max_digits=5)),
            ],
            options={
                'db_table': 'fieldinfo',
            },
        ),
        migrations.CreateModel(
            name='History',
            fields=[
                ('history_id', models.AutoField(primary_key=True, serialize=False)),
                ('picture_path', models.CharField(max_length=255)),
                ('wheat_counts', models.IntegerField()),
                ('upload_time', models.CharField(max_length=50)),
                ('field', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='app.fieldinfo')),
            ],
            options={
                'db_table': 'history',
            },
        ),
    ]