# Generated by Django 3.2.7 on 2021-09-19 10:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myApp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Member',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=20)),
                ('details', models.CharField(max_length=200)),
            ],
        ),
        migrations.DeleteModel(
            name='Feature',
        ),
    ]
