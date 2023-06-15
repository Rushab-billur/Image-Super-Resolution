from django.db import models
import os
import datetime

# Create your models here.

def file_path(request, file_name):
    old_filename = file_name
    timeNow = datetime.datetime.now().strftime('%Y%m%d%H:%M:%S')
    file_name = "%s%s" % (timeNow, old_filename)

    return os.path.join('static/LR_uploads/', file_name)

def file_path_sr(request, file_name):
    old_filename = file_name
    timeNow = datetime.datetime.now().strftime('%Y%m%d%H:%M:%S')
    file_name = "%s%s" % (timeNow, old_filename)

    return os.path.join('static/SR/', file_name)

class Image_table(models.Model):
    lr_image = models.ImageField(upload_to=file_path, null=True)

class Image_table_sr(models.Model):
    sr_image = models.ImageField(upload_to=file_path_sr, null=True)


class LR_Image_table(models.Model):
    lr_image = models.ImageField(upload_to=file_path, null=True)

class SR_Image_table_sr(models.Model):
    sr_image = models.ImageField(upload_to=file_path_sr, null=True)
