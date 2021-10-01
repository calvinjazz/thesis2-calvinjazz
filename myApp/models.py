from django.db import models

# Create your models here.
class Member(models.Model):
    name = models.CharField(max_length=20)
    details = models.CharField(max_length=200)

class Files(models.Model):
    filename = models.CharField(max_length=100)
    description = models.CharField(max_length=100)
    file = models.FileField(upload_to='store/files/')

    def __str__(self):
        return self.name