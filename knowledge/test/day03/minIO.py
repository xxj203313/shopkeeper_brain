from minio import Minio

# 初始化客户端
client = Minio(
    "192.168.10.170:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False  # 是否使用https
)

if not client.bucket_exists("mybucket"):
    client.make_bucket("mybucket")

# 上传文件
client.fput_object(
    bucket_name="mybucket",
    object_name="images/p1.jpg",  # Object名称含路径
    file_path="D:/doc/output/万用表RS-12的使用/auto/images/6ed1c2d2c192fe7422f77d0eb13133a4f4b01ea3e738ce2017bc5df75185e1a0.jpg",  # 上传的本地文件路径
    content_type="image/jpeg"  # MIME类型
)

# 上传后访问地址
url = f"http://192.168.10.170:9000/mybucket/images/p1.jpg"
print(url)