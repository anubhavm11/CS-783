import hashlib
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

print(md5("best.hdf5"))
print(md5("BagOfWords_Features.npy"))
print(md5("BagOfWords_Labels.npy"))
print(md5("ImageName.npy"))
print(md5("Image_Name_resnet.npy"))
print(md5("kmeans.pickle"))