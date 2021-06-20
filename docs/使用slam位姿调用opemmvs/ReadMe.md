###                                   ORB-SLAM2同OpenMVS实现三维重建

<img src="image/1624171330898.gif" style="zoom: 150%;" />





### ORB-SLAM2   位姿导出

Note:

为与OpenMVS进行对接本次进对ORB-SLAM2进行部分修改，使之可以为OpenMVS提供稀疏点云、关键帧的位姿、内参，以及稀疏点云在各个View中的可见性。

主要更改如下

- 在Map文件下增添如下函数

```c++
public:    
    void Save(const string &filename,const cv::MatSize image_size);
    void SaveMapPoint(ofstream &f, MapPoint* mp);
    void SaveKeyFrame(ofstream &f, KeyFrame* kf);
protected:
std::vector<int> KeyId;
```

- 在System下增加：

```c++
void System::SaveMap(const string &filename,const cv::MatSize image_size);
```

- 在mono_tum.cc或者orbslam的其他Examples中对 **System::SaveMap(const string &filename,const cv::MatSize image_size)**这个函数进行调用即可。

```c++
SLAM.SaveMap("../Examples/output/sfm.txt",im.size);
```

**OpenMVS接受SLAM的数据格式**

<img src="image/2021-06-20%2015-04-40%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png" style="zoom:80%;" />

**代码实现**

![](image/2021-06-20%2011-25-13%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

![](image/2021-06-20%2011-24-05%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

![](image/2021-06-20%2011-22-51%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)



**ORB-SLAM2**

数据集地址： https://vision.in.tum.de/data/datasets/rgbd-dataset/download#freiburg1_plant

<img src="image/1624165907236.gif" style="zoom:200%;" />

**ORB-SLAM2位姿导出结果**

![](image/2021-06-20%2013-45-01%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

![](image/2021-06-20%2013-45-38%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

**ORB-SLAM2导出位姿验证**

在与OpenMVS 进行对接之前，一定确保自己导出的信息是准确的，可以将相机三位空间坐标点以及相机在空间中的位置保存成ply、obj等三维格式，在meshlab中进行查看，或者如果你用的rgb-d相机的话，同样可以将深度图、rgb图一同投影下来，在meshlab下进行查看

   <img src="image/1624166730211.gif" style="zoom:85%;" /> <img src="image/1624167295997.gif" style="zoom:85%;" />

   <img src="image/1624167593035.gif" style="zoom:85%;" />



#### OPENMVS接口

为与SLAM进行对接，我们加入了read_pose.cpp、read_pose.h这两个c++文件，目的是对SLAM导出的位姿和稀疏点云进行读取，并对OpenMVS进行初始化。

主要核心函数有

```c++
bool load_scene(string file,Scene &scene);
bool read_mvs_pose(string file,MVSPOSE &mvs_pose);
bool save_pointcloud_obj(string name, vector<POINT3F> points,int num_keyframes,RGB color)
```

#### InterfaceColMap接口理解

Scene.image中包含 尺度 分辨率 name 以及相关连的相机(这个platforms里面放置的是相机的内参和位姿)

在platforms类中包含相机的内参 位姿

Camera类中包含相机ID 分辨率 相机内参，通过Read函数对stream进行读取自身参数

Image类这里包含图片ID，外参，相关连的相机，图片name以及稀疏点在当前图片的投影点

Point类中 点的ID，位置，颜色，以及当前点的可见图像id

首先明确一点，我们SLAM的内容要如实的传入到**Scene**这个类中

![](image/2021-06-18%2019-40-14%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

首先将数据传入imgaes中

![](image/2021-06-20%2014-06-00%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

然后将数据传入platforms中

![](image/2021-06-18%2019-42-59%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

以及poses

![](image/2021-06-20%2014-16-19%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

**具体代码实现(节选)**

![](image/2021-06-20%2014-21-08%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

![](image/2021-06-20%2014-21-40%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

![](image/2021-06-20%2014-22-23%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

![](image/2021-06-20%2014-23-08%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)



```c++
// load and estimate a dense point-cloud
#define use_custom_pose
#ifdef use_custom_pose
    if(!load_scene(string(MAKE_PATH_SAFE(OPT::strInputFileName)),scene))
		return EXIT_FAILURE;
#else
	if (!scene.Load(MAKE_PATH_SAFE(OPT::strInputFileName)))
		return EXIT_FAILURE;
#endif
```



**三维重建过程**

<img src="image/1624170770614.gif" style="zoom:80%;" /> <img src="image/1624170983479.gif" style="zoom:80%;" />

​                                 稠密重建                                                                                         mesh重构

<img src="image/1624171137549.gif" style="zoom:80%;" /> <img src="image/1624171330898.gif" style="zoom:80%;" />

​                                 mesh优化                                                                                      纹理贴图



